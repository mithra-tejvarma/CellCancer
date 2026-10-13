"""Verify a completed full-backbone run and prepare a paper review artifact.

Test data are evaluated at the saved validation-selected threshold only.
The 0.5 comparison is computed on validation data to diagnose the threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.classifier_config import classifier_config
from src.classifier_evaluation import calculate_metrics, predict_manifest
from src.classifier_model import load_checkpoint, set_trainable
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits
from src.config import write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _macro_f1(tn, fp, fn, tp):
    normal = 2 * tn / max(2 * tn + fp + fn, 1)
    blast = 2 * tp / max(2 * tp + fp + fn, 1)
    return (normal + blast) / 2


def subject_bootstrap(frame, score_columns, thresholds, seed=42, replicates=2000):
    """Resample subjects, retaining every cell from each selected subject."""
    groups = [group for _, group in frame.groupby("subject_id")]
    rng = np.random.default_rng(seed)
    values = {key: {name: [] for name in ("accuracy", "balanced_accuracy", "blast_sensitivity", "normal_specificity", "macro_f1")}
              for key in score_columns}
    for _ in range(replicates):
        selected = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[index] for index in selected], ignore_index=True)
        truth = sample.class_id.to_numpy()
        for key, column in score_columns.items():
            pred = sample[column].to_numpy() >= thresholds[key]
            tn = int(((truth == 0) & ~pred).sum())
            fp = int(((truth == 0) & pred).sum())
            fn = int(((truth == 1) & ~pred).sum())
            tp = int(((truth == 1) & pred).sum())
            if (tn + fp) == 0 or (tp + fn) == 0:
                continue
            metrics = {"accuracy": (tn + tp) / len(sample),
                       "balanced_accuracy": ((tn / (tn + fp)) + (tp / (tp + fn))) / 2,
                       "blast_sensitivity": tp / (tp + fn),
                       "normal_specificity": tn / (tn + fp),
                       "macro_f1": _macro_f1(tn, fp, fn, tp)}
            for name, value in metrics.items():
                values[key][name].append(value)
    return {key: {name: {"lower_95": float(np.quantile(series, .025)),
                         "upper_95": float(np.quantile(series, .975)), "replicates": len(series)}
                  for name, series in metrics.items()} for key, metrics in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = classifier_config()
    config = json.loads((run / "training_config.json").read_text())
    complete = json.loads((run / "completed.json").read_text())
    threshold = json.loads((run / "threshold_config.json").read_text())
    split_saved = json.loads((run / "split_fingerprint.json").read_text())
    saved_val = json.loads((run / "val_metrics.json").read_text())
    saved_test = json.loads((run / "test_metrics.json").read_text())
    if config["fine_tune_mode"] != "full" or complete["fine_tune_mode"] != "full":
        raise ValueError("Experiment is not full backbone fine-tuning")
    source, _ = inspect_cnmc(cfg, save=False)
    splits, split_info = create_or_load_splits(cfg, source)
    if split_info["split_fingerprint"] != split_saved["split_fingerprint"]:
        raise ValueError("Saved split fingerprint differs from source manifests")
    checkpoint = run / "best_model.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, info, payload = load_checkpoint(checkpoint, device)
    if payload["fine_tune_mode"] != "full" or payload["split_fingerprint"] != split_info["split_fingerprint"]:
        raise ValueError("Best checkpoint provenance mismatch")
    if payload["epoch"] != threshold["epoch"] or info["decision_threshold"] != threshold["threshold"]:
        raise ValueError("Best checkpoint and threshold mismatch")
    checkpoint_hash = sha256(checkpoint)
    if saved_val["checkpoint_sha256"] != checkpoint_hash or saved_test["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Metrics refer to a different checkpoint")
    baseline_path = Path(config["init_checkpoint"])
    if sha256(baseline_path) != config["init_sha256"]:
        raise ValueError("Initialization checkpoint changed")
    baseline, _, _ = load_checkpoint(baseline_path, "cpu")
    set_trainable(model, "full")
    if not all(param.requires_grad for param in model.parameters()):
        raise ValueError("Full mode did not unfreeze all parameters")
    changed_features = sum(not torch.equal(value.cpu(), baseline.state_dict()[name])
                           for name, value in model.state_dict().items() if name.startswith("features.") and name.endswith("weight"))
    if changed_features == 0:
        raise ValueError("Backbone feature weights did not change")
    frame_val = splits["val"]
    val_scores = predict_manifest(model, frame_val, info["image_size"], device, config["batch_size"], args.workers)
    measured_val = calculate_metrics(frame_val.class_id, val_scores, threshold["threshold"])
    val_at_half = calculate_metrics(frame_val.class_id, val_scores, .5)
    for key in ("tn", "fp", "fn", "tp"):
        if measured_val[key] != saved_val[key]:
            raise ValueError(f"Validation metric mismatch: {key}")
    frame_test = splits["test"].copy()
    test_scores = predict_manifest(model, frame_test, info["image_size"], device, config["batch_size"], args.workers)
    measured_test = calculate_metrics(frame_test.class_id, test_scores, threshold["threshold"])
    for key in ("tn", "fp", "fn", "tp"):
        if measured_test[key] != saved_test[key]:
            raise ValueError(f"Test metric mismatch: {key}")
    baseline_predictions = pd.read_csv(cfg["output_path"] / "classifier/paper_results/test_predictions.csv")
    base = baseline_predictions.set_index("absolute_path")
    if set(base.index) != set(frame_test.absolute_path):
        raise ValueError("Baseline and full runs have different test images")
    frame_test["full_blast_score"] = test_scores
    frame_test["baseline_blast_score"] = frame_test.absolute_path.map(base.blast_score)
    interval = subject_bootstrap(frame_test, {"baseline": "baseline_blast_score", "full": "full_blast_score"},
                                 {"baseline": .5, "full": threshold["threshold"]})
    # The same sampled subjects enter both models, so uncertainty in the change
    # is paired rather than inferred by comparing two marginal intervals.
    bootstrap_delta = {}
    for name in ("accuracy", "balanced_accuracy", "blast_sensitivity", "normal_specificity", "macro_f1"):
        # subject_bootstrap returns marginal bounds only; recompute paired draws below.
        bootstrap_delta[name] = None
    groups = [group for _, group in frame_test.groupby("subject_id")]
    rng = np.random.default_rng(42)
    deltas = {name: [] for name in bootstrap_delta}
    for _ in range(2000):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        old_m = calculate_metrics(sample.class_id, sample.baseline_blast_score, .5)
        new_m = calculate_metrics(sample.class_id, sample.full_blast_score, threshold["threshold"])
        if old_m["balanced_accuracy"] is None:
            continue
        for name in deltas:
            deltas[name].append(new_m[name] - old_m[name])
    bootstrap_delta = {name: {"lower_95": float(np.quantile(series, .025)),
                              "upper_95": float(np.quantile(series, .975)),
                              "mean": float(np.mean(series))} for name, series in deltas.items()}
    history = pd.read_csv(run / "training_history.csv")
    if len(history) != config["epochs"] or int(history.val_macro_f1.idxmax()) + 1 != payload["epoch"]:
        raise ValueError("History and selected epoch disagree")
    review = run / "review"
    review.mkdir(exist_ok=True)
    frame_test[["relative_path", "subject_id", "class_id", "baseline_blast_score", "full_blast_score"]].to_csv(review / "paired_test_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_loss, label="Training")
    axes[0].plot(history.epoch, history.val_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.val_macro_f1, label="Validation macro F1")
    axes[1].axvline(payload["epoch"], color="gray", linestyle="--", label="Selected epoch")
    axes[1].set(xlabel="Epoch", ylabel="Macro F1", title="Model selection")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(review / "training_curves.png", dpi=200); plt.close(fig)
    checks = {"best_checkpoint_sha256": checkpoint_hash, "selected_epoch": payload["epoch"],
              "completed_epochs": len(history), "changed_backbone_weight_tensors": changed_features,
              "threshold": threshold["threshold"], "threshold_at_search_floor": threshold["threshold"] == .05,
              "val_at_saved_threshold": measured_val, "val_at_0_5": val_at_half,
              "test_recomputed_at_saved_threshold": measured_test,
              "subject_cluster_bootstrap_95_intervals": interval,
              "paired_full_minus_baseline_subject_bootstrap_95_intervals": bootstrap_delta,
              "total_epoch_seconds": float(history.epoch_seconds.sum()),
              "median_epoch_seconds": float(history.epoch_seconds.median()),
              "peak_cuda_allocated_bytes": int(history.peak_allocated_bytes.max()),
              "peak_cuda_reserved_bytes": int(history.peak_reserved_bytes.max())}
    write_json(review / "review_metrics.json", checks)
    lines = ["# Full fine-tuning review", "", f"Checkpoint: `{checkpoint}`", "",
             f"Best epoch: {payload['epoch']} of {len(history)}; all feature blocks trainable; {changed_features} backbone weight tensors changed from baseline.",
             f"Threshold: {threshold['threshold']:.2f}, selected on validation; this is the lower search boundary.", "",
             "| Metric | Baseline fold_2 | Full fold_2 |", "|---|---:|---:|"]
    old = json.loads((cfg["output_path"] / "classifier/paper_results/classification_metrics.json").read_text())
    for name in ("accuracy", "balanced_accuracy", "blast_sensitivity", "normal_specificity", "macro_f1", "roc_auc"):
        lines.append(f"| {name} | {old[name]:.3f} | {measured_test[name]:.3f} |")
    lines += ["", "Both are cell-level internal test results on the same subject-disjoint fold_2 set. The fold_2 results were examined in earlier work. No whole-smear ALL reference labels are available.",
              "", "Do not tune the threshold using fold_2. Scores are uncalibrated; the low validation-selected threshold needs investigation before deployment or clinical claims.", ""]
    (review / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(review)


if __name__ == "__main__":
    main()
