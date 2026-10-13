"""Development-only classifier error and threshold audit; never selects on fold_2."""
from __future__ import annotations

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
from PIL import Image
import torch

from src.classifier_config import classifier_config
from src.classifier_evaluation import calculate_metrics, predict_manifest
from src.classifier_gradcam import cam_for_image
from src.classifier_model import load_checkpoint
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits
from src.config import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_table(truth, scores):
    # Dense near zero because the old selection hit the 0.05 lower boundary.
    grid = np.unique(np.r_[np.linspace(.001, .1, 200), np.linspace(.105, .995, 179), .05, .5])
    rows = []
    for threshold in grid:
        metric = calculate_metrics(truth, scores, float(threshold))
        metric["predicted_blast_fraction"] = (metric["tp"] + metric["fp"]) / metric["n"]
        rows.append(metric)
    return pd.DataFrame(rows)


def select_operating_point(table):
    eligible = table[table.blast_sensitivity >= .95]
    if eligible.empty:
        return None
    row = eligible.sort_values(["normal_specificity", "blast_sensitivity", "threshold"],
                               ascending=[False, False, False]).iloc[0]
    return {key: float(row[key]) for key in ("threshold", "blast_sensitivity", "normal_specificity", "blast_precision", "macro_f1", "balanced_accuracy", "predicted_blast_fraction")}


def subject_table(frame, threshold):
    rows = []
    for subject, group in frame.groupby("subject_id"):
        metric = calculate_metrics(group.class_id, group.blast_score, threshold)
        rows.append({"subject_id": subject, "class_id": int(group.class_id.iloc[0]),
                     "cells": len(group), "tn": metric["tn"], "fp": metric["fp"],
                     "fn": metric["fn"], "tp": metric["tp"],
                     "normal_specificity": metric["normal_specificity"],
                     "blast_sensitivity": metric["blast_sensitivity"],
                     "mean_blast_score": float(group.blast_score.mean()),
                     "mean_red": float(group.mean_red.mean()),
                     "mean_green": float(group.mean_green.mean()),
                     "mean_blue": float(group.mean_blue.mean()),
                     "foreground_fraction": float(group.foreground_fraction.mean()),
                     "foreground_red": float(group.foreground_red.mean()),
                     "foreground_green": float(group.foreground_green.mean()),
                     "foreground_blue": float(group.foreground_blue.mean()),
                     "median_width": float(group.width.median()), "median_height": float(group.height.median())})
    return pd.DataFrame(rows)


def image_properties(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        small = np.asarray(rgb.resize((32, 32)), dtype=np.float32)
    foreground = small[small.max(axis=2) > 30]
    foreground_mean = foreground.mean(axis=0) if len(foreground) else np.array([np.nan] * 3)
    return width, height, *small.mean(axis=(0, 1)).tolist(), len(foreground) / 1024, *foreground_mean.tolist()


def main():
    cfg = classifier_config()
    output = cfg["output_path"] / "classifier/improvement_audit"
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = cfg["output_path"] / "classifier/runs/efficientnet_b3_20261013T110511Z_67988092/best_model.pth"
    full_path = cfg["output_path"] / "classifier/full_finetune/full_20261013T142418Z_d2b3a8ab/best_model.pth"
    source, _ = inspect_cnmc(cfg, save=False)
    splits, info = create_or_load_splits(cfg, source)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = {}
    for name, path in (("baseline", baseline_path), ("full", full_path)):
        model, meta, payload = load_checkpoint(path, device)
        if payload["split_fingerprint"] != info["split_fingerprint"]:
            raise ValueError(f"{name} checkpoint and split fingerprint mismatch")
        models[name] = (model, meta, sha(path))
    # Reconstruct development predictions; never run a threshold search on fold_2.
    frames = {}
    for split in ("train", "val"):
        frame = splits[split][["absolute_path", "relative_path", "subject_id", "class_id", "fold"]].copy()
        for name, (model, meta, _) in models.items():
            frame[f"{name}_blast_score"] = predict_manifest(model, frame, meta["image_size"], device, 8, 2)
        if split == "val":
            properties = [image_properties(path) for path in frame.absolute_path]
            frame[["width", "height", "mean_red", "mean_green", "mean_blue",
                   "foreground_fraction", "foreground_red", "foreground_green", "foreground_blue"]] = properties
        frame.to_csv(output / f"{split}_predictions.csv", index=False)
        frames[split] = frame
    val = frames["val"].copy()
    val["blast_score"] = val.full_blast_score
    full_threshold = models["full"][1]["decision_threshold"]
    val["predicted_class_id"] = (val.blast_score >= full_threshold).astype(int)
    val["error_type"] = np.select([
        (val.class_id == 0) & (val.predicted_class_id == 1),
        (val.class_id == 1) & (val.predicted_class_id == 0)], ["false_positive", "false_negative"], default="correct")
    val.to_csv(output / "validation_errors.csv", index=False)
    subjects = subject_table(val, full_threshold)
    subjects.to_csv(output / "validation_subjects.csv", index=False)
    train = frames["train"].copy()
    train["blast_score"] = train.full_blast_score
    train_subjects = []
    for subject, group in train.groupby("subject_id"):
        metric = calculate_metrics(group.class_id, group.blast_score, full_threshold)
        train_subjects.append({"subject_id": subject, "class_id": int(group.class_id.iloc[0]),
                               "cells": len(group), "fp": metric["fp"], "fn": metric["fn"],
                               "specificity": metric["normal_specificity"], "sensitivity": metric["blast_sensitivity"]})
    pd.DataFrame(train_subjects).to_csv(output / "training_subjects.csv", index=False)
    table = threshold_table(val.class_id, val.full_blast_score)
    table.to_csv(output / "validation_threshold_curve.csv", index=False)
    selected = select_operating_point(table)
    macro = table.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
    chosen = {"objective": "maximize validation normal specificity subject to blast sensitivity >= 0.95",
              "selected": selected, "eligible": selected is not None,
              "macro_f1_best": {key: float(macro[key]) for key in ("threshold", "macro_f1", "blast_sensitivity", "normal_specificity")},
              "source": "saved subject-disjoint development validation only", "scores_calibrated": False,
              "checkpoint_sha256": models["full"][2], "split_fingerprint": info["split_fingerprint"]}
    write_json(output / "validation_threshold_study.json", chosen)
    comparison = []
    for name in ("baseline", "full"):
        for policy, threshold in (("common_0.5", .5), ("stored", models[name][1]["decision_threshold"])):
            metric = calculate_metrics(val.class_id, val[f"{name}_blast_score"], threshold)
            comparison.append({"model": name, "policy": policy, **metric})
    pd.DataFrame(comparison).to_csv(output / "validation_model_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, group in val.groupby("class_id"):
        axes[0].hist(group.full_blast_score, bins=np.linspace(0, 1, 41), alpha=.6,
                     label="Normal" if label == 0 else "Blast")
    axes[0].axvline(full_threshold, color="black", linestyle="--", label="Stored threshold")
    axes[0].set(xlabel="Uncalibrated blast score", ylabel="Validation cells", title="Validation score distribution")
    axes[0].legend()
    axes[1].plot(table.threshold, table.normal_specificity, label="Specificity")
    axes[1].plot(table.threshold, table.blast_sensitivity, label="Blast sensitivity")
    axes[1].plot(table.threshold, table.macro_f1, label="Macro F1")
    axes[1].axhline(.95, color="gray", linestyle=":")
    axes[1].set(xlabel="Blast-score threshold", ylabel="Validation metric", title="Threshold trade-off")
    axes[1].legend(); fig.tight_layout(); fig.savefig(output / "validation_threshold_analysis.png", dpi=200); plt.close(fig)
    # Error categories are selected on validation, not on the old test set.
    gradcam_rows = []
    model, meta, _ = models["full"]
    for category in ("false_positive", "false_negative", "correct"):
        subset = val[val.error_type == category].head(2)
        for index, row in enumerate(subset.itertuples()):
            with Image.open(row.absolute_path) as image:
                _, _, overlay = cam_for_image(model, image, meta["image_size"], device, int(row.predicted_class_id))
            filename = f"gradcam_{category}_{index}.png"
            Image.fromarray(overlay).save(output / filename)
            gradcam_rows.append({"category": category, "source": row.relative_path,
                                 "blast_score": row.blast_score, "overlay": filename})
    pd.DataFrame(gradcam_rows).to_csv(output / "validation_gradcam_examples.csv", index=False)
    report = {"split_fingerprint": info["split_fingerprint"],
              "checkpoint_sha256": {name: entry[2] for name, entry in models.items()},
              "counts": info["counts"], "train_class_weights": [4667/(2*1600), 4667/(2*3067)],
              "sampler": "shuffled DataLoader; no weighted sampler",
              "augmentation": "horizontal flip, rotation ±10 degrees, brightness/contrast ±8%; validation deterministic",
              "batchnorm": "running statistics frozen; affine parameters trainable in full mode",
              "validation_normal_subjects": int((subjects.class_id == 0).sum()),
              "validation_subject_errors": subjects[["subject_id", "cells", "fp", "fn", "normal_specificity", "blast_sensitivity"]].replace({np.nan: None}).to_dict("records"),
              "threshold_study": chosen}
    write_json(output / "audit.json", report)
    print(output)


if __name__ == "__main__":
    main()
