"""Cell-level metrics and held-out fold_2 evaluation."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve, classification_report)
from torch.utils.data import DataLoader

from .classifier_config import classifier_output
from .classifier_model import load_checkpoint
from .classifier_transforms import CNMCDataset
from .cnmc_data import inspect_cnmc
from .cnmc_splits import create_or_load_splits, validate_splits
from .config import read_json, write_json


def calculate_metrics(y_true, blast_scores, threshold: float = 0.5) -> dict:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(blast_scores, dtype=float)
    if len(truth) == 0 or len(truth) != len(scores) or not set(truth).issubset({0, 1}):
        raise ValueError("Invalid binary metric inputs")
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    safe = lambda a, b: float(a / b) if b else None
    return {"n": int(len(truth)), "threshold": float(threshold), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "accuracy": float(accuracy_score(truth, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if len(set(truth)) == 2 else None,
            "blast_precision": safe(tp, tp + fp), "blast_sensitivity": safe(tp, tp + fn),
            "normal_specificity": safe(tn, tn + fp),
            "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, scores)) if len(set(truth)) == 2 else None}


def predict_manifest(model, frame: pd.DataFrame, image_size: int, device, batch_size=8, workers=0):
    loader = DataLoader(CNMCDataset(frame, image_size), batch_size=batch_size, num_workers=workers,
                        pin_memory=str(device).startswith("cuda"), shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
    return scores


def _plot_evaluation(truth, scores, metrics: dict, folder: Path):
    pred = (np.asarray(scores) >= metrics["threshold"]).astype(int)
    cm = confusion_matrix(truth, pred, labels=[0, 1])
    for normalized, filename in ((False, "confusion_matrix.png"), (True, "confusion_matrix_normalized.png")):
        view = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalized else cm
        fig, ax = plt.subplots(figsize=(5, 4)); art = ax.imshow(view, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{view[i,j]:.2f}" if normalized else str(view[i,j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Blast"],
               yticklabels=["Normal", "Blast"], xlabel="Predicted", ylabel="Actual")
        fig.colorbar(art, ax=ax); fig.tight_layout(); fig.savefig(folder / filename, dpi=220); plt.close(fig)
    if len(set(truth)) == 2:
        fpr, tpr, _ = roc_curve(truth, scores)
        fig, ax = plt.subplots(); ax.plot(fpr, tpr, label=f"AUC {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray"); ax.legend()
        ax.set(xlabel="False positive rate", ylabel="Blast sensitivity", title="Held-out test ROC")
        fig.tight_layout(); fig.savefig(folder / "roc_curve.png", dpi=220); plt.close(fig)
        precision, recall, _ = precision_recall_curve(truth, scores)
        fig, ax = plt.subplots(); ax.plot(recall, precision)
        ax.set(xlabel="Blast recall", ylabel="Blast precision", title="Held-out test PR")
        fig.tight_layout(); fig.savefig(folder / "pr_curve.png", dpi=220); plt.close(fig)


def _plot_examples(frame: pd.DataFrame, folder: Path):
    categories = [("Correct normal", 0, 0), ("Correct blast", 1, 1),
                  ("False positive", 0, 1), ("False negative", 1, 0)]
    fig, axes = plt.subplots(4, 2, figsize=(8, 14))
    for row_index, (title, truth, prediction) in enumerate(categories):
        subset = frame[(frame.class_id == truth) & (frame.predicted_class_id == prediction)].head(2)
        for ax, (_, item) in zip(axes[row_index], subset.iterrows()):
            with Image.open(item.absolute_path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"{title} | blast score {item.blast_score:.3f}")
            ax.axis("off")
        for ax in axes[row_index][len(subset):]:
            ax.text(0.5, 0.5, f"No {title.lower()} samples", ha="center", va="center")
            ax.axis("off")
    fig.tight_layout(); fig.savefig(folder / "evaluation_examples.png", dpi=220); plt.close(fig)


def evaluate(cfg: dict, checkpoint: Path) -> dict:
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    source, _ = inspect_cnmc(cfg, save=False)
    splits, split_info = create_or_load_splits(cfg, source)
    validate_splits(splits)  # create_or_load_splits already checked source coverage and exclusions
    checkpoint = Path(checkpoint).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    folder = classifier_output(cfg) / "paper_results"
    previous = folder / "classification_metrics.json"
    if previous.exists() and read_json(previous).get("checkpoint_sha256") != digest:
        raise FileExistsError("Paper results belong to another checkpoint; archive them before evaluating a new run")
    model, info, payload = load_checkpoint(checkpoint, device)
    if payload.get("split_fingerprint") != split_info["split_fingerprint"]:
        raise ValueError("Checkpoint was trained with a different C-NMC split manifest")
    frame = splits["test"]
    if set(frame.fold) != {"fold_2"}:
        raise ValueError("Held-out test manifest is not fold_2")
    scores = predict_manifest(model, frame, info["image_size"], device, cfg["classifier_batch_size"], cfg["classifier_workers"])
    metric = calculate_metrics(frame.class_id, scores, info["decision_threshold"])
    metric.update({"checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                   "split_fingerprint": split_info["split_fingerprint"], "test_subjects": int(frame.subject_id.nunique()),
                   "positive_class": "Leukemic Blast (ALL)", "unit": "cell image", "split": "fold_2"})
    predictions = frame[["absolute_path", "relative_path", "subject_id", "class_id"]].copy()
    predictions["blast_score"] = scores
    predictions["predicted_class_id"] = (np.asarray(scores) >= info["decision_threshold"]).astype(int)
    predictions["checkpoint_sha256"] = digest
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    write_json(previous, metric)
    pd.DataFrame([metric]).to_csv(folder / "classification_metrics.csv", index=False)
    pd.DataFrame(classification_report(frame.class_id, predictions.predicted_class_id, labels=[0, 1],
                                       target_names=["Normal (HEM)", "Blast (ALL)"],
                                       zero_division=0, output_dict=True)).T.to_csv(folder / "classification_report.csv")
    _plot_evaluation(frame.class_id.to_numpy(), scores, metric, folder)
    _plot_examples(predictions, folder)
    return metric
