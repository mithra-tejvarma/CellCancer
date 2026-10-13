"""Describe the frozen candidate's test trade-off without changing selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import load_config, write_json


def metrics_from_counts(counts):
    tn, fp, fn, tp = counts
    return {"accuracy": (tn + tp) / (tn + fp + fn + tp),
            "normal_specificity": tn / max(tn + fp, 1),
            "blast_sensitivity": tp / max(tp + fn, 1),
            "balanced_accuracy": (tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2,
            "macro_f1": (2*tn/max(2*tn+fp+fn, 1) + 2*tp/max(2*tp+fp+fn, 1))/2}


def main():
    root = load_config()["output_path"] / "classifier"
    full_run = root / "full_finetune/full_20261013T142418Z_d2b3a8ab"
    trial_root = root / "improvements"
    selection = json.loads((trial_root / "validation_selection.json").read_text())
    if selection["selected_arm"] != "class_weight_trial" or selection["dashboard_updated"]:
        raise ValueError("Unexpected selection/deployment state")
    trial = Path(selection["checkpoint"]).parent
    old_metric = json.loads((full_run / "test_metrics.json").read_text())
    new_metric = json.loads((trial / "qualified_test_metrics.json").read_text())
    old = pd.read_csv(full_run / "review/paired_test_predictions.csv")
    new = pd.read_csv(trial / "qualified_test_predictions.csv")
    merged = old.merge(new, on="relative_path", suffixes=("_old", "_new"), validate="one_to_one")
    if len(merged) != old_metric["n"] or not (merged.class_id_old == merged.class_id_new).all() or not (merged.subject_id_old == merged.subject_id_new).all():
        raise ValueError("Candidate and deployed model predictions are not paired on the same subjects and cells")
    truth = merged.class_id_old.to_numpy()
    per_subject = []
    for subject, group in merged.groupby("subject_id_old"):
        y = group.class_id_old.to_numpy()
        pair = []
        for column, threshold in (("full_blast_score", old_metric["threshold"]),
                                  ("blast_score", new_metric["threshold"])):
            pred = group[column].to_numpy() >= threshold
            pair.append(np.array([((y == 0) & ~pred).sum(), ((y == 0) & pred).sum(),
                                  ((y == 1) & ~pred).sum(), ((y == 1) & pred).sum()]))
        per_subject.append(pair)
    counts = np.asarray(per_subject, dtype=np.int64)
    rng = np.random.default_rng(42)
    draws = {name: [] for name in ("accuracy", "normal_specificity", "blast_sensitivity", "balanced_accuracy", "macro_f1")}
    for _ in range(2000):
        sample = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        before, after = (metrics_from_counts(sample[index]) for index in (0, 1))
        for name in draws:
            draws[name].append(after[name] - before[name])
    intervals = {name: {"lower_95": float(np.quantile(values, .025)),
                        "upper_95": float(np.quantile(values, .975))} for name, values in draws.items()}
    result = {"comparison": "validation-selected unweighted-loss trial versus established full model",
              "test_prior_exposure": True, "selected_thresholds": {"full": old_metric["threshold"], "candidate": new_metric["threshold"]},
              "full": old_metric, "candidate": new_metric, "paired_subject_bootstrap_delta_95_intervals": intervals,
              "deployment_decision": "retain established full model; candidate loses substantial blast sensitivity"}
    write_json(trial_root / "qualified_comparison.json", result)
    lines = ["# Qualified classifier improvement comparison", "",
             "The candidate was selected using development validation before one fold_2 evaluation. Fold_2 was previously examined during earlier experiments, so this is a qualified internal comparison, not an untouched external test.", "",
             "| Previously examined fold_2 | Established full | Unweighted-loss candidate |", "|---|---:|---:|"]
    for name in ("normal_specificity", "blast_sensitivity", "balanced_accuracy", "macro_f1", "roc_auc", "accuracy"):
        lines.append(f"| {name} | {old_metric[name]:.3f} | {new_metric[name]:.3f} |")
    lines += ["", f"Confusion matrices: established TN {old_metric['tn']}, FP {old_metric['fp']}, FN {old_metric['fn']}, TP {old_metric['tp']}; candidate TN {new_metric['tn']}, FP {new_metric['fp']}, FN {new_metric['fn']}, TP {new_metric['tp']}.", "",
              f"Paired subject-bootstrap 95% interval for candidate minus established specificity: {intervals['normal_specificity']['lower_95']:+.3f} to {intervals['normal_specificity']['upper_95']:+.3f}; sensitivity: {intervals['blast_sensitivity']['lower_95']:+.3f} to {intervals['blast_sensitivity']['upper_95']:+.3f}.", "",
              "The candidate improves the normal-cell trade-off but falls below the prespecified 0.95 blast-sensitivity requirement on fold_2. The dashboard stays on the established checkpoint. No further settings should be tuned on fold_2.", ""]
    (trial_root / "qualified_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(trial_root / "qualified_comparison.md")


if __name__ == "__main__":
    main()
