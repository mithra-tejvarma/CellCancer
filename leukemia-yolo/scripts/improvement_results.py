"""Summarize controlled validation-only trials without changing deployment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, write_json


def main():
    output = load_config()["output_path"] / "classifier/improvements"
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((load_config()["output_path"] / "classifier/improvement_audit/validation_threshold_study.json").read_text())
    initial = audit["selected"]
    rows = [{"arm": "existing_full_checkpoint", "epoch": 16,
             "threshold": initial["threshold"], "specificity": initial["normal_specificity"],
             "sensitivity": initial["blast_sensitivity"], "macro_f1": initial["macro_f1"],
             "roc_auc": None, "eligible": True, "run": "existing", "training_seconds": 0}]
    for arm in ("control", "augmentation_trial", "class_weight_trial"):
        completed = sorted((output / arm).glob("trial_*/completed.json"), key=lambda p: p.stat().st_mtime_ns)
        if not completed:
            continue
        run = completed[-1].parent
        config = json.loads((run / "training_config.json").read_text())
        threshold = json.loads((run / "threshold_config.json").read_text())
        history = pd.read_csv(run / "training_history.csv")
        best = history[history.epoch == threshold["epoch"]].iloc[0]
        rows.append({"arm": arm, "epoch": int(best.epoch), "threshold": threshold["threshold"],
                     "specificity": float(best.val_specificity), "sensitivity": float(best.val_sensitivity),
                     "macro_f1": float(best.val_macro_f1), "roc_auc": float(best.val_roc_auc),
                     "eligible": bool(threshold["eligible"]), "run": str(run),
                     "training_seconds": float(history.epoch_seconds.sum()),
                     "peak_allocated_bytes": int(history.peak_allocated_bytes.max()),
                     "init_sha256": config["init_sha256"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "validation_comparison.csv", index=False)
    write_json(output / "validation_comparison.json", rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    frame.set_index("arm")[["specificity", "sensitivity", "macro_f1"]].plot.bar(ax=ax)
    ax.set(ylim=(0, 1), ylabel="Development validation metric", title="Controlled trial comparison")
    ax.axhline(.95, color="gray", linestyle=":")
    fig.tight_layout(); fig.savefig(output / "validation_comparison.png", dpi=200); plt.close(fig)
    lines = ["# Specificity improvement study", "",
             "All thresholds and candidate checkpoints below were selected using development validation only. The previous fold_2 test set was not used to select these trials.", "",
             "| Arm | Threshold | Normal specificity | Blast sensitivity | Macro F1 | Eligible |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['threshold']:.3f} | {row['specificity']:.3f} | {row['sensitivity']:.3f} | {row['macro_f1']:.3f} | {row['eligible']} |")
    lines += ["", "The original full checkpoint remains deployed. Only two normal subjects occur in development validation, so these differences may be unstable. Trial results are exploratory and do not establish fold_2 or whole-smear performance.", "",
              "The augmentation and class-weight arms each start from the same full checkpoint as the unchanged-policy control. All use the same train/validation manifests, image size, learning rate, batch and accumulation settings, and validation objective.", ""]
    (output / "improvement_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "improvement_results.md")


if __name__ == "__main__":
    main()
