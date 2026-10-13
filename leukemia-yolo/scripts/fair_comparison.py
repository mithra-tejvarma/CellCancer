"""Report pre-existing models at common and stored thresholds; no test tuning."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.classifier_evaluation import calculate_metrics
from src.config import load_config, write_json


def main():
    output = load_config()["output_path"]
    full = output / "classifier/full_finetune/full_20261013T142418Z_d2b3a8ab"
    frame = pd.read_csv(full / "review/paired_test_predictions.csv")
    baseline_saved = json.loads((output / "classifier/paper_results/classification_metrics.json").read_text())
    full_saved = json.loads((full / "test_metrics.json").read_text())
    rows = []
    for name, column, stored in (("baseline", "baseline_blast_score", .5),
                                 ("full", "full_blast_score", .05)):
        for policy, threshold in (("common_0.5", .5),
                                  ("stored_original" if name == "baseline" else "stored_validation", stored)):
            metric = calculate_metrics(frame.class_id, frame[column], threshold)
            metric.update({"model": name, "policy": policy, "split": "fold_2",
                           "previously_examined_test": True})
            rows.append(metric)
            if policy != "common_0.5":
                saved = baseline_saved if name == "baseline" else full_saved
                if any(metric[key] != saved[key] for key in ("tn", "fp", "fn", "tp")):
                    raise ValueError(f"{name} stored metrics do not reproduce")
    folder = output / "classifier/improvement_audit"
    pd.DataFrame(rows).to_csv(folder / "test_fair_comparison_reporting_only.csv", index=False)
    write_json(folder / "test_fair_comparison_reporting_only.json", rows)
    print(folder / "test_fair_comparison_reporting_only.csv")


if __name__ == "__main__":
    main()
