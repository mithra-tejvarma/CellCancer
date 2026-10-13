"""One qualified fold_2 evaluation after validation-only trial selection.

The selection record is saved before test inference. This script never changes
thresholds or deployment based on the test result and refuses to overwrite one.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.classifier_config import classifier_config
from src.classifier_evaluation import calculate_metrics, predict_manifest, _plot_evaluation
from src.classifier_model import load_checkpoint
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits
from src.config import write_json


def main():
    cfg = classifier_config()
    root = cfg["output_path"] / "classifier/improvements"
    rows = json.loads((root / "validation_comparison.json").read_text())
    verified = json.loads((root / "trial_verification.json").read_text())
    candidates = [row for row in rows if row["arm"] in {"control", "augmentation_trial", "class_weight_trial"} and row["eligible"]]
    if len(candidates) != 3 or len(verified["arms"]) != 3:
        raise ValueError("All three controlled arms must complete and verify before final evaluation")
    winner = max(candidates, key=lambda row: (row["specificity"], row["macro_f1"]))
    run = Path(winner["run"])
    checkpoint = run / "best_model.pth"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selection_path = root / "validation_selection.json"
    selection = {"selected_arm": winner["arm"], "checkpoint": str(checkpoint.resolve()),
                 "checkpoint_sha256": digest, "threshold": winner["threshold"],
                 "objective": "maximize validation specificity subject to blast sensitivity >= 0.95",
                 "validation_specificity": winner["specificity"],
                 "validation_blast_sensitivity": winner["sensitivity"],
                 "source": "development validation only", "dashboard_updated": False}
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("Existing frozen validation selection differs")
    else:
        write_json(selection_path, selection)
    result_path = run / "qualified_test_metrics.json"
    if result_path.exists():
        raise FileExistsError("Qualified test result already exists; preserving prior evaluation")
    source, _ = inspect_cnmc(cfg, save=False)
    splits, split_info = create_or_load_splits(cfg, source)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, info, payload = load_checkpoint(checkpoint, device)
    if payload["split_fingerprint"] != split_info["split_fingerprint"] or info["decision_threshold"] != winner["threshold"]:
        raise ValueError("Selected checkpoint, split, or threshold mismatch")
    frame = splits["test"]
    if set(frame.fold) != {"fold_2"}:
        raise ValueError("Unexpected test split")
    scores = predict_manifest(model, frame, info["image_size"], device, 4, 2)
    metrics = calculate_metrics(frame.class_id, scores, info["decision_threshold"])
    metrics.update({"split": "fold_2", "subjects": int(frame.subject_id.nunique()),
                    "checkpoint_sha256": digest, "split_fingerprint": split_info["split_fingerprint"],
                    "selection_record": str(selection_path),
                    "prior_fold_2_exposure": True,
                    "interpretation": "Qualified internal comparison only; fold_2 was examined in earlier work."})
    predictions = frame[["relative_path", "subject_id", "class_id"]].copy()
    predictions["blast_score"] = scores
    predictions.to_csv(run / "qualified_test_predictions.csv", index=False)
    write_json(result_path, metrics)
    figures = run / "qualified_test_figures"
    figures.mkdir(exist_ok=True)
    _plot_evaluation(frame.class_id.to_numpy(), scores, metrics, figures, "Previously examined fold_2")
    print(json.dumps({"selected_arm": winner["arm"], "metrics": str(result_path),
                      "completed_utc": datetime.now(timezone.utc).isoformat()}))


if __name__ == "__main__":
    main()
