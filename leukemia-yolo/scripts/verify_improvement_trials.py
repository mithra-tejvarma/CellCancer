"""Check controlled trial provenance and reproduce selected validation metrics."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.classifier_evaluation import calculate_metrics
from src.classifier_model import load_checkpoint
from src.config import load_config, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = load_config()["output_path"] / "classifier/improvements"
    rows = []
    init_hashes = set()
    for arm in ("control", "augmentation_trial", "class_weight_trial"):
        runs = sorted((root / arm).glob("trial_*/completed.json"))
        if not runs:
            continue
        run = runs[-1].parent
        config = json.loads((run / "training_config.json").read_text())
        threshold = json.loads((run / "threshold_config.json").read_text())
        if config["arm"] != arm or sha(config["init_checkpoint"]) != config["init_sha256"]:
            raise ValueError(f"{arm} initialization provenance mismatch")
        init_hashes.add(config["init_sha256"])
        model, info, checkpoint = load_checkpoint(run / "best_model.pth")
        if checkpoint["trial_arm"] != arm or checkpoint["split_fingerprint"] != config["split_fingerprint"]:
            raise ValueError(f"{arm} checkpoint provenance mismatch")
        if checkpoint["epoch"] != threshold["epoch"] or info["decision_threshold"] != threshold["threshold"]:
            raise ValueError(f"{arm} threshold metadata mismatch")
        predictions = pd.read_csv(run / f"validation_epoch_{threshold['epoch']:02d}.csv")
        measured = calculate_metrics(predictions.class_id, predictions.blast_score, threshold["threshold"])
        expected = checkpoint["validation_metrics"]
        if any(measured[key] != expected[key] for key in ("tn", "fp", "fn", "tp")):
            raise ValueError(f"{arm} selected validation metrics mismatch")
        if (measured["blast_sensitivity"] >= .95) != threshold["eligible"]:
            raise ValueError(f"{arm} eligibility metadata mismatch")
        write_json(run / "verification.json", {"checkpoint_sha256": sha(run / "best_model.pth"),
            "init_sha256": config["init_sha256"], "split_fingerprint": config["split_fingerprint"],
            "selected_epoch": checkpoint["epoch"], "validation_metrics_reproduced": measured,
            "test_set_used": False})
        if not (run / "class_mapping.json").exists():
            write_json(run / "class_mapping.json", info["class_mapping"])
        if not (run / "preprocessing_config.json").exists():
            write_json(run / "preprocessing_config.json", {"image_size": info["image_size"],
                "version": info["preprocessing_version"], "normalization": info["normalization"]})
        if not (run / "split_fingerprint.json").exists():
            write_json(run / "split_fingerprint.json", {"split_fingerprint": config["split_fingerprint"]})
        if not (run / "environment_report.json").exists():
            write_json(run / "environment_report.json", {"python": sys.executable,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "note": "Reconstructed after run; GPU peak is in completed.json"})
        rows.append({"arm": arm, "run": str(run), "epoch": checkpoint["epoch"],
                     "threshold": threshold["threshold"], "normal_specificity": measured["normal_specificity"],
                     "blast_sensitivity": measured["blast_sensitivity"], "macro_f1": measured["macro_f1"]})
    if len(init_hashes) > 1:
        raise ValueError("Controlled arms did not share one initialization checkpoint")
    write_json(root / "trial_verification.json", {"arms": rows, "shared_init_sha256": next(iter(init_hashes), None)})
    print(root / "trial_verification.json")


if __name__ == "__main__":
    main()
