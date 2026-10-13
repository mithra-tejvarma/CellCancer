"""Stage 2 settings; the existing Stage 1 configuration remains authoritative."""
from __future__ import annotations

from pathlib import Path

from .config import load_config


CLASS_NAMES = {0: "Normal (HEM)", 1: "Leukemic Blast (ALL)"}
WEIGHTS_ID = "EfficientNet_B3_Weights.IMAGENET1K_V1"
PREPROCESS_VERSION = "cnmc_rgb_square_v1"


def classifier_config(path: str | Path | None = None) -> dict:
    cfg = load_config(path)
    for key in ("classifier_image_size", "classifier_batch_size", "classifier_head_epochs",
                "classifier_fine_epochs", "classifier_patience"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["classifier_image_size"] not in (224, 300):
        raise ValueError("classifier_image_size must be 224 or 300")
    if type(cfg["classifier_workers"]) is not int or cfg["classifier_workers"] < 0:
        raise ValueError("classifier_workers must be nonnegative")
    for key in ("classifier_head_lr", "classifier_fine_lr"):
        if not 0 < float(cfg[key]) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if not 0.15 <= float(cfg["classifier_val_subject_fraction"]) <= 0.2:
        raise ValueError("Validation subject fraction must be 0.15–0.20")
    if (cfg["cnmc_root"] == cfg["output_path"] or cfg["cnmc_root"] in cfg["output_path"].parents
            or cfg["output_path"] in cfg["cnmc_root"].parents):
        raise ValueError("Classifier outputs must not be inside the source dataset")
    return cfg


def classifier_output(cfg: dict) -> Path:
    path = cfg["output_path"] / "classifier"
    for part in ("", "figures", "splits", "runs", "paper_results", "gradcam"):
        (path / part).mkdir(parents=True, exist_ok=True)
    return path


def latest_classifier_run(cfg: dict) -> Path:
    runs = [p for p in (classifier_output(cfg) / "runs").iterdir()
            if p.is_dir() and not p.name.startswith("smoke") and (p / "best_model.pth").is_file()]
    if not runs:
        raise FileNotFoundError("No non-smoke best_model.pth; train EfficientNet-B3 first")
    return max(runs, key=lambda p: (p / "best_model.pth").stat().st_mtime_ns)
