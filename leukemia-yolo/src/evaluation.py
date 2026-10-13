"""Held-out test evaluation with checkpoint and dataset provenance."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

from .bccd import file_hash
from .config import ensure_outputs, read_json, run_id, write_json
from .environment import require_cuda
from .training import resolve_checkpoint
from .validation import validate_processed


def evaluate(config: dict, run: Path | None = None) -> dict:
    from ultralytics import YOLO
    dataset = validate_processed(config["processed_data_path"])
    require_cuda(config["device"])
    checkpoint = resolve_checkpoint(config, run)
    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError("Checkpoint is not a detection model")
    output = ensure_outputs(config)
    figure_root = output / "figures" / "test_evaluation"
    name = run_id("test")
    metrics = model.val(data=str(config["processed_data_path"] / "data.yaml"), split="test",
                        imgsz=config["imgsz"], device=config["device"], plots=True,
                        project=str(figure_root), name=name, exist_ok=False)
    evaluation_dir = figure_root / name
    values = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}
    report = {"task": "WBC localization", "split": "test", "test_images": dataset["splits"]["test"]["images"],
              "metrics": values, "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
              "dataset_fingerprint": dataset["dataset_fingerprint"],
              "evaluated_utc": datetime.now(timezone.utc).isoformat(), "figures": str(evaluation_dir),
              "training_run": str(checkpoint.parent.parent)}
    paper = output / "paper_results"
    snapshot = paper / name
    snapshot.mkdir(exist_ok=False)
    write_json(snapshot / "detection_metrics.json", report)
    with (snapshot / "detection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)
    # Stable files intentionally describe only the latest complete evaluation;
    # timestamped copies preserve older results and provenance.
    shutil.copy2(snapshot / "detection_metrics.json", paper / "detection_metrics.json")
    shutil.copy2(snapshot / "detection_metrics.csv", paper / "detection_metrics.csv")
    return report


def evaluation_figure_paths(report: dict) -> dict:
    root = Path(report["figures"])
    names = ("confusion_matrix.png", "BoxPR_curve.png", "BoxF1_curve.png")
    return {name: root / name for name in names}
