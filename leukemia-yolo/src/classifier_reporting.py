"""Conference text assembled only from files actually produced by Stage 2."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .classifier_config import classifier_output, latest_classifier_run
from .config import read_json, write_json

LIMITATION = ("Cell-level C-NMC classification on subject-disjoint source folds is not clinical validation. "
              "Grad-CAM is explanatory visualization, not proof of a specific medical feature. "
              "YOLO localization and EfficientNet classification scores are separate; no end-to-end accuracy has been measured.")


def assemble_classifier_report(cfg: dict, run: Path | None = None) -> dict:
    base = classifier_output(cfg)
    report_dir = base / "paper_results"
    get = lambda p: read_json(p) if p.is_file() else None
    inspection = get(base / "dataset_inspection.json")
    split = get(base / "splits" / "summary.json")
    metrics = get(report_dir / "classification_metrics.json")
    if run is None:
        try:
            run = latest_classifier_run(cfg)
        except FileNotFoundError:
            run = None
    run = Path(run) if run else None
    config = get(run / "training_config.json") if run else None
    hardware = get(run / "environment_metadata.json") if run else None
    history = pd.read_csv(run / "training_history.csv") if run and (run / "training_history.csv").exists() else None
    cam = get(base / "gradcam" / "gradcam_summary.json")
    if metrics and run and str((run / "best_model.pth").resolve()) != metrics.get("checkpoint"):
        metrics = None
    if cam and run and str((run / "best_model.pth").resolve()) != cam.get("checkpoint"):
        cam = None
    available = lambda value: value if value is not None else "Not available"
    data = {"source": "C-NMC-2019 (TCIA); Kaggle avk256/cnmc-leukemia",
            "source_images": available(inspection.get("total_images") if inspection else None),
            "source_class_counts": available(inspection.get("class_counts") if inspection else None),
            "source_subjects": available(inspection.get("subject_count") if inspection else None),
            "split_protocol": available(split.get("protocol") if split else None),
            "split_counts": available(split.get("counts") if split else None),
            "architecture": "Torchvision EfficientNet-B3", "pretrained_weights": "EfficientNet_B3_Weights.IMAGENET1K_V1",
            "training_configuration": available(config), "hardware": available(hardware),
            "completed_epochs": len(history) if history is not None else "Not available",
            "training_duration_seconds": float(history.elapsed_seconds.iloc[-1]) if history is not None and len(history) else "Not available",
            "best_validation_macro_f1": float(history.val_macro_f1.max()) if history is not None and len(history) else "Not available",
            "held_out_test_metrics": available(metrics), "gradcam": available(cam),
            "limitations": LIMITATION}
    table = pd.DataFrame([{"item": key, "value": str(value)} for key, value in data.items()])
    table.to_csv(report_dir / "classifier_results_table.csv", index=False)
    lines = ["# C-NMC Stage 2 classification experiment", "", *[f"- **{key.replace('_', ' ').title()}:** {value}" for key, value in data.items()]]
    (report_dir / "classifier_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "classifier_methodology.md").write_text(
        "# Methodology\n\nC-NMC labeled training folds only. fold_2 is held out; fold_0 and fold_1 "
        "are split by stratified subject ID. Class 0 is HEM and class 1 is ALL. "
        "EfficientNet-B3 ImageNet weights, two-phase fine tuning, TRAIN-only class weights, "
        "and validation macro-F1 checkpoint selection are used. The 224-pixel square resize "
        "differs from the pretrained weights' 320-resize/300-center-crop recipe.\n", encoding="utf-8")
    (report_dir / "classifier_limitations.md").write_text("# Limitations\n\n" + LIMITATION + "\n", encoding="utf-8")
    write_json(report_dir / "classifier_experiment_metadata.json", data)
    return data
