"""Paper outputs from observed artifacts only; unavailable fields stay explicit."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from .bccd import file_hash
from .config import ensure_outputs, read_json, write_json
from .training import latest_main_run, results_table

LIMITATION = "The BCCD-based YOLOv8 experiment evaluates leukocyte localization only. Blast classification is a separate future stage trained using leukemia-labeled cell images."


def _optional(path: Path):
    return read_json(path) if path.is_file() else None


def build_summary(config: dict) -> dict:
    output = ensure_outputs(config)
    source = _optional(output / "dataset_inspection.json")
    prep = _optional(output / "preprocessing_report.json")
    environment = _optional(output / "environment_report.json")
    metrics = _optional(output / "paper_results" / "detection_metrics.json")
    prediction = _optional(output / "predictions" / "latest.json")
    try:
        run = latest_main_run(config)
        training = read_json(run / "run_metadata.json")
        checkpoint = run / "weights" / "best.pt"
        checkpoint_hash = file_hash(checkpoint)
    except (FileNotFoundError, ValueError):
        run = None
        training = None
        checkpoint_hash = None
    if prep and training and prep.get("fingerprint") != training.get("dataset_fingerprint"):
        prep = None
        source = None
    if source and prep and source.get("source_root") != prep.get("source_root"):
        source = None
    if metrics and (not training or metrics.get("checkpoint_sha256") != checkpoint_hash or
                    metrics.get("dataset_fingerprint") != training.get("dataset_fingerprint")):
        metrics = None  # no mixing test results from a different checkpoint/dataset
    if prediction and prediction.get("checkpoint_sha256") != checkpoint_hash:
        prediction = None
    def available(value):
        return "Not available" if value is None else value
    checks = environment.get("checks", []) if environment else []
    def check_value(name):
        return next((r["value"] for r in checks if r["check"] == name), None)
    values = {
        "dataset_description": ("Kaggle BCCD blood-smear images with pre-defined train/val/test splits "
                                "and per-image JSON rectangle annotations" if prep and prep.get("source_format") == "kaggle_json"
                                else "BCCD blood-smear microscopy images with Pascal VOC RBC, WBC and platelet boxes"),
        "source_images": available(source.get("total_images") if source else None),
        "source_wbc_annotations": available(source.get("total_wbc_boxes") if source else None),
        "split_train": available(prep.get("splits", {}).get("train") if prep else None),
        "split_val": available(prep.get("splits", {}).get("val") if prep else None),
        "split_test": available(prep.get("splits", {}).get("test") if prep else None),
        "model_variant": "YOLOv8n object detection",
        "pretrained_weights": training.get("pretrained_weights", "Not available") if training else "Not available",
        "actual_training_configuration": available(training.get("effective_config") if training else None),
        "hardware_os": available(check_value("Operating system")),
        "system_ram_gib": available(check_value("System RAM total / available GiB")),
        "gpu_model": available(training.get("gpu_start", {}).get("gpu_name") if training else None),
        "gpu_dedicated_vram_gib": available(training.get("gpu_start", {}).get("dedicated_vram_gib") if training else None),
        "training_duration_seconds": available(training.get("duration_seconds") if training else None),
        "completed_epochs": available(training.get("completed_epochs") if training else None),
        "checkpoint_sha256": available(checkpoint_hash),
        "validation_latest_epoch_metrics": "Not available",
        "test_precision": available(metrics.get("metrics", {}).get("precision") if metrics else None),
        "test_recall": available(metrics.get("metrics", {}).get("recall") if metrics else None),
        "test_mAP50": available(metrics.get("metrics", {}).get("mAP50") if metrics else None),
        "test_mAP50_95": available(metrics.get("metrics", {}).get("mAP50_95") if metrics else None),
        "loss_curves": "Not available", "precision_recall_curve": "Not available",
        "confusion_matrix": "Not available", "example_detections": "Not available",
        "limitations": LIMITATION,
    }
    if run:
        frame = results_table(run)
        if not frame.empty:
            columns = [c for c in ("epoch", "metrics/precision(B)", "metrics/recall(B)",
                                   "metrics/mAP50(B)", "metrics/mAP50-95(B)") if c in frame]
            values["validation_latest_epoch_metrics"] = frame[columns].iloc[-1].to_dict()
            loss_columns = [("train/box_loss", "Box loss"), ("train/cls_loss", "Class loss"),
                            ("train/dfl_loss", "DFL loss")]
            present = [(column, label) for column,label in loss_columns if column in frame]
            if present:
                figure_folder = paper_figure_folder = output / "paper_results" / "figures"
                figure_folder.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8,5))
                for column,label in present:
                    ax.plot(frame["epoch"],frame[column],label=label,linewidth=1.8)
                ax.set_xlabel("Completed epoch")
                ax.set_ylabel("Training loss")
                ax.set_title("YOLOv8n WBC detector training losses")
                ax.legend()
                ax.grid(alpha=.2)
                fig.tight_layout()
                loss_path = paper_figure_folder / "training_loss_curves.png"
                fig.savefig(loss_path,dpi=300)
                plt.close(fig)
                values["loss_curves"] = str(loss_path)
    if metrics:
        fig = Path(metrics["figures"])
        for key,name in (("precision_recall_curve","BoxPR_curve.png"),
                         ("confusion_matrix","confusion_matrix.png")):
            if (fig / name).is_file():
                target = output / "paper_results" / "figures" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fig / name,target)
                values[key] = str(target)
    if prediction:
        folder = Path(prediction["prediction_dir"])
        images = sorted(folder.glob("*_prediction.png"))
        if images:
            targets = []
            example_folder = output / "paper_results" / "figures" / "detection_examples"
            example_folder.mkdir(parents=True, exist_ok=True)
            for image in images[:6]:
                target = example_folder / image.name
                shutil.copy2(image, target)
                targets.append(str(target))
            values["example_detections"] = targets
    paper = output / "paper_results"
    # A compact 300-DPI table stays honest when the real test is pending.
    metric_rows = [("Precision", values["test_precision"]), ("Recall", values["test_recall"]),
                   ("mAP@0.5", values["test_mAP50"]), ("mAP@0.5:0.95", values["test_mAP50_95"])]
    fig, ax = plt.subplots(figsize=(7,2.6))
    ax.axis("off")
    table = ax.table(cellText=[[label,str(value)] for label,value in metric_rows],
                     colLabels=["Held-out test metric", "Observed value"], loc="center",cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1,1.45)
    fig.tight_layout()
    fig.savefig(paper / "detection_metrics_table.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    write_json(paper / "experiment_metadata.json", values)
    with (paper / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("field", "value"))
        writer.writerows((key, value) for key, value in values.items())
    lines = ["# Stage 1: BCCD YOLOv8n WBC localization", "", "| Field | Observed value |", "|---|---|"]
    for key,value in values.items():
        lines.append(f"| {key.replace('_',' ')} | {str(value).replace('|','/')} |")
    lines += ["", LIMITATION, "", "Test metrics are measured on the held-out test split. Validation metrics come from training results.csv."]
    (paper / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values


def summary_table(values: dict):
    return pd.DataFrame([{"field": key.replace("_", " "), "observed value": str(value)} for key,value in values.items()])


def main():
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Export provenance-aware Stage 1 conference results")
    parser.add_argument("--config")
    args = parser.parse_args()
    print(summary_table(build_summary(load_config(args.config))).to_string(index=False))


if __name__ == "__main__":
    main()
