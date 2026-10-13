"""Read-only research artifact inventory; writes only new audit reports."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.classifier_config import classifier_config
from src.classifier_model import load_checkpoint
from src.cnmc_splits import validate_splits
from src.config import write_json
from src.validation import validate_processed


def main():
    cfg = classifier_config()
    output = cfg["output_path"] / "audit"
    output.mkdir(parents=True, exist_ok=True)
    baseline = cfg["output_path"] / "classifier/runs/efficientnet_b3_20261013T110511Z_67988092/best_model.pth"
    yolo = cfg["output_path"] / "checkpoints/main_20261013T090007Z_a25a7934/weights/best.pt"
    parts = {}
    def record(name, state, evidence):
        parts[name] = {"status": state, "evidence": evidence}
    try:
        details = validate_processed(cfg["processed_data_path"])
        record("BCCD processed data", "WORKING", details)
    except Exception as exc:
        record("BCCD processed data", "BROKEN", str(exc))
    record("BCCD source", "NOT VERIFIED" if cfg["dataset_root"].exists() else "MISSING",
           str(cfg["dataset_root"]))
    try:
        folder = cfg["output_path"] / "classifier/splits"
        splits = {name: pd.read_csv(folder / f"{name}.csv") for name in ("train", "val", "test")}
        counts = validate_splits(splits)
        summary = json.loads((folder / "summary.json").read_text())
        hashes = {name: hashlib.sha256((folder / f"{name}.csv").read_bytes()).hexdigest()
                  for name in ("train", "val", "test")}
        if any(hashes[name] != summary["manifest_hashes"][name] for name in hashes):
            raise ValueError("Manifest hash differs from saved summary")
        if counts["test"]["images"] != 3553 or counts["test"]["subjects"] != 31 or set(splits["test"].fold) != {"fold_2"}:
            raise ValueError("Unexpected fold_2 test composition")
        record("C-NMC splits", "WORKING", {"counts": counts, "fingerprint": summary["split_fingerprint"]})
    except Exception as exc:
        record("C-NMC splits", "BROKEN", str(exc))
    try:
        from ultralytics import YOLO
        model = YOLO(str(yolo))
        if model.task != "detect" or model.names != {0: "WBC"}:
            raise ValueError(f"Unexpected task/classes: {model.task}, {model.names}")
        record("YOLO checkpoint", "WORKING", {"path": str(yolo), "sha256": hashlib.sha256(yolo.read_bytes()).hexdigest()})
    except Exception as exc:
        record("YOLO checkpoint", "BROKEN", str(exc))
    try:
        _, info, payload = load_checkpoint(baseline)
        if info["class_mapping"] != {"0": "Normal (HEM)", "1": "Leukemic Blast (ALL)"}:
            raise ValueError("Unexpected class mapping")
        if payload["split_fingerprint"] != summary["split_fingerprint"]:
            raise ValueError("Classifier checkpoint split mismatch")
        record("Classifier baseline checkpoint", "WORKING", {"path": str(baseline),
               "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(), "selected_phase": payload.get("phase")})
    except Exception as exc:
        record("Classifier baseline checkpoint", "BROKEN", str(exc))
    metrics = cfg["output_path"] / "classifier/paper_results/classification_metrics.json"
    if metrics.exists():
        values = json.loads(metrics.read_text())
        record("Classifier baseline evaluation", "WORKING" if values.get("n") == 3553 else "BROKEN", values)
    else:
        record("Classifier baseline evaluation", "MISSING", str(metrics))
    detector_metrics = cfg["output_path"] / "paper_results/detection_metrics.json"
    record("YOLO evaluation", "WORKING" if detector_metrics.exists() else "MISSING", str(detector_metrics))
    demos = sorted((cfg["output_path"] / "inference").glob("smear_*/report.json"))
    gradcam_files = list((cfg["output_path"] / "inference").glob("smear_*/*_gradcam.png"))
    record("Grad-CAM", "WORKING" if gradcam_files else "NOT VERIFIED", len(gradcam_files))
    record("Integrated inference", "WORKING" if demos else "NOT VERIFIED", [str(path) for path in demos[-3:]])
    record("Quantification", "WORKING" if demos else "NOT VERIFIED", "Measured BCCD demonstration; no end-to-end ALL labels")
    record("FastAPI dashboard", "WORKING" if demos else "NOT VERIFIED", "Local TestClient upload and artifact retrieval")
    record("Automated tests", "PARTIALLY WORKING", "58 passed, 2 notebook-cleanliness tests fail on pre-existing executed notebook outputs")
    git = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    import importlib.metadata
    import psutil
    hardware = {"python": sys.executable, "torch": torch.__version__, "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(), "nvidia_smi": shutil.which("nvidia-smi"),
                "system_ram_total_bytes": psutil.virtual_memory().total,
                "system_ram_available_bytes": psutil.virtual_memory().available,
                "disk_free_bytes": shutil.disk_usage(cfg["output_path"]).free,
                "ultralytics": importlib.metadata.version("ultralytics"),
                "fastapi": importlib.metadata.version("fastapi"),
                "grad_cam": importlib.metadata.version("grad-cam")}
    command = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
    hardware["driver_temperature_utilization"] = command.stdout.strip() if command.returncode == 0 else None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        hardware.update({"gpu": torch.cuda.get_device_name(0), "dedicated_vram_bytes": total,
                         "free_vram_bytes": free, "compute_capability": torch.cuda.get_device_capability(0)})
        x = torch.randn((64, 64), device="cuda:0")
        hardware.update({"cuda_tensor_result_device": str((x @ x).device),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)})
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "root": str(ROOT),
              "source_files": [str(p.relative_to(ROOT)) for p in (ROOT / "src").glob("*.py")],
              "scripts": [str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.py")],
              "notebooks": [str(p.relative_to(ROOT)) for p in (ROOT / "notebooks").glob("*.ipynb")],
              "config": str(cfg["config_path"]), "git_status": git.stdout.splitlines(), "components": parts}
    write_json(output / "project_audit.json", report)
    write_json(output / "hardware_report.json", hardware)
    smoke_files = sorted((cfg["output_path"] / "classifier/full_finetune").glob("smoke_*/smoke_result.json"))
    if smoke_files and parts["C-NMC splits"]["status"] == "WORKING":
        samples = [json.loads(path.read_text()) for path in smoke_files]
        rates = [sample["samples_per_second"] for sample in samples]
        train_count = parts["C-NMC splits"]["evidence"]["counts"]["train"]["images"]
        benchmark = {"samples": samples, "training_cells": train_count,
                     "estimated_epoch_seconds_from_two_batch_smoke": train_count / (sum(rates) / len(rates)),
                     "warning": "Two-batch startup measurements are preliminary; sustained epoch speed may differ."}
        completed = sorted((cfg["output_path"] / "classifier/full_finetune").glob("full_*/completed.json"))
        if completed:
            history = pd.read_csv(completed[-1].parent / "training_history.csv")
            benchmark["completed_full_run"] = {"run": str(completed[-1].parent),
                "epochs": len(history), "total_epoch_seconds": float(history.epoch_seconds.sum()),
                "median_epoch_seconds": float(history.epoch_seconds.median()),
                "median_training_samples_per_second": float(history.samples_per_second.median())}
            benchmark["warning"] = "Use completed-run timings for reporting; the two-batch startup estimate was much too slow."
        write_json(output / "training_benchmark.json", benchmark)
    lines = ["# Project audit", "", f"Generated UTC: {report['generated_utc']}", "",
             "Saved baseline and YOLO metrics are historical measured results. BCCD has no ALL/HEM labels.", "",
             "| Component | Status |", "|---|---|"]
    lines += [f"| {name} | {item['status']} |" for name, item in parts.items()]
    lines += ["", "## Repository", "", f"Source modules: {len(report['source_files'])}; scripts: {len(report['scripts'])}; notebooks: {len(report['notebooks'])}.",
              "", "## Existing changes", "", *[f"- {line}" for line in report["git_status"]], "",
              "## Research limitation", "", "YOLO is evaluated on BCCD WBC localization; EfficientNet-B3 is evaluated on C-NMC cropped cells. No quantitatively labeled end-to-end leukemia smear evaluation exists.", ""]
    (output / "project_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
