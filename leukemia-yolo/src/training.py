"""Windows-safe training orchestration and observable run provenance."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .config import ensure_outputs, read_json, run_id, write_json
from .environment import nvidia_smi, require_cuda, windows_media_type
from .validation import validate_processed


def checkpoint_for_run(run: Path, name: str = "best.pt") -> Path:
    path = Path(run) / "weights" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path.resolve()


def latest_main_run(config: dict) -> Path:
    pointer = config["output_path"] / "checkpoints" / "latest_main.json"
    if not pointer.is_file():
        raise FileNotFoundError("No completed main run. Finish notebook 04 or specify --run.")
    info = read_json(pointer)
    run = Path(info["run_dir"])
    if not (run / "run_metadata.json").is_file():
        raise FileNotFoundError(f"Run metadata missing: {run}")
    metadata = read_json(run / "run_metadata.json")
    if metadata.get("status") != "completed" or metadata.get("kind") != "main":
        raise ValueError("Latest run is not a completed main training run")
    current = validate_processed(config["processed_data_path"])["dataset_fingerprint"]
    if current != metadata["dataset_fingerprint"]:
        raise ValueError("Checkpoint was trained on a different processed dataset")
    return run


def resolve_checkpoint(config: dict, run: Path | None = None) -> Path:
    selected = Path(run).resolve() if run else latest_main_run(config)
    metadata = read_json(selected / "run_metadata.json")
    current = validate_processed(config["processed_data_path"])["dataset_fingerprint"]
    if metadata.get("status") != "completed" or metadata.get("kind") != "main" or metadata.get("dataset_fingerprint") != current:
        raise ValueError("Selected checkpoint is not a completed main run of this dataset")
    return checkpoint_for_run(selected)


def results_table(run: Path):
    import pandas as pd
    path = Path(run) / "results.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        if "epoch" in frame:
            frame = frame[pd.to_numeric(frame["epoch"], errors="coerce").notna()]
        required = [key for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss",
                                    "metrics/precision(B)", "metrics/recall(B)",
                                    "metrics/mAP50(B)", "metrics/mAP50-95(B)") if key in frame]
        if required:
            frame = frame.dropna(subset=required)
        return frame
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def progress_columns(frame):
    desired = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
               "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
               "metrics/mAP50-95(B)", "time"]
    return frame[[name for name in desired if name in frame.columns]]


def training_readiness(config: dict) -> dict:
    dataset = validate_processed(config["processed_data_path"])
    gpu = require_cuda(config["device"])
    return {"dataset": str(config["processed_data_path"]), "train_images": dataset["splits"]["train"]["images"],
            "fingerprint": dataset["dataset_fingerprint"], "gpu": gpu, "model": config["model"],
            "epochs": config["epochs"], "imgsz": config["imgsz"], "batch": config["batch"],
            "amp": config["amp"], "workers": config["workers"], "output": str(config["output_path"])}


def choose_cache(config: dict) -> str | bool:
    if config["cache"] != "auto":
        return config["cache"]
    image_bytes = sum(p.stat().st_size for p in (config["processed_data_path"] / "images").rglob("*") if p.is_file())
    free = shutil.disk_usage(config["processed_data_path"]).free
    return "disk" if windows_media_type(config["processed_data_path"]).upper() == "SSD" and free > max(2 * 2**30, 4 * image_bytes) else False


def _check_smoke(config: dict, smoke_run: Path, fingerprint: str) -> None:
    info = read_json(Path(smoke_run) / "run_metadata.json")
    if info.get("status") != "completed" or info.get("kind") != "smoke" or info.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Main training requires a successful 1-epoch CUDA smoke run for this dataset")
    checkpoint_for_run(smoke_run, "last.pt")


def run_training(config: dict, *, kind: str, smoke_run: Path | None = None,
                 resume: Path | None = None, allow_batch_fallback: bool = False) -> Path:
    if kind not in ("smoke", "main"):
        raise ValueError("kind must be smoke or main")
    readiness = training_readiness(config)
    if resume:
        if kind != "main":
            raise ValueError("Only a main run can resume")
        last = Path(resume).resolve()
        if last.name != "last.pt" or not last.is_file():
            raise FileNotFoundError("Resume requires an existing last.pt")
        run = last.parent.parent
        info = read_json(run / "run_metadata.json")
        if info.get("kind") != "main" or info.get("dataset_fingerprint") != readiness["fingerprint"]:
            raise ValueError("Resume checkpoint provenance does not match this dataset")
        if info.get("status") == "completed":
            raise ValueError("Run is already complete; use a new experiment to train again")
        original = info.get("effective_config", {})
        for key in ("data", "epochs", "imgsz", "seed"):
            current = (str(config["processed_data_path"] / "data.yaml") if key == "data" else config[key])
            if original.get(key) != current:
                raise ValueError(f"Resume config changed {key}; use original settings or start a new run")
    else:
        if kind == "main":
            if not smoke_run:
                raise ValueError("Pass --smoke-run <successful smoke directory> before main training")
            _check_smoke(config, Path(smoke_run), readiness["fingerprint"])
        run = ensure_outputs(config) / "checkpoints" / run_id(kind)
        run.mkdir(parents=True, exist_ok=False)
        info = {"kind": kind, "status": "running", "dataset_fingerprint": readiness["fingerprint"],
                "pretrained_weights": config["model"], "started_utc": datetime.now(timezone.utc).isoformat(),
                "smoke_run": str(Path(smoke_run).resolve()) if smoke_run else None}
    from ultralytics import YOLO
    import torch
    cache = choose_cache(config)
    effective = {"data": str(config["processed_data_path"] / "data.yaml"),
                 "epochs": 1 if kind == "smoke" else config["epochs"], "imgsz": config["imgsz"],
                 "batch": config["batch"], "device": config["device"], "amp": config["amp"],
                 "workers": config["workers"], "patience": config["patience"], "seed": config["seed"],
                 "plots": True, "save": True, "save_period": 5, "cache": cache,
                 "project": str(run.parent), "name": run.name, "exist_ok": True}
    if resume:
        effective = info["effective_config"]
    info["effective_config"] = effective.copy()
    info["gpu_start"] = readiness["gpu"]
    info["nvidia_smi_start"] = nvidia_smi()
    write_json(run / "run_metadata.json", info)
    write_json(run / "effective_training_config.json", effective)
    print(f"RUN_START={run}", flush=True)
    started = time.monotonic()
    try:
        model = YOLO(str(last) if resume else "yolov8n.pt")
        if model.task != "detect":
            raise ValueError(f"Expected YOLO object detector, got task={model.task}")
        if resume:
            model.train(resume=True)
        else:
            try:
                model.train(**effective)
            except (RuntimeError, ValueError) as exc:
                message = str(exc).lower()
                if not allow_batch_fallback or config["batch"] != 0.75 or not any(
                    token in message for token in ("out of memory", "cuda oom", "autobatch", "auto batch")):
                    raise
                # The first failed run is preserved. A fresh run makes the fallback explicit.
                failed_run = run
                fallback = run.parent / run_id(f"{kind}_batch_fallback")
                fallback.mkdir(exist_ok=False)
                info["status"] = "failed_autobatch_or_oom"
                info["error"] = str(exc)
                write_json(run / "run_metadata.json", info)
                print(f"Auto-batch/OOM failed. Original run preserved at {run}. Retrying batch=4 in {fallback}.", flush=True)
                run = fallback
                effective.update({"batch": 4, "name": run.name})
                info["status"] = "running"
                info["fallback_from"] = str(failed_run)
                info.pop("error", None)
                info["effective_config"] = effective.copy()
                write_json(run / "run_metadata.json", info)
                write_json(run / "effective_training_config.json", effective)
                print(f"RUN_START={run}", flush=True)
                model = YOLO("yolov8n.pt")
                model.train(**effective)
        best = checkpoint_for_run(run)
        last_saved = checkpoint_for_run(run, "last.pt")
        YOLO(str(best))  # checkpoint-load smoke verification
        frame = results_table(run)
        if frame.empty or not (run / "results.png").is_file():
            raise RuntimeError("Training returned without complete results.csv/results.png")
        info.update({"status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "duration_seconds": round(time.monotonic() - started, 2),
                     "completed_epochs": len(frame), "best_checkpoint": str(best),
                     "last_checkpoint": str(last_saved), "actual_batch": getattr(getattr(model, "trainer", None), "batch_size", None),
                     "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(config["device"]) / 2**30,
                     "nvidia_smi_end": nvidia_smi()})
        write_json(run / "run_metadata.json", info)
        if kind == "main":
            write_json(config["output_path"] / "checkpoints" / "latest_main.json", {"run_dir": str(run)})
        return run
    except BaseException as exc:
        info.update({"status": "failed", "error": str(exc), "finished_utc": datetime.now(timezone.utc).isoformat()})
        write_json(run / "run_metadata.json", info)
        raise


def stream_training(config: dict, *, kind: str, smoke_run: Path | None = None, resume: Path | None = None,
                    allow_batch_fallback: bool = False) -> Path:
    """Notebook wrapper: unbuffered child process, live stdout/stderr, saved console log."""
    output = ensure_outputs(config)
    log_path = output / "checkpoints" / f"console_{run_id(kind)}.log"
    command = [sys.executable, "-u", str(config["root"] / "scripts" / "train_yolo.py"),
               "--mode", kind, "--config", str(config["config_path"])]
    if smoke_run:
        command += ["--smoke-run", str(smoke_run)]
    if resume:
        command += ["--resume", str(resume)]
    if allow_batch_fallback:
        command += ["--allow-batch-fallback"]
    with subprocess.Popen(command, cwd=config["root"], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                          bufsize=1, shell=False) as process, log_path.open("w", encoding="utf-8") as log:
        run = None
        active = None
        last_progress_rows = 0
        last_read = 0.0
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if line.startswith("RUN_START="):
                active = Path(line.removeprefix("RUN_START=").strip())
                last_progress_rows = 0
            if active and time.monotonic() - last_read >= 3:
                last_read = time.monotonic()
                frame = progress_columns(results_table(active))
                if len(frame) > last_progress_rows:
                    last_progress_rows = len(frame)
                    latest = f"LATEST_RESULTS: {frame.iloc[-1].to_dict()}\n"
                    print(latest, end="", flush=True)
                    log.write(latest)
                    log.flush()
            if line.startswith("RUN_DIR="):
                run = Path(line.removeprefix("RUN_DIR=").strip())
        code = process.wait()
    if code or run is None:
        raise RuntimeError(f"Training child exited with code {code}; inspect {log_path}")
    metadata_path = run / "run_metadata.json"
    info = read_json(metadata_path)
    info["console_log"] = str(log_path)
    write_json(metadata_path, info)
    return run

