"""Central portable configuration; safe to import without torch."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value, root: Path | None = None) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else ((root or project_root()) / path).resolve()


def load_config(path: str | Path | None = None) -> dict:
    root = project_root()
    config = yaml.safe_load((root / "config.example.yaml").read_text(encoding="utf-8"))
    selected = path or os.environ.get("LEUKEMIA_YOLO_CONFIG")
    local = resolve_path(selected) if selected else root / "config.local.yaml"
    if local.exists():
        overrides = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Configuration must be a YAML mapping")
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(overrides)
    elif selected:
        raise FileNotFoundError(f"Configuration not found: {local}")
    for key in ("dataset_root", "processed_data_path", "output_path"):
        config[key] = resolve_path(config[key])
    if config["model"] != "yolov8n.pt":
        raise ValueError("Stage 1 requires pretrained YOLOv8n detection weights: yolov8n.pt")
    for key in ("epochs", "imgsz", "patience"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("workers", "seed", "device"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer; GPU training never defaults to CPU")
    batch = config["batch"]
    if not ((type(batch) is int and batch > 0) or (type(batch) is float and 0 < batch < 1)):
        raise ValueError("batch must be a positive integer or a GPU memory fraction between 0 and 1")
    for key in ("amp", "crop_wbcs"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("confidence", "iou"):
        if isinstance(config[key], bool) or not 0 < config[key] < 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if config["cache"] not in ("auto", "disk", False):
        raise ValueError("cache must be auto, disk, or false")
    source, processed, output = (config[k] for k in ("dataset_root", "processed_data_path", "output_path"))
    for first, second in ((source, processed), (source, output), (processed, output)):
        if first == second or first in second.parents or second in first.parents:
            raise ValueError("Source, processed, and output directories must not overlap")
    config["root"] = root
    config["config_path"] = local if local.exists() else root / "config.example.yaml"
    return config


def ensure_outputs(config: dict) -> Path:
    output = config["output_path"]
    for directory in ("", "checkpoints", "figures", "predictions", "paper_results", "splits", "wbc_crops"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, default=str, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
