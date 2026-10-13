"""WBC prediction and unlabeled crops; no blast/normal class claims."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil

from PIL import Image

from .bccd import file_hash
from .config import ensure_outputs, run_id, write_json
from .environment import require_cuda
from .training import resolve_checkpoint
from .validation import IMAGE_SUFFIXES, validate_processed


def prediction_sources(config: dict, custom: Path | None = None, limit: int = 6) -> list[Path]:
    root = config["processed_data_path"] / "images" / "test"
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if custom:
        custom = Path(custom).expanduser().resolve()
        if not custom.is_file():
            raise FileNotFoundError(custom)
        images.append(custom)
    if len(images) < min(6, limit):
        raise ValueError("Need six held-out examples to display six predictions")
    return images[:limit] + ([custom] if custom and custom not in images[:limit] else [])


def predict(config: dict, *, run: Path | None = None, custom: Path | None = None,
            limit: int = 6, conf: float | None = None, iou: float | None = None,
            crop: bool | None = None) -> dict:
    from ultralytics import YOLO
    validate_processed(config["processed_data_path"])
    require_cuda(config["device"])
    checkpoint = resolve_checkpoint(config, run)
    conf = config["confidence"] if conf is None else conf
    iou = config["iou"] if iou is None else iou
    crop = config["crop_wbcs"] if crop is None else crop
    if not 0 < conf < 1 or not 0 < iou < 1:
        raise ValueError("Confidence and NMS IoU thresholds must be between 0 and 1")
    images = prediction_sources(config, custom, limit)
    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError("Checkpoint is not a detection model")
    output = ensure_outputs(config)
    run_name = run_id("predictions")
    predictions = output / "predictions" / run_name
    crops = output / "wbc_crops" / run_name
    predictions.mkdir(exist_ok=False)
    if crop:
        crops.mkdir(parents=True, exist_ok=False)
    rows, examples = [], []
    for index, path in enumerate(images):
        with Image.open(path) as image:
            original = image.convert("RGB")
        result = model.predict(source=str(path), conf=conf, iou=iou, device=config["device"],
                               verbose=False, save=False)[0]
        annotated = Image.fromarray(result.plot()[..., ::-1])
        output_name = f"{index:03d}_{path.stem}_{file_hash(path)[:8]}_prediction.png"
        annotated.save(predictions / output_name)
        current_count = 0
        for j, box in enumerate(result.boxes):
            if int(box.cls.item()) != 0:
                raise ValueError("Model produced a class other than WBC (class 0)")
            x1,y1,x2,y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            x1,y1 = max(0,x1),max(0,y1)
            x2,y2 = min(original.width,x2),min(original.height,y2)
            if x2 <= x1 or y2 <= y1:
                continue
            current_count += 1
            crop_name = f"{index:03d}_{path.stem}_wbc_{j:03d}.png" if crop else ""
            if crop:
                original.crop((x1,y1,x2,y2)).save(crops / crop_name)
            rows.append({"source_image": str(path), "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                         "confidence": float(box.conf.item()), "crop_filename": crop_name,
                         "model_checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
                         "confidence_threshold": conf, "iou_threshold": iou})
        examples.append({"source": str(path), "prediction": str(predictions / output_name),
                         "detected_wbcs": current_count})
    fields = ("source_image", "xmin", "ymin", "xmax", "ymax", "confidence", "crop_filename",
              "model_checkpoint", "checkpoint_sha256", "confidence_threshold", "iou_threshold")
    metadata = predictions / "prediction_metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(predictions / "prediction_metadata.json", {"examples": examples, "detections": rows})
    shutil.copy2(metadata, output / "prediction_metadata.csv")
    write_json(output / "predictions" / "latest.json", {"prediction_dir": str(predictions),
                                                        "checkpoint_sha256": file_hash(checkpoint)})
    return {"prediction_dir": str(predictions), "crop_dir": str(crops) if crop else None,
            "metadata_csv": str(metadata), "examples": examples, "detections": len(rows)}
