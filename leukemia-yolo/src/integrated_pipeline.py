"""Research-only whole-smear WBC detection, classification and counting."""
from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .classifier_gradcam import cam_for_image
from .classifier_model import load_checkpoint
from .classifier_transforms import build_transform
from .config import run_id, write_json


def digest(path: Path) -> str:
    block = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def decode_image(data: bytes, max_bytes=15 * 1024 * 1024, max_pixels=40_000_000) -> Image.Image:
    if not data or len(data) > max_bytes:
        raise ValueError("Image is empty or exceeds the upload limit")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > max_pixels:
                raise ValueError("Image dimensions exceed the limit")
            image.load()
            return image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or damaged image") from exc


def clip_box(values, width, height, margin=0):
    x1, y1, x2, y2 = map(float, values)
    x1, y1 = max(0, int(x1 - margin)), max(0, int(y1 - margin))
    x2, y2 = min(width, int(x2 + margin)), min(height, int(y2 + margin))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (left[2] - left[0]) * (left[3] - left[1])
    area_b = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(area_a + area_b - intersection, 1)


def unique_boxes(items, threshold=.7):
    """Defensive duplicate filter after Ultralytics NMS, ordered by confidence."""
    kept = []
    for item in sorted(items, key=lambda value: value["detection_confidence"], reverse=True):
        if all(_iou(item["box"], prior["box"]) < threshold for prior in kept):
            kept.append(item)
    return kept


def quantify(cells, total_detected=None):
    normal = sum(cell["label"] == "Normal (HEM)" for cell in cells)
    blast = sum(cell["label"] == "Leukemic Blast (ALL)" for cell in cells)
    uncertain = sum(cell["label"] == "Unclassified" for cell in cells)
    if normal + blast + uncertain != len(cells):
        raise ValueError("Unknown cell label")
    detected = len(cells) if total_detected is None else total_detected
    if detected < len(cells):
        raise ValueError("Detected count cannot be below processed count")
    uncertain += detected - len(cells)
    accepted = normal + blast
    return {"total_detected_wbc": detected, "total_classified_wbc": accepted,
            "predicted_normal": normal, "predicted_blast": blast, "unclassified": uncertain,
            "estimated_blast_fraction_percent": 100 * blast / accepted if accepted else None,
            "fraction_denominator": "accepted classified WBCs in this image"}


class SmearPipeline:
    def __init__(self, detector_checkpoint, classifier_checkpoint, *, device=0, confidence=.25,
                 iou=.7, uncertainty_margin=.05, crop_margin=0, gradcam=False):
        from ultralytics import YOLO
        self.detector_checkpoint = Path(detector_checkpoint).resolve()
        self.classifier_checkpoint = Path(classifier_checkpoint).resolve()
        self.detector = YOLO(str(self.detector_checkpoint))
        if self.detector.task != "detect" or self.detector.names != {0: "WBC"}:
            raise ValueError("Detector checkpoint must have the single WBC class")
        self.device = device
        self.classifier, self.info, _ = load_checkpoint(self.classifier_checkpoint,
                                                          f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.transform = build_transform(self.info["image_size"])
        self.confidence, self.iou = confidence, iou
        self.uncertainty_margin, self.crop_margin = uncertainty_margin, crop_margin
        self.gradcam = gradcam

    def analyze(self, data: bytes, output_root: Path):
        image = decode_image(data)
        source_sha256 = hashlib.sha256(data).hexdigest()
        run = Path(output_root) / run_id("smear")
        run.mkdir(parents=True, exist_ok=False)
        result = self.detector.predict(np.asarray(image), conf=self.confidence, iou=self.iou,
                                       device=self.device, verbose=False, save=False)[0]
        raw = []
        for box in result.boxes:
            if int(box.cls.item()) != 0:
                continue
            region = clip_box(box.xyxy[0].tolist(), image.width, image.height)
            if region:
                raw.append({"box": region, "detection_confidence": float(box.conf.item())})
        detections = unique_boxes(raw)
        cells = []
        crops = []
        for index, detection in enumerate(detections):
            region = clip_box(detection["box"], image.width, image.height, self.crop_margin)
            crop = image.crop(region)
            crop_path = run / f"cell_{index:03d}.png"
            crop.save(crop_path)
            detection["crop_filename"] = crop_path.name
            crops.append(crop)
        if crops:
            tensors = torch.stack([self.transform(crop) for crop in crops]).to(next(self.classifier.parameters()).device)
            self.classifier.eval()
            with torch.no_grad():
                scores = torch.softmax(self.classifier(tensors).float(), dim=1)[:, 1].cpu().tolist()
            threshold = float(self.info["decision_threshold"])
            for index, (detection, crop, score) in enumerate(zip(detections, crops, scores)):
                uncertain = abs(score - threshold) < self.uncertainty_margin or min(crop.size) < 16
                label = "Unclassified" if uncertain else ("Leukemic Blast (ALL)" if score >= threshold else "Normal (HEM)")
                cell = {**detection, "label": label, "blast_score": score,
                        "normal_score": 1 - score, "decision_threshold": threshold}
                if self.gradcam and not uncertain:
                    _, _, overlay = cam_for_image(self.classifier, crop, self.info["image_size"],
                                                   next(self.classifier.parameters()).device, int(score >= threshold))
                    name = f"cell_{index:03d}_gradcam.png"
                    Image.fromarray(overlay).save(run / name)
                    cell["gradcam_filename"] = name
                cells.append(cell)
        summary = quantify(cells)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for cell in cells:
            color = "red" if cell["label"] == "Leukemic Blast (ALL)" else "green" if cell["label"] == "Normal (HEM)" else "orange"
            draw.rectangle(cell["box"], outline=color, width=3)
            draw.text(cell["box"][:2], cell["label"], fill=color)
        annotated.save(run / "annotated.png")
        warnings = ["For research and assistive screening only; not a standalone medical diagnosis.",
                    "BCCD detector and C-NMC classifier have not been quantitatively validated together on leukemia-labeled whole smears.",
                    "Scores are uncalibrated model outputs; blast fraction is limited to accepted WBCs in this image."]
        report = {**summary, "cells": cells, "annotated_filename": "annotated.png", "warnings": warnings,
                  "source_sha256": source_sha256, "execution_utc": datetime.now(timezone.utc).isoformat(),
                  "model_versions": {"detector_checkpoint_sha256": digest(self.detector_checkpoint),
                                     "classifier_checkpoint_sha256": digest(self.classifier_checkpoint),
                                     "preprocessing_version": self.info["preprocessing_version"],
                                     "class_mapping": self.info["class_mapping"]},
                  "thresholds": {"detection": self.confidence, "nms_iou": self.iou,
                                 "classification": self.info["decision_threshold"],
                                 "uncertainty_margin": self.uncertainty_margin}}
        write_json(run / "report.json", report)
        with (run / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["box", "detection_confidence", "crop_filename", "label", "blast_score",
                      "normal_score", "decision_threshold", "gradcam_filename"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(cells)
        return run, report
