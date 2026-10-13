"""Prediction contract for a single YOLO-derived WBC crop or a directory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import torch

from .classifier_model import load_checkpoint
from .classifier_transforms import build_transform
from .config import write_json


def predict_cell(image: Image.Image | Path | str, model, preprocessing, device, threshold=0.5) -> dict:
    """Return class ID/name and softmax scores; scores are not calibrated clinical probabilities."""
    path = str(image) if not isinstance(image, Image.Image) else None
    if path:
        with Image.open(path) as source:
            tensor = preprocessing(source.convert("RGB"))
    else:
        tensor = preprocessing(image.convert("RGB"))
    model.eval()
    with torch.no_grad():
        normal, blast = torch.softmax(model(tensor.unsqueeze(0).to(device)).float(), dim=1)[0].cpu().tolist()
    label = int(blast >= threshold)
    return {"image_path": path, "predicted_class_id": label,
            "predicted_class": "Leukemic Blast (ALL)" if label else "Normal (HEM)",
            "normal_score": normal, "blast_score": blast, "decision_threshold": threshold}


def predict_paths(checkpoint: Path, source: Path, output: Path, device="cpu") -> pd.DataFrame:
    model, info, _ = load_checkpoint(checkpoint, device)
    transform = build_transform(info["image_size"])
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    paths = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"))
    rows = []
    for path in paths:
        item = predict_cell(path, model, transform, device, info["decision_threshold"])
        item["checkpoint"] = str(Path(checkpoint).resolve())
        rows.append(item)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    existing = output / "classifier_predictions.csv"
    if existing.is_file():
        previous = pd.read_csv(existing)
        if "checkpoint" not in previous or set(previous.checkpoint) != {str(Path(checkpoint).resolve())}:
            raise FileExistsError("Prediction directory contains results from a different checkpoint")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "classifier_predictions.csv", index=False)
    write_json(output / "classifier_predictions.json", rows)
    return frame
