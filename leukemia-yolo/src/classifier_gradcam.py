"""Grad-CAM explanations for actual classifier predictions."""
from __future__ import annotations

from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .classifier_model import load_checkpoint
from .classifier_transforms import build_transform
from .config import read_json, write_json


def cam_for_image(model, image: Image.Image, image_size: int, device, target_class: int):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model.eval()
    rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    display_image = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = build_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    layer = model.features[-1]
    with torch.no_grad():
        activation = model.features(tensor)
    if activation.ndim != 4 or min(activation.shape[-2:]) <= 1:
        raise ValueError("EfficientNet Grad-CAM target has no spatial map")
    # Do not use no_grad here; Grad-CAM needs derivatives of the target logit.
    with GradCAM(model=model, target_layers=[layer]) as cam:
        heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_class)])[0]
    overlay = show_cam_on_image(display_image, heatmap, use_rgb=True)
    return np.asarray(rgb), heatmap, overlay


def generate_gradcam(checkpoint: Path, predictions: Path, output: Path, *, per_category=4, device=None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model, info, _ = load_checkpoint(checkpoint, device)
    frame = pd.read_csv(predictions)
    if not {"absolute_path", "class_id", "predicted_class_id", "blast_score"}.issubset(frame.columns):
        raise ValueError("Expected held-out test_predictions.csv columns")
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if "checkpoint_sha256" not in frame or set(frame.checkpoint_sha256) != {digest}:
        raise ValueError("Prediction CSV belongs to a different classifier checkpoint")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    previous = output / "gradcam_summary.json"
    if previous.exists() and read_json(previous).get("checkpoint") != str(Path(checkpoint).resolve()):
        raise FileExistsError("Grad-CAM output belongs to another checkpoint; choose a new output directory")
    categories = {
        "correct_blast": frame[(frame.class_id == 1) & (frame.predicted_class_id == 1)],
        "correct_normal": frame[(frame.class_id == 0) & (frame.predicted_class_id == 0)],
        "false_positive": frame[(frame.class_id == 0) & (frame.predicted_class_id == 1)],
        "false_negative": frame[(frame.class_id == 1) & (frame.predicted_class_id == 0)],
    }
    records, selected = [], []
    for category, subset in categories.items():
        for local_index, (_, row) in enumerate(subset.head(per_category).iterrows()):
            image_path = Path(row.absolute_path)
            with Image.open(image_path) as image:
                original, heatmap, overlay = cam_for_image(model, image, info["image_size"], device, int(row.predicted_class_id))
            destination = output / f"{category}_{image_path.stem}.png"
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            for ax, content, title in zip(axes, (original, heatmap, overlay), ("Original", "CAM", "Overlay")):
                ax.imshow(content, cmap="jet" if title == "CAM" else None, vmin=0 if title == "CAM" else None,
                          vmax=1 if title == "CAM" else None)
                ax.set_title(title); ax.axis("off")
            fig.suptitle(f"{category.replace('_', ' ')} | blast score {row.blast_score:.3f} | target: predicted class")
            fig.tight_layout(); fig.savefig(destination, dpi=220); plt.close(fig)
            records.append({"category": category, "source": str(image_path), "figure": str(destination),
                            "target_class": int(row.predicted_class_id), "blast_score": float(row.blast_score)})
            if len(selected) < 8 and local_index < 2:
                selected.append((category, overlay))
    if selected:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax, item in zip(axes.flat, selected):
            ax.imshow(item[1]); ax.set_title(item[0].replace("_", " ")); ax.axis("off")
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(output / "gradcam_publication_grid.png", dpi=250); plt.close(fig)
    summary = {"checkpoint": str(Path(checkpoint).resolve()), "target": "predicted class",
               "category_counts": {key: len(value) for key, value in categories.items()},
               "generated": records, "interpretation": "CAM highlights model-influential pixels, not verified cell morphology."}
    write_json(output / "gradcam_summary.json", summary)
    return summary
