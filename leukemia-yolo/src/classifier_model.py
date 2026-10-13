"""Torchvision EfficientNet-B3 with an explicit two-logit classifier."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from .classifier_config import CLASS_NAMES, PREPROCESS_VERSION, WEIGHTS_ID
from .classifier_transforms import MEAN, STD


def build_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    return model


def set_trainable(model: nn.Module, phase: str) -> None:
    if phase not in ("head", "fine", "partial", "full"):
        raise ValueError("phase must be head, fine, partial or full")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    if phase == "full":
        for param in model.features.parameters():
            param.requires_grad = True
    elif phase in ("fine", "partial"):
        for block in model.features[-2:]:
            for param in block.parameters():
                param.requires_grad = True


def metadata(image_size: int, threshold: float = 0.5) -> dict:
    return {"architecture": "torchvision.efficientnet_b3", "weights": WEIGHTS_ID,
            "class_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
            "image_size": image_size, "normalization": {"mean": MEAN, "std": STD},
            "preprocessing_version": PREPROCESS_VERSION, "decision_threshold": threshold,
            "positive_class": 1}


def load_checkpoint(path: Path, device: str | torch.device = "cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint missing: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    info = data.get("metadata", {})
    if (info.get("architecture") != "torchvision.efficientnet_b3" or
            info.get("class_mapping") != {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]} or
            info.get("preprocessing_version") != PREPROCESS_VERSION or
            info.get("normalization") != {"mean": MEAN, "std": STD}):
        raise ValueError("Checkpoint architecture, class mapping or preprocessing is incompatible")
    model = build_model(pretrained=False)
    model.load_state_dict(data["model_state"])
    model.to(device).eval()
    return model, info, data
