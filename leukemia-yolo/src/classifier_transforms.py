"""RGB transforms shared by training, evaluation and crop inference."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .classifier_config import PREPROCESS_VERSION

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224, train: bool = False):
    if image_size not in (224, 300):
        raise ValueError("image_size must be 224 or 300")
    ops = [T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(10),
                T.ColorJitter(brightness=0.08, contrast=0.08)]
    return T.Compose([*ops, T.ToTensor(), T.Normalize(MEAN, STD)])


class CNMCDataset(Dataset):
    def __init__(self, manifest, image_size: int = 224, train: bool = False):
        self.frame = manifest.reset_index(drop=True)
        self.transform = build_transform(image_size, train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(Path(row.absolute_path)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row.class_id)
