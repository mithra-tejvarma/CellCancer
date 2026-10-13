"""Qualitative C-NMC validation crop gallery without test-set selection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import load_config


def main():
    folder = load_config()["output_path"] / "classifier/improvement_audit"
    frame = pd.read_csv(folder / "validation_errors.csv")
    categories = ["false_positive", "false_negative", "correct"]
    fig, axes = plt.subplots(3, 4, figsize=(11, 9))
    for row, category in enumerate(categories):
        group = frame[frame.error_type == category]
        # Show subjects, not only multiple cells from one subject.
        selected = group.groupby("subject_id", group_keys=False).head(2).head(4)
        for column, (_, item) in enumerate(selected.iterrows()):
            with Image.open(item.absolute_path) as image:
                axes[row, column].imshow(image.convert("RGB"))
            axes[row, column].set_title(f"{category.replace('_', ' ')}\n{item.subject_id} | score {item.blast_score:.3f}")
            axes[row, column].axis("off")
        for column in range(len(selected), 4):
            axes[row, column].axis("off")
    fig.suptitle("Development validation examples; scores are uncalibrated")
    fig.tight_layout(); fig.savefig(folder / "validation_error_gallery.png", dpi=180); plt.close(fig)
    print(folder / "validation_error_gallery.png")


if __name__ == "__main__":
    main()
