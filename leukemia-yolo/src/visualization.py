"""Source and processed bounding-box plots; no model inference here."""
from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from .preprocessing import denormalize_box
from .validation import SPLITS, read_labels

COLORS = {"RBC": "#ed503b", "WBC": "#35c86b", "Platelets": "#3b82f6"}


def source_gallery(records: list[dict], output: Path, count: int = 9):
    selected = records[:count]
    if not selected:
        raise ValueError("No source images to plot")
    cols = 3
    fig, axes = plt.subplots((len(selected)+cols-1)//cols, cols, figsize=(15, 5*((len(selected)+cols-1)//cols)))
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, selected):
        with Image.open(record["image"]) as image:
            ax.imshow(image.convert("RGB"))
        for obj in record["objects"]:
            x1,y1,x2,y2 = obj["box"]
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=COLORS[obj["class"]],linewidth=1.7))
        ax.set_title(record["id"], fontsize=10)
    fig.suptitle("Original BCCD VOC boxes: RBC red, WBC green, platelets blue", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def class_chart(counts: dict, output: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    names = ["RBC", "WBC", "Platelets"]
    ax.bar(names, [counts.get(name, 0) for name in names], color=[COLORS[name] for name in names])
    ax.set_ylabel("Annotated boxes")
    ax.set_title("BCCD source annotation distribution")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    return fig


def processed_gallery(root: Path, output: Path, count: int = 9, seed: int = 42):
    images = [p for split in SPLITS for p in (root / "images" / split).glob("*") if p.is_file()]
    if len(images) < count:
        raise ValueError(f"Need at least {count} processed images; found {len(images)}")
    selected = random.Random(seed).sample(sorted(images), count)
    fig, axes = plt.subplots(3, 3, figsize=(15,15))
    for ax, path in zip(axes.flat, selected):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width,height = rgb.size
            ax.imshow(rgb)
        for _, x,y,w,h in read_labels(root / "labels" / path.parent.name / f"{path.stem}.txt"):
            x1,y1,x2,y2 = denormalize_box((x,y,w,h),width,height)
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=COLORS["WBC"],linewidth=2))
        ax.set_title(f"{path.parent.name}: {path.stem}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Single-class WBC labels", fontsize=16)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def box_distribution(dimensions: list[dict], output: Path):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax,key in zip(axes,("width","height")):
        ax.hist([r[key] for r in dimensions],bins=20,color="#35c86b")
        ax.set_xlabel(f"Normalized box {key}")
        ax.set_ylabel("WBC boxes")
    fig.suptitle("WBC box dimensions")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=200)
    return fig


def before_after(record: dict, processed_root: Path, output: Path):
    from .validation import SPLITS
    found = [p for split in SPLITS for p in (processed_root / "images" / split).glob(f"{record['id']}.*")]
    if not found:
        raise FileNotFoundError(record["id"])
    image_path = found[0]
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width,height = rgb.size
        fig, axes = plt.subplots(1,2,figsize=(13,6))
        for ax in axes:
            ax.imshow(rgb)
            ax.axis("off")
    for obj in record["objects"]:
        x1,y1,x2,y2 = obj["box"]
        axes[0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=COLORS[obj["class"]],linewidth=2))
    for _,x,y,w,h in read_labels(processed_root / "labels" / image_path.parent.name / f"{record['id']}.txt"):
        x1,y1,x2,y2 = denormalize_box((x,y,w,h),width,height)
        axes[1].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=COLORS["WBC"],linewidth=2))
    axes[0].set_title("Source VOC: all blood-cell classes")
    axes[1].set_title("YOLO: WBC only")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=200)
    return fig


def main():
    import argparse
    from .config import load_config
    from .validation import validate_processed
    parser = argparse.ArgumentParser(description="Validate and plot processed WBC images")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    report = validate_processed(config["processed_data_path"], config["output_path"] / "validation_report.json")
    processed_gallery(config["processed_data_path"], config["output_path"] / "figures" / "processed_wbc_examples.png")
    box_distribution(report["box_dimensions"], config["output_path"] / "figures" / "wbc_box_dimensions.png")
    print(report["splits"])


if __name__ == "__main__":
    main()
