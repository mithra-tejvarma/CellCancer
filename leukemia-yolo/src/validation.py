"""Strict label, image, manifest, and duplicate-content validation."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

from PIL import Image
import yaml

from .bccd import IMAGE_SUFFIXES, file_hash, pixel_hash
from .config import read_json, write_json

SPLITS = ("train", "val", "test")


def validate_split_sets(splits: dict, identifiers) -> None:
    expected = set(identifiers)
    seen = set()
    for name in SPLITS:
        values = splits.get(name, [])
        if not values:
            raise ValueError(f"Empty or missing split: {name}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs in {name}")
        if seen.intersection(values):
            raise ValueError(f"Split overlap detected in {name}: {sorted(seen.intersection(values))}")
        seen.update(values)
    if seen != expected:
        raise ValueError(f"Incomplete splits: missing={sorted(expected - seen)}, unknown={sorted(seen - expected)}")


def read_labels(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: nonnumeric label") from exc
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}:{number}: expected five finite values")
        cls, x, y, width, height = values
        if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{number}: invalid WBC class or normalized coordinates")
        tolerance = 2e-8
        if min(x - width / 2, y - height / 2) < -tolerance or max(x + width / 2, y + height / 2) > 1 + tolerance:
            raise ValueError(f"{path}:{number}: box extends outside image")
        if values in rows:
            raise ValueError(f"{path}:{number}: duplicate label")
        rows.append(values)
    return rows


def validate_processed(root: Path, output: Path | None = None) -> dict:
    root = Path(root).resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Preprocess the dataset first: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    base = Path(data.get("path", "."))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    if base != root or data.get("names") != {0: "WBC"}:
        raise ValueError("data.yaml must resolve to this dataset and names: {0: WBC}")
    manifest = read_json(root / "manifest.json")
    validate_split_sets(manifest["splits"], manifest["images"])
    seen_pixels, seen_bytes, rows, sizes = {}, {}, {}, []
    duplicate_images = []
    for split in SPLITS:
        if data.get(split) != f"images/{split}":
            raise ValueError(f"Unexpected YAML path for {split}")
        folder = root / "images" / split
        label_folder = root / "labels" / split
        images = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        stems = [p.stem for p in images]
        if Counter(stems) != Counter(manifest["splits"][split]):
            raise ValueError(f"{split}: images differ from frozen split manifest")
        if {p.stem for p in label_folder.glob("*.txt")} != set(stems):
            raise ValueError(f"{split}: labels and images do not match")
        boxes = empty = 0
        for path in images:
            with Image.open(path) as image:
                image.load()
            labels = read_labels(label_folder / f"{path.stem}.txt")
            expected = manifest["images"][path.stem]
            if expected.get("source_split") is not None and expected["source_split"] != split:
                raise ValueError(f"{path.name}: processed split differs from Kaggle source folder")
            if labels != expected["labels"]:
                raise ValueError(f"{path.name}: labels differ from source-WBC conversion provenance")
            byte_digest, pixel_digest = file_hash(path), pixel_hash(path)
            if byte_digest != expected["sha256"] or pixel_digest != expected["pixel_sha256"]:
                raise ValueError(f"{path.name}: image differs from source manifest")
            for digest, seen in ((byte_digest, seen_bytes), (pixel_digest, seen_pixels)):
                if digest in seen and seen[digest][0] != split:
                    raise ValueError(f"Cross-split duplicate image: {path} and {seen[digest][1]}")
                if digest in seen:
                    duplicate_images.append([str(path), seen[digest][1]])
                seen[digest] = (split, str(path))
            boxes += len(labels)
            empty += not labels
            sizes.extend({"split": split, "width": r[3], "height": r[4]} for r in labels)
        if not boxes:
            raise ValueError(f"{split} contains no positive WBC samples; cannot train/evaluate")
        rows[split] = {"images": len(images), "wbc_boxes": boxes, "empty_labels": empty}
    report = {"ready": True, "dataset_fingerprint": manifest["fingerprint"], "splits": rows,
              "box_dimensions": sizes, "within_split_duplicates": duplicate_images}
    if output:
        write_json(Path(output), report)
    return report


def main():
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Validate processed WBC data and prevent image leakage")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    print(validate_processed(config["processed_data_path"], config["output_path"] / "validation_report.json"))


if __name__ == "__main__":
    main()
