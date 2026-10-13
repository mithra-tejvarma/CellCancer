"""Transactional, deterministic Pascal VOC to single-class YOLO conversion."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

import yaml

from .bccd import file_hash, inspect_dataset, pixel_hash
from .config import ensure_outputs, read_json, run_id, write_json
from .validation import SPLITS, validate_processed, validate_split_sets

CONVERSION_VERSION = 1


def normalize_box(box, width, height) -> list[float]:
    import math
    xmin, ymin, xmax, ymax = box
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in box):
        raise ValueError("Nonfinite coordinates or invalid image dimensions")
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError("Box must be positive and clipped to image bounds before normalization")
    return [(xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
            (xmax - xmin) / width, (ymax - ymin) / height]


def denormalize_box(values, width, height) -> list[float]:
    x, y, w, h = values
    return [(x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height]


def wbc_labels(record: dict) -> list[list[float]]:
    rows = []
    for obj in record["objects"]:
        if obj["class"] == "WBC":
            values = normalize_box(obj["box"], record["width"], record["height"])
            row = [0] + [float(f"{v:.10f}") for v in values]
            if row in rows:
                raise ValueError(f"{record['id']}: duplicate WBC annotation")
            restored = denormalize_box(row[1:], record["width"], record["height"])
            if max(abs(a - b) for a, b in zip(restored, obj["box"])) > 1e-4:
                raise ValueError("YOLO conversion failed round-trip tolerance")
            rows.append(row)
    return rows


def choose_splits(records: list[dict], source: dict, seed: int) -> tuple[dict, str]:
    ids = {r["id"] for r in records}
    positive_ids = {r["id"] for r in records if any(o["class"] == "WBC" for o in r["objects"])}
    groups = defaultdict(list)
    for r in records:
        groups[r["pixel_sha256"]].append(r["id"])
    try:
        splits = {}
        for name in SPLITS:
            lines = source.get(name, [])
            if any(len(line.split()) != 1 for line in lines if line.strip()):
                raise ValueError(f"{name}.txt is not a one-ID-per-line detection split")
            splits[name] = [Path(line.strip()).stem for line in lines if line.strip()]
        validate_split_sets(splits, ids)
        if any(not positive_ids.intersection(splits[name]) for name in SPLITS):
            raise ValueError("An official split has no WBC-positive images")
        membership = {identifier: s for s in SPLITS for identifier in splits[s]}
        if any(len({membership[i] for i in group}) > 1 for group in groups.values()):
            raise ValueError("Official files contain identical image content across splits")
        return splits, "Preserved distinct complete official train/val/test files; trainval ignored."
    except ValueError as exc:
        reason = str(exc)
    if len(groups) < 3:
        raise ValueError("Need at least three distinct image-content groups for disjoint splits")
    rng = random.Random(seed)
    ordered = sorted(groups.values(), key=lambda g: sorted(g))
    positive_groups = [g for g in ordered if positive_ids.intersection(g)]
    if len(positive_groups) < 3:
        raise ValueError("Need at least three distinct WBC-positive image-content groups")
    rng.shuffle(positive_groups)
    seeded = positive_groups[:3]
    remaining_groups = [g for g in ordered if g not in seeded]
    rng.shuffle(remaining_groups)
    n = len(ids)
    val_n = max(1, round(n * .15))
    test_n = max(1, round(n * .15))
    targets = {"train": n - val_n - test_n, "val": val_n, "test": test_n}
    splits = {name: [] for name in SPLITS}
    for name, group in zip(SPLITS, seeded):
        splits[name].extend(group)
    for group in remaining_groups:
        name = max(SPLITS, key=lambda s: targets[s] - len(splits[s]))
        splits[name].extend(group)
    for name in SPLITS:
        splits[name].sort()
    validate_split_sets(splits, ids)
    return splits, f"Seeded 70/15/15 fallback (seed={seed}); identical pixels grouped. Source unusable: {reason}"


def preprocess(config: dict) -> dict:
    inspection = inspect_dataset(config)
    records = inspection["records"]
    for record in records:
        record["sha256"] = file_hash(Path(record["image"]))
        record["pixel_sha256"] = pixel_hash(Path(record["image"]))
        record["xml_sha256"] = file_hash(Path(record["xml"]))
    signature = {"version": CONVERSION_VERSION, "seed": config["seed"],
                 "files": [{k: r[k] for k in ("id", "sha256", "xml_sha256")} for r in records],
                 "source_splits": inspection["source_split_files"]}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    target = config["processed_data_path"]
    output = ensure_outputs(config)
    if target.exists():
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file() or read_json(manifest_path).get("fingerprint") != fingerprint:
            raise ValueError(f"Existing dataset differs or is incomplete: {target}. Use a NEW processed_data_path; existing data preserved.")
        validate_processed(target, output / "validation_report.json")
        manifest = read_json(manifest_path)
        if any(manifest["images"][r["id"]]["labels"] != wbc_labels(r) for r in records):
            raise ValueError("Processed manifest labels disagree with source WBC annotations")
        report = {**manifest["report"], "reused": True}
        for split, identifiers in manifest["splits"].items():
            split_path = output / "splits" / f"{split}.txt"
            if not split_path.is_file():
                split_path.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
        if not (output / "splits" / "manifest.json").is_file():
            write_json(output / "splits" / "manifest.json", manifest)
        write_json(output / "preprocessing_report.json", report)
        return report
    splits, explanation = choose_splits(records, inspection["source_split_files"], config["seed"])
    lookup = {r["id"]: r for r in records}
    manifest_images = {r["id"]: {"filename": Path(r["image"]).name, "sha256": r["sha256"],
                               "pixel_sha256": r["pixel_sha256"], "labels": wbc_labels(r)} for r in records}
    for name in SPLITS:
        if not any(manifest_images[i]["labels"] for i in splits[name]):
            raise ValueError(f"{name} has no WBC-positive images. Review the data/split design before continuing.")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / run_id(f".{target.name}_staging")
    stage.mkdir()
    for split, identifiers in splits.items():
        (stage / "images" / split).mkdir(parents=True)
        (stage / "labels" / split).mkdir(parents=True)
        for identifier in identifiers:
            record = lookup[identifier]
            shutil.copy2(record["image"], stage / "images" / split / Path(record["image"]).name)
            content = "".join("0 " + " ".join(f"{v:.10f}" for v in row[1:]) + "\n"
                              for row in manifest_images[identifier]["labels"])
            (stage / "labels" / split / f"{identifier}.txt").write_text(content, encoding="utf-8")
    data = {"path": str(stage.resolve()), **{s: f"images/{s}" for s in SPLITS}, "names": {0: "WBC"}}
    (stage / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = {"fingerprint": fingerprint, "seed": config["seed"], "split_reason": explanation,
              "splits": {s: len(v) for s, v in splits.items()}, "total_wbc_boxes": inspection["total_wbc_boxes"],
              "source_root": inspection["source_root"],
              "clipped_boxes": inspection["clipped_boxes"], "reused": False,
              "coordinate_policy": "No implicit VOC minus-one adjustment. Clip to [0,width] / [0,height]; use requested continuous-edge formulas.",
              "processed_data_path": str(target)}
    manifest = {"fingerprint": fingerprint, "splits": splits, "images": manifest_images, "report": report}
    write_json(stage / "manifest.json", manifest)
    validate_processed(stage)
    data["path"] = str(target.resolve())
    (stage / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    stage.rename(target)
    for split, identifiers in splits.items():
        (output / "splits" / f"{split}.txt").write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    write_json(output / "splits" / "manifest.json", manifest)
    write_json(output / "preprocessing_report.json", report)
    validate_processed(target, output / "validation_report.json")
    return report


def conversion_table(record: dict):
    import pandas as pd
    rows = []
    for obj in record["objects"]:
        if obj["class"] == "WBC":
            values = normalize_box(obj["box"], record["width"], record["height"])
            restored = denormalize_box(values, record["width"], record["height"])
            rows.append({"image": record["id"], "source_VOC": obj["source_box"], "clipped_VOC": obj["box"],
                         "class_id": 0, "x_center": values[0], "y_center": values[1],
                         "width": values[2], "height": values[3],
                         "roundtrip_max_error_px": max(abs(a-b) for a,b in zip(restored, obj["box"]))})
    return pd.DataFrame(rows)


def main():
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Convert and validate BCCD single-class WBC data")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    print(preprocess(config))
    from .bccd import inspect_dataset
    from .visualization import before_after
    selected = next(r for r in inspect_dataset(config)["records"] if any(o["class"] == "WBC" for o in r["objects"]))
    before_after(selected, config["processed_data_path"], config["output_path"] / "figures" / "preprocessing_examples.png")


if __name__ == "__main__":
    main()
