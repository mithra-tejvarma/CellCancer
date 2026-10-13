"""Inspect Kaggle BCCD JSON splits or the original Pascal VOC release."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from defusedxml import ElementTree as ET
from PIL import Image

from .config import ensure_outputs, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {"rbc": "RBC", "wbc": "WBC", "platelets": "Platelets", "platelet": "Platelets"}


def locate_bccd(root: Path) -> Path:
    root = Path(root)
    candidates = (root, root / "BCCD", root / "BCCD_Dataset", *root.glob("*/"))
    for candidate in candidates:
        if (all((candidate / split / part).is_dir() for split in ("train", "val", "test")
                for part in ("img", "ann")) or
                all((candidate / part).is_dir() for part in ("JPEGImages", "Annotations"))):
            return candidate.resolve()
    raise FileNotFoundError(
        f"BCCD layout not found under {root}. Expected train|val|test/img and ann JSON folders, "
        "or JPEGImages and Annotations XML folders.")


def source_format(root: Path) -> str:
    return "kaggle_json" if (root / "train" / "img").is_dir() else "voc_xml"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def _checked_box(raw: list[float], width: int, height: int, annotation: Path, index: int, name: str):
    if not all(math.isfinite(v) for v in raw) or raw[2] <= raw[0] or raw[3] <= raw[1]:
        raise ValueError(f"{annotation.name}: nonfinite or nonpositive box {raw}")
    bounded = [min(max(raw[0], 0), width), min(max(raw[1], 0), height),
               min(max(raw[2], 0), width), min(max(raw[3], 0), height)]
    if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
        raise ValueError(f"{annotation.name}: box outside image after clipping {raw}")
    clip = {"object": index, "class": name, "original": raw, "clipped": bounded} if raw != bounded else None
    return bounded, clip


def parse_voc(xml_path: Path, image_path: Path) -> dict:
    if not Path(xml_path).is_file():
        raise FileNotFoundError(f"Missing annotation: {xml_path}")
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    tree = ET.parse(xml_path).getroot()
    objects, clipped = [], []
    for index, obj in enumerate(tree.findall("object")):
        name = (obj.findtext("name") or "").strip()
        canonical = CLASS_NAMES.get(name.lower())
        if canonical is None:
            raise ValueError(f"{xml_path.name}: unknown object class {name!r}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"{xml_path.name}: object {index} has no bndbox")
        try:
            raw = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{xml_path.name}: invalid box coordinates") from exc
        bounded, clip = _checked_box(raw, width, height, xml_path, index, canonical)
        if clip:
            clipped.append(clip)
        objects.append({"class": canonical, "box": bounded, "source_box": raw})
    return {"id": Path(image_path).stem, "image": str(image_path), "annotation": str(xml_path),
            "annotation_format": "voc_xml", "xml": str(xml_path),
            "width": width, "height": height, "objects": objects, "clipping": clipped,
            "annotation_size": [tree.findtext("size/width"), tree.findtext("size/height")],
            "xml_size": [tree.findtext("size/width"), tree.findtext("size/height")]}


def parse_kaggle_json(annotation_path: Path, image_path: Path, class_ids: dict[int, str]) -> dict:
    """Read Dataset Ninja rectangles: points.exterior=[[xmin,ymin],[xmax,ymax]]."""
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation: {annotation_path}")
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise ValueError(f"{annotation_path.name}: expected a JSON object with an objects array")
    objects, clipped, rejected = [], [], []
    for index, obj in enumerate(data["objects"]):
        if not isinstance(obj, dict) or obj.get("geometryType") != "rectangle":
            raise ValueError(f"{annotation_path.name}: object {index} is not a rectangle")
        class_id = obj.get("classId")
        expected = class_ids.get(class_id)
        name = CLASS_NAMES.get(str(obj.get("classTitle", expected or "")).strip().lower())
        if name is None or (expected is not None and name != expected) or (class_ids and expected is None):
            raise ValueError(f"{annotation_path.name}: object {index} has unknown or mismatched class title/id")
        try:
            corners = obj["points"]["exterior"]
            if len(corners) != 2 or any(len(point) != 2 for point in corners):
                raise ValueError("expected two corner points")
            (x1, y1), (x2, y2) = [[float(value) for value in point] for point in corners]
            raw = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{annotation_path.name}: object {index} has invalid rectangle points") from exc
        try:
            bounded, clip = _checked_box(raw, width, height, annotation_path, index, name)
        except ValueError as exc:
            if name == "WBC":
                raise
            rejected.append({"object": index, "class": name, "source_box": raw, "reason": str(exc)})
            continue
        if clip:
            clipped.append(clip)
        objects.append({"class": name, "box": bounded, "source_box": raw})
    size = data.get("size", {})
    return {"id": image_path.stem, "image": str(image_path), "annotation": str(annotation_path),
            "annotation_format": "kaggle_json", "width": width, "height": height,
            "objects": objects, "clipping": clipped, "rejected_boxes": rejected,
            "annotation_size": [size.get("width"), size.get("height")]}


def source_splits(root: Path) -> dict:
    if source_format(root) == "kaggle_json":
        return {split: [p.stem for p in sorted((root / split / "img").iterdir())
                        if p.suffix.lower() in IMAGE_SUFFIXES]
                for split in ("train", "val", "test")}
    folder = root / "ImageSets" / "Main"
    return {p.stem: p.read_text(encoding="utf-8-sig").splitlines()
            for p in sorted(folder.glob("*.txt"))} if folder.is_dir() else {}


def inspect_dataset(config: dict, strict: bool = True) -> dict:
    root = locate_bccd(config["dataset_root"])
    fmt = source_format(root)
    class_ids = {}
    if fmt == "kaggle_json":
        meta = root / "meta.json"
        if not meta.is_file():
            raise FileNotFoundError(f"Kaggle BCCD meta.json missing: {meta}")
        classes = json.loads(meta.read_text(encoding="utf-8-sig")).get("classes", [])
        class_ids = {item["id"]: CLASS_NAMES[item["title"].lower()] for item in classes}
        if set(class_ids.values()) != {"RBC", "WBC", "Platelets"}:
            raise ValueError(f"Unexpected Kaggle class mapping in {meta}: {class_ids}")
        images = sorted(p for split in ("train", "val", "test")
                        for p in (root / split / "img").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        annotations = [p for split in ("train", "val", "test")
                       for p in (root / split / "ann").glob("*.json")]
    else:
        images = sorted(p for p in (root / "JPEGImages").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        annotations = list((root / "Annotations").glob("*.xml"))
    records, missing, unreadable, invalid = [], [], [], []
    identifiers = [p.stem for p in images]
    for path in images:
        annotation = (root / path.parent.parent.name / "ann" / f"{path.name}.json"
                      if fmt == "kaggle_json" else root / "Annotations" / f"{path.stem}.xml")
        if not annotation.is_file():
            missing.append(path.name)
            continue
        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:
            unreadable.append({"image": path.name, "error": str(exc)})
            continue
        try:
            record = (parse_kaggle_json(annotation, path, class_ids) if fmt == "kaggle_json"
                      else parse_voc(annotation, path))
            record["source_split"] = path.parent.parent.name if fmt == "kaggle_json" else None
            records.append(record)
        except Exception as exc:
            invalid.append({"image": path.name, "error": str(exc)})
    counts = Counter(obj["class"] for record in records for obj in record["objects"])
    rejected_boxes = [{"image": r["id"], "source_split": r["source_split"], **box}
                      for r in records for box in r.get("rejected_boxes", [])]
    raw_counts = counts + Counter(box["class"] for box in rejected_boxes)
    positive = sum(any(obj["class"] == "WBC" for obj in record["objects"]) for record in records)
    expected_annotations = ({str(root / path.parent.parent.name / "ann" / f"{path.name}.json")
                             for path in images} if fmt == "kaggle_json" else
                            {str(root / "Annotations" / f"{path.stem}.xml") for path in images})
    orphan = [str(p.relative_to(root)) for p in annotations if str(p) not in expected_annotations]
    report = {"source_root": str(root), "source_format": fmt, "total_images": len(images),
              "total_annotations": len(annotations), "total_json": len(annotations) if fmt == "kaggle_json" else 0,
              "total_xml": len(annotations) if fmt == "voc_xml" else 0,
              "class_distribution": {k: counts[k] for k in ("RBC", "WBC", "Platelets")},
              "raw_class_distribution": {k: raw_counts[k] for k in ("RBC", "WBC", "Platelets")},
              "total_wbc_boxes": counts["WBC"], "images_with_wbc": positive,
              "images_without_wbc": len(records) - positive, "missing_annotations": missing,
              "missing_xml": missing if fmt == "voc_xml" else [],
              "unreadable_images": unreadable, "invalid_annotations": invalid,
              "invalid_boxes_or_xml": invalid if fmt == "voc_xml" else [],
              "duplicate_stems": [k for k, v in Counter(identifiers).items() if v > 1],
              "orphan_annotations": orphan, "orphan_xml": orphan if fmt == "voc_xml" else [],
              "split_directory_exists": (root / "train").is_dir() if fmt == "kaggle_json" else (root / "ImageSets" / "Main").is_dir(),
              "source_split_files": source_splits(root), "records": records,
              "rejected_non_wbc_boxes": rejected_boxes,
              "clipped_boxes": [{"image": r["id"], **clip} for r in records for clip in r["clipping"]]}
    report["ready"] = bool(images) and not any((missing, unreadable, invalid, report["duplicate_stems"], orphan))
    write_json(ensure_outputs(config) / "dataset_inspection.json", report)
    if strict and not report["ready"]:
        raise ValueError("Source integrity failed. Inspect outputs/dataset_inspection.json before preprocessing.")
    return report


def main():
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Inspect BCCD source files without changing them")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    report = inspect_dataset(config)
    from .visualization import source_gallery, class_chart
    source_gallery(report["records"], config["output_path"] / "figures" / "original_bccd_examples.png")
    class_chart(report["class_distribution"], config["output_path"] / "figures" / "source_class_counts.png")
    print({k: v for k, v in report.items() if k != "records"})


if __name__ == "__main__":
    main()
