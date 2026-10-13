"""Inspect Pascal VOC using actual image pixels, never embedded XML paths."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

from defusedxml import ElementTree as ET
from PIL import Image

from .config import ensure_outputs, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {"rbc": "RBC", "wbc": "WBC", "platelets": "Platelets", "platelet": "Platelets"}


def locate_bccd(root: Path) -> Path:
    root = Path(root)
    for candidate in (root, root / "BCCD"):
        if all((candidate / part).is_dir() for part in ("JPEGImages", "Annotations")):
            return candidate.resolve()
    raise FileNotFoundError(f"Expected JPEGImages/ and Annotations/ in {root} or {root / 'BCCD'}")


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
        if not all(math.isfinite(v) for v in raw) or raw[2] <= raw[0] or raw[3] <= raw[1]:
            raise ValueError(f"{xml_path.name}: nonfinite or nonpositive box {raw}")
        bounded = [min(max(raw[0], 0), width), min(max(raw[1], 0), height),
                   min(max(raw[2], 0), width), min(max(raw[3], 0), height)]
        if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
            raise ValueError(f"{xml_path.name}: box outside image after clipping {raw}")
        if raw != bounded:
            clipped.append({"object": index, "class": canonical, "original": raw, "clipped": bounded})
        objects.append({"class": canonical, "box": bounded, "source_box": raw})
    return {"id": Path(image_path).stem, "image": str(image_path), "xml": str(xml_path),
            "width": width, "height": height, "objects": objects, "clipping": clipped,
            "xml_size": [tree.findtext("size/width"), tree.findtext("size/height")]}


def source_splits(root: Path) -> dict:
    folder = root / "ImageSets" / "Main"
    return {p.stem: p.read_text(encoding="utf-8-sig").splitlines()
            for p in sorted(folder.glob("*.txt"))} if folder.is_dir() else {}


def inspect_dataset(config: dict, strict: bool = True) -> dict:
    root = locate_bccd(config["dataset_root"])
    images = sorted(p for p in (root / "JPEGImages").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    xmls = list((root / "Annotations").glob("*.xml"))
    records, missing, unreadable, invalid = [], [], [], []
    identifiers = [p.stem for p in images]
    for path in images:
        xml = root / "Annotations" / f"{path.stem}.xml"
        if not xml.is_file():
            missing.append(path.name)
            continue
        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:
            unreadable.append({"image": path.name, "error": str(exc)})
            continue
        try:
            records.append(parse_voc(xml, path))
        except Exception as exc:
            invalid.append({"image": path.name, "error": str(exc)})
    counts = Counter(obj["class"] for record in records for obj in record["objects"])
    positive = sum(any(obj["class"] == "WBC" for obj in record["objects"]) for record in records)
    report = {"source_root": str(root), "total_images": len(images), "total_xml": len(xmls),
              "class_distribution": {k: counts[k] for k in ("RBC", "WBC", "Platelets")},
              "total_wbc_boxes": counts["WBC"], "images_with_wbc": positive,
              "images_without_wbc": len(records) - positive, "missing_xml": missing,
              "unreadable_images": unreadable, "invalid_boxes_or_xml": invalid,
              "duplicate_stems": [k for k, v in Counter(identifiers).items() if v > 1],
              "orphan_xml": [p.name for p in xmls if p.stem not in identifiers],
              "split_directory_exists": (root / "ImageSets" / "Main").is_dir(),
              "source_split_files": source_splits(root), "records": records,
              "clipped_boxes": [{"image": r["id"], **clip} for r in records for clip in r["clipping"]]}
    report["ready"] = bool(images) and not any((missing, unreadable, invalid, report["duplicate_stems"], report["orphan_xml"]))
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
