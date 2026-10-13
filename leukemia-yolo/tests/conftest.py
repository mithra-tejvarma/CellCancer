from pathlib import Path
import json
import pytest
from PIL import Image, ImageDraw
import yaml

from src.config import load_config


@pytest.fixture
def synthetic_bccd(tmp_path):
    root = tmp_path / "BCCD_Dataset" / "BCCD"
    for part in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (root / part).mkdir(parents=True)
    for index in range(12):
        name = f"BloodImage_{index:05d}"
        image = Image.new("RGB", (100, 80), (80+index*8, 40+index*5, 50+index*6))
        drawer = ImageDraw.Draw(image)
        drawer.ellipse((20, 10, 60, 50), fill=(180, 160-index, 140+index))
        image.save(root / "JPEGImages" / f"{name}.jpg")
        # One RBC-only image proves that empty YOLO labels are allowed.
        objects = [("RBC", (2, 2, 13, 16)), ("Platelets", (70, 5, 78, 12))]
        if index != 3:
            objects.append(("WBC", (20, 10, 60, 50)))
        xml_objects = "".join(
            f"<object><name>{label}</name><bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin>"
            f"<xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox></object>" for label, box in objects)
        (root / "Annotations" / f"{name}.xml").write_text(
            f"<annotation><path>C:\\obsolete\\{name}.jpg</path><size><width>999</width><height>999</height></size>"
            f"{xml_objects}</annotation>", encoding="utf-8")
    for split, indexes in (("train", range(8)), ("val", range(8,10)), ("test", range(10,12)),
                           ("trainval", range(10))):
        (root / "ImageSets/Main" / f"{split}.txt").write_text(
            "\n".join(f"BloodImage_{i:05d}" for i in indexes) + "\n", encoding="utf-8")
    overrides = {"dataset_root": str(root.parent), "processed_data_path": str(tmp_path / "processed"),
                 "output_path": str(tmp_path / "outputs")}
    config_path = tmp_path / "config.local.yaml"
    config_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    return root, config_path, load_config(config_path)


@pytest.fixture
def synthetic_kaggle(tmp_path):
    """Small fixture with the exact img/ann, *.jpeg.json and meta.json layout."""
    root = tmp_path / "BCCD_Kaggle"
    class_ids = {"Platelets": 13304, "RBC": 13302, "WBC": 13303}
    (root / "meta.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps({"classes": [
        {"title": name, "id": number, "shape": "rectangle"} for name, number in class_ids.items()]}), encoding="utf-8")
    for index in range(15):
        split = "train" if index < 6 else "val" if index < 9 else "test"
        (root / split / "img").mkdir(parents=True, exist_ok=True)
        (root / split / "ann").mkdir(parents=True, exist_ok=True)
        name = f"BloodImage_{index:05d}.jpeg"
        image = Image.new("RGB", (100, 80), (60+index*7, 20+index*5, 45+index*6))
        ImageDraw.Draw(image).ellipse((20, 10, 60, 50), fill=(170, 80+index, 120))
        image.save(root / split / "img" / name)
        boxes = [("RBC", [[2,2],[13,16]]), ("Platelets", [[70,5],[78,12]])]
        if index != 3:
            boxes.append(("WBC", [[20,10],[60,50]]))
        objects = [{"classId": class_ids[label], "classTitle": label, "geometryType": "rectangle",
                    "points": {"exterior": corners, "interior": []}} for label,corners in boxes]
        (root / split / "ann" / f"{name}.json").write_text(json.dumps({
            "size": {"width": 999, "height": 999}, "objects": objects}), encoding="utf-8")
    overrides = {"dataset_root": str(root), "processed_data_path": str(tmp_path / "processed"),
                 "output_path": str(tmp_path / "outputs")}
    config_path = tmp_path / "config.local.yaml"
    config_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    return root, config_path, load_config(config_path)
