import json

import pytest

from src.bccd import inspect_dataset, locate_bccd, parse_kaggle_json
from src.preprocessing import preprocess
from src.validation import validate_processed


CLASS_IDS = {13302: "RBC", 13303: "WBC", 13304: "Platelets"}


def test_real_kaggle_layout_and_json_conversion(synthetic_kaggle):
    root, _, config = synthetic_kaggle
    assert locate_bccd(root) == root.resolve()
    report = inspect_dataset(config)
    assert report["source_format"] == "kaggle_json"
    assert (report["total_images"], report["total_json"], report["total_xml"]) == (15,15,0)
    assert {s: len(report["source_split_files"][s]) for s in ("train","val","test")} == {"train":6,"val":3,"test":6}
    assert report["total_wbc_boxes"] == 14
    assert report["records"][0]["width"] == 100  # JSON says 999: actual image wins.
    result = preprocess(config)
    assert result["source_format"] == "kaggle_json"
    assert result["splits"] == {"train":6,"val":3,"test":6}
    assert "Preserved Kaggle" in result["split_reason"]
    processed = config["processed_data_path"]
    assert (processed / "labels/train/BloodImage_00003.txt").read_text() == ""
    assert (processed / "labels/test/BloodImage_00009.txt").read_text().startswith("0 0.4000000000")
    assert validate_processed(processed)["ready"]
    assert preprocess(config)["reused"]


def test_missing_json_and_invalid_wbc_stop(synthetic_kaggle):
    root, _, config = synthetic_kaggle
    missing = root / "train/ann/BloodImage_00000.jpeg.json"
    missing.unlink()
    report = inspect_dataset(config, strict=False)
    assert "BloodImage_00000.jpeg" in report["missing_annotations"]
    assert not report["ready"]
    with pytest.raises(ValueError, match="integrity"):
        preprocess(config)


def test_zero_area_rbc_logged_but_wbc_is_critical(synthetic_kaggle):
    root, _, config = synthetic_kaggle
    path = root / "train/ann/BloodImage_00000.jpeg.json"
    data = json.loads(path.read_text())
    data["objects"][0]["points"]["exterior"] = [[3,3],[3,3]]
    path.write_text(json.dumps(data),encoding="utf-8")
    report = inspect_dataset(config)
    assert len(report["rejected_non_wbc_boxes"]) == 1
    assert report["raw_class_distribution"]["RBC"] == report["class_distribution"]["RBC"] + 1
    data["objects"][-1]["points"]["exterior"] = [[20,10],[20,10]]
    path.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError,match="integrity"):
        inspect_dataset(config)


def test_json_class_mismatch_and_split_leakage(synthetic_kaggle):
    root, _, config = synthetic_kaggle
    image = root / "train/img/BloodImage_00000.jpeg"
    path = root / "train/ann/BloodImage_00000.jpeg.json"
    data = json.loads(path.read_text())
    data["objects"][0]["classId"] = 13303
    path.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError,match="mismatched"):
        parse_kaggle_json(path,image,CLASS_IDS)
    data["objects"][0]["classId"] = 13302
    path.write_text(json.dumps(data),encoding="utf-8")
    # An image duplicated in the held-out split must fail, never be reshuffled.
    (root / "test/img/BloodImage_00009.jpeg").write_bytes(image.read_bytes())
    with pytest.raises(ValueError,match="pre-defined splits are invalid"):
        preprocess(config)
