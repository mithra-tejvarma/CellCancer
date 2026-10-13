from pathlib import Path
import pytest

from src.bccd import inspect_dataset, pixel_hash
from src.preprocessing import choose_splits
from src.validation import validate_split_sets


def test_official_splits_preserved_and_trainval_ignored(synthetic_bccd):
    _, _, config = synthetic_bccd
    report = inspect_dataset(config)
    records = report["records"]
    for record in records:
        record["pixel_sha256"] = pixel_hash(Path(record["image"]))
    splits,reason = choose_splits(records,report["source_split_files"],42)
    assert [len(splits[s]) for s in ("train","val","test")] == [8,2,2]
    assert "Preserved" in reason
    assert "trainval" not in splits


def test_overlap_and_incomplete_rejected():
    with pytest.raises(ValueError,match="overlap"):
        validate_split_sets({"train":["a"],"val":["a"],"test":["c"]},{"a","b","c"})
    with pytest.raises(ValueError,match="Incomplete"):
        validate_split_sets({"train":["a"],"val":["b"],"test":["c"]},{"a","b","c","d"})


def test_bad_official_splits_fallback_is_deterministic(synthetic_bccd):
    _, _, config = synthetic_bccd
    report = inspect_dataset(config)
    records = report["records"]
    for record in records:
        record["pixel_sha256"] = pixel_hash(Path(record["image"]))
    source = {"train":["BloodImage_00000"], "val":["BloodImage_00000"], "test":["BloodImage_00001"]}
    first,reason = choose_splits(records,source,42)
    second,_ = choose_splits(records,source,42)
    assert first == second and "fallback" in reason
    validate_split_sets(first,{r["id"] for r in records})


def test_identical_pixels_never_cross_splits(synthetic_bccd):
    root, _, config = synthetic_bccd
    # Different filenames, identical decoded image; official files now leak.
    from PIL import Image
    first = root / "JPEGImages/BloodImage_00000.jpg"
    second = root / "JPEGImages/BloodImage_00008.jpg"
    second.write_bytes(first.read_bytes())
    report = inspect_dataset(config)
    records = report["records"]
    for record in records:
        record["pixel_sha256"] = pixel_hash(Path(record["image"]))
    splits,reason = choose_splits(records,report["source_split_files"],42)
    assert "fallback" in reason
    members = {identifier: split for split,identifiers in splits.items() for identifier in identifiers}
    assert members["BloodImage_00000"] == members["BloodImage_00008"]
