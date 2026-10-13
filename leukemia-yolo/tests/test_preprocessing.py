from pathlib import Path
import yaml

from src.bccd import file_hash
from src.preprocessing import preprocess
from src.validation import read_labels, validate_processed


def test_preprocessing_idempotent_and_source_unchanged(synthetic_bccd):
    root, _, config = synthetic_bccd
    source = root / "Annotations/BloodImage_00000.xml"
    before = file_hash(source)
    first = preprocess(config)
    second = preprocess(config)
    assert not first["reused"] and second["reused"]
    assert file_hash(source) == before
    processed = config["processed_data_path"]
    data = yaml.safe_load((processed / "data.yaml").read_text())
    assert data == {"path":str(processed.resolve()),"train":"images/train",
                    "val":"images/val","test":"images/test","names":{0:"WBC"}}
    assert (processed / "labels/train/BloodImage_00003.txt").read_text() == ""
    assert read_labels(processed / "labels/train/BloodImage_00000.txt") == [[0,.4,.375,.4,.5]]
    assert validate_processed(processed)["ready"]


def test_changed_source_never_replaces_existing_processed(synthetic_bccd):
    root, _, config = synthetic_bccd
    preprocess(config)
    xml = root / "Annotations/BloodImage_00000.xml"
    xml.write_text(xml.read_text().replace("<xmin>20</xmin>","<xmin>21</xmin>"),encoding="utf-8")
    import pytest
    with pytest.raises(ValueError,match="Existing dataset differs"):
        preprocess(config)
    assert (config["processed_data_path"] / "manifest.json").is_file()
