import pytest

from src.preprocessing import preprocess
from src.validation import read_labels, validate_processed
from src.environment import require_cuda


def test_reject_foreign_class_and_missing_label(synthetic_bccd):
    _, _, config = synthetic_bccd
    preprocess(config)
    label = config["processed_data_path"] / "labels/train/BloodImage_00000.txt"
    original = label.read_text()
    label.write_text(original.replace("0 ","1 ",1))
    with pytest.raises(ValueError,match="invalid WBC class"):
        validate_processed(config["processed_data_path"])
    label.write_text(original)
    label.unlink()
    with pytest.raises(ValueError,match="labels and images"):
        validate_processed(config["processed_data_path"])


def test_out_of_range_label(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("0 0.9 0.5 0.4 0.1\n")
    with pytest.raises(ValueError,match="outside image"):
        read_labels(path)


def test_cuda_error_is_explicit(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version,"cuda",None)
    with pytest.raises(RuntimeError,match="CPU-only"):
        require_cuda(0)
