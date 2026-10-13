import torch
import pytest
from src.classifier_model import build_model, load_checkpoint, metadata


def test_checkpoint_load_and_label_contract(tmp_path):
    model = build_model(pretrained=False)
    path = tmp_path / "best_model.pth"
    torch.save({"model_state": model.state_dict(), "metadata": metadata(224)}, path)
    loaded, info, _ = load_checkpoint(path)
    assert loaded.classifier[-1].out_features == 2
    assert info["class_mapping"] == {"0": "Normal (HEM)", "1": "Leukemic Blast (ALL)"}
    torch.save({"model_state": model.state_dict(), "metadata": {**metadata(224), "class_mapping": {"0": "Blast", "1": "Normal"}}}, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path)
