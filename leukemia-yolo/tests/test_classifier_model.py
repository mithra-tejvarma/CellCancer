import pytest
import torch
from src.classifier_model import build_model, set_trainable
from src.classifier_training import require_classifier_cuda


def test_b3_output_and_trainable_phases():
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 2)
    set_trainable(model, "head")
    assert all(not p.requires_grad for p in model.features.parameters())
    set_trainable(model, "fine")
    assert any(p.requires_grad for p in model.features[-1].parameters())


def test_cuda_gate(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="will not switch to CPU"):
        require_classifier_cuda()
