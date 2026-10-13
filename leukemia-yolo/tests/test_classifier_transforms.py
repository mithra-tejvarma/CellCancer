import torch
from src.classifier_transforms import CNMCDataset
from src.cnmc_data import inspect_cnmc


def test_rgb_tensor_and_deterministic_evaluation(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    frame, _ = inspect_cnmc(cfg)
    dataset = CNMCDataset(frame.head(1), 224)
    first, label = dataset[0]
    second, _ = dataset[0]
    assert first.shape == (3, 224, 224) and torch.equal(first, second)
    assert label in (0, 1)
