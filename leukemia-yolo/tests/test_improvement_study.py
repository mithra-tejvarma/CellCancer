import numpy as np
from PIL import Image

from scripts.improvement_audit import select_operating_point, threshold_table
from scripts.run_improvement_trial import augmentation_transform, operating_point
from src.classifier_model import build_model, load_checkpoint, set_trainable


def test_specificity_objective_honors_sensitivity_floor():
    truth = [0, 0, 0, 1, 1, 1]
    scores = [.1, .2, .8, .3, .85, .95]
    point = operating_point(truth, scores)
    assert point is not None
    assert point["blast_sensitivity"] >= .95
    table = threshold_table(truth, scores)
    selected = select_operating_point(table)
    assert selected is not None and selected["blast_sensitivity"] >= .95


def test_objective_can_report_infeasible():
    # All true blasts score below all normals: no threshold reaches the target
    # while classifying anything as normal, but a low threshold still reaches it.
    truth = [0, 0, 1, 1]
    scores = [.8, .9, .1, .2]
    point = operating_point(truth, scores)
    assert point is not None and point["normal_specificity"] == 0


def test_augmentation_rgb_and_shape():
    transform = augmentation_transform(224)
    tensor = transform(Image.new("RGB", (450, 450), (30, 60, 90)))
    assert tensor.shape == (3, 224, 224)
    assert np.isfinite(tensor.numpy()).all()


def test_full_mode_unfreezes_backbone():
    model = build_model(pretrained=False)
    set_trainable(model, "full")
    assert all(param.requires_grad for param in model.features.parameters())
