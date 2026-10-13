import pytest
from src.classifier_evaluation import calculate_metrics


def test_known_confusion_and_metrics():
    m = calculate_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.2])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["blast_sensitivity"] == m["normal_specificity"] == m["blast_precision"] == 0.5
    assert m["macro_f1"] == 0.5


def test_no_zero_division():
    m = calculate_metrics([0, 0], [0.1, 0.2])
    assert m["blast_precision"] is None and m["roc_auc"] is None
    with pytest.raises(ValueError):
        calculate_metrics([], [])
