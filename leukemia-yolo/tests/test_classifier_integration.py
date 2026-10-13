"""CPU synthetic integration verifies the held-out evaluation contract, not model quality."""
import torch
import pytest

from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits
from src.classifier_model import build_model, metadata
from src.classifier_evaluation import evaluate
from src.classifier_inference import predict_paths


@pytest.mark.parametrize("test_duplicate", [False, True])
def test_synthetic_checkpoint_evaluation_and_crop_prediction(synthetic_cnmc, tmp_path, test_duplicate):
    _, training, _, cfg = synthetic_cnmc
    cfg["classifier_workers"] = 0
    cfg["classifier_batch_size"] = 2
    if test_duplicate:
        candidate = training / "fold_1" / "all" / "UID_1_1_1_all.bmp"
        candidate.write_bytes((training / "fold_2" / "all" / "UID_6_6_1_all.bmp").read_bytes())
    source, _ = inspect_cnmc(cfg)
    splits, summary = create_or_load_splits(cfg, source)
    assert summary["excluded_images"] == (2 if test_duplicate else 0)
    checkpoint = tmp_path / "best_model.pth"
    model = build_model(pretrained=False)
    torch.save({"model_state": model.state_dict(), "metadata": metadata(224),
                "split_fingerprint": summary["split_fingerprint"]}, checkpoint)
    metric = evaluate(cfg, checkpoint)
    assert metric["split"] == "fold_2" and metric["n"] == len(splits["test"]) == 4
    assert metric["tp"] + metric["tn"] + metric["fp"] + metric["fn"] == 4
    results = cfg["output_path"] / "classifier" / "paper_results"
    for name in ("classification_metrics.json", "classification_metrics.csv", "confusion_matrix.png",
                 "roc_curve.png", "pr_curve.png", "evaluation_examples.png", "test_predictions.csv"):
        assert (results / name).is_file()
    image = training / "fold_2" / "all" / "UID_6_6_1_all.bmp"
    predicted = predict_paths(checkpoint, image, tmp_path / "predictions", device="cpu")
    assert len(predicted) == 1 and predicted.predicted_class_id.iloc[0] in (0, 1)
    assert (tmp_path / "predictions" / "classifier_predictions.json").is_file()
