"""Mock Ultralytics only to test orchestration; these are NOT model results."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.config import read_json
from src.evaluation import evaluate
from src.inference import predict
from src.preprocessing import preprocess
from src.reporting import build_summary
from src.training import run_training, resolve_checkpoint


class FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
        self.task = "detect"
        self.trainer = SimpleNamespace(batch_size=4)

    def train(self, **args):
        run = Path(args["project"]) / args["name"]
        (run / "weights").mkdir(parents=True, exist_ok=True)
        (run / "weights" / "best.pt").write_bytes(b"synthetic checkpoint")
        (run / "weights" / "last.pt").write_bytes(b"synthetic checkpoint")
        (run / "results.csv").write_text(
            "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),time\n"
            "1,1.2,0.4,1.0,0.2,0.3,0.1,0.05,3.1\n", encoding="utf-8")
        (run / "results.png").write_bytes(b"synthetic figure")

    def val(self, **args):
        assert args["split"] == "test"
        figures = Path(args["project"]) / args["name"]
        figures.mkdir(parents=True, exist_ok=False)
        for name in ("confusion_matrix.png", "BoxPR_curve.png", "BoxF1_curve.png"):
            (figures / name).write_bytes(b"synthetic plot")
        return SimpleNamespace(box=SimpleNamespace(mp=.2, mr=.3, map50=.1, map=.05))

    def predict(self, **args):
        assert args["conf"] == .25 and args["iou"] == .7
        box = SimpleNamespace(cls=torch.tensor(0), conf=torch.tensor(.8),
                              xyxy=torch.tensor([[20,10,60,50]]))
        return [SimpleNamespace(boxes=[box], plot=lambda: np.zeros((80,100,3),dtype=np.uint8))]


def test_mocked_train_test_predict_provenance(synthetic_kaggle,monkeypatch):
    _, _, config = synthetic_kaggle
    preprocess(config)
    monkeypatch.setitem(sys.modules,"ultralytics",SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr("src.training.require_cuda",lambda device: {"gpu_name":"synthetic GPU", "device":device})
    monkeypatch.setattr("src.evaluation.require_cuda",lambda device: {"device":device})
    monkeypatch.setattr("src.inference.require_cuda",lambda device: {"device":device})
    monkeypatch.setattr("src.training.nvidia_smi",lambda: {"available":False})
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda device: 0)
    smoke = run_training(config,kind="smoke")
    assert read_json(smoke / "run_metadata.json")["completed_epochs"] == 1
    with pytest.raises(ValueError,match="smoke"):
        run_training(config,kind="main")
    main = run_training(config,kind="main",smoke_run=smoke)
    assert resolve_checkpoint(config) == main / "weights" / "best.pt"
    assert read_json(main / "effective_training_config.json")["batch"] == .75
    result = evaluate(config)
    assert result["split"] == "test" and result["test_images"] == 6
    predictions = predict(config,limit=6)
    assert len(predictions["examples"]) == 6 and predictions["detections"] == 6
    assert len(list(Path(predictions["crop_dir"]).glob("*.png"))) == 6
    summary = build_summary(config)
    assert summary["test_mAP50"] == .1
    assert summary["checkpoint_sha256"] == result["checkpoint_sha256"]
    # Everything above is synthetic and lives only in pytest's temporary tree.
