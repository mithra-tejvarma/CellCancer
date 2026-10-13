"""Two-phase EfficientNet-B3 training; call from a guarded script on Windows."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .classifier_config import classifier_output
from .classifier_evaluation import calculate_metrics
from .cnmc_data import inspect_cnmc
from .classifier_model import build_model, metadata, set_trainable
from .classifier_transforms import CNMCDataset
from .cnmc_splits import create_or_load_splits, validate_splits
from .config import run_id, write_json


def require_classifier_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(f"NVIDIA CUDA unavailable (torch={torch.__version__}, build={torch.version.cuda}). "
                           "Install a compatible CUDA PyTorch wheel; classifier training will not switch to CPU.")
    return torch.device("cuda:0")


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_weights(frame: pd.DataFrame, device) -> torch.Tensor:
    counts = np.bincount(frame.class_id.astype(int), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training split requires both classes")
    return torch.tensor(len(frame) / (2.0 * counts), dtype=torch.float32, device=device)


def _save(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _plots(folder: Path):
    import matplotlib.pyplot as plt
    path = folder / "training_history.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    for column, filename, label in (("train_loss", "loss_curve.png", "Cross-entropy loss"),
                                    ("val_accuracy", "accuracy_curve.png", "Validation accuracy"),
                                    ("val_macro_f1", "f1_curve.png", "Validation macro F1")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(frame) + 1), frame[column], marker="o", label=column)
        if column == "train_loss":
            ax.plot(range(1, len(frame) + 1), frame.val_loss, marker="s", label="val_loss")
        ax.set(xlabel="Completed epoch", ylabel=label, title=label); ax.legend(); fig.tight_layout()
        fig.savefig(folder / filename, dpi=220); plt.close(fig)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    if not state:
        return
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(cfg: dict, mode: str = "full", resume: Path | None = None, *, batch_size: int | None = None,
          max_minutes: float | None = None, run_dir: Path | None = None) -> Path:
    """Train on CUDA. Smoke uses one head epoch and at most two batches."""
    if mode not in ("smoke", "head", "fine", "full"):
        raise ValueError("mode must be smoke, head, fine or full")
    device = require_classifier_cuda()
    seed_all(cfg["seed"])
    source, _ = inspect_cnmc(cfg, save=False)
    splits, split_info = create_or_load_splits(cfg, source)
    validate_splits(splits, source)
    for name, frame in splits.items():
        missing = [p for p in frame.absolute_path if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{name} references missing source image: {missing[0]}")
    base = classifier_output(cfg) / "runs"
    if resume:
        resume = Path(resume).resolve()
        if resume.name != "last_checkpoint.pth":
            raise ValueError("Resume from last_checkpoint.pth, not the selected best_model.pth")
        run_dir = resume.parent
    else:
        run_dir = Path(run_dir) if run_dir else base / run_id("smoke" if mode == "smoke" else "efficientnet_b3")
        if run_dir.exists() and any(p.name != "training_log.txt" for p in run_dir.iterdir()):
            raise FileExistsError(f"Run already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or cfg["classifier_batch_size"]
    if batch_size not in (2, 4, 8) and batch_size <= 0:
        raise ValueError("Invalid batch size")
    configuration = {"mode": mode, "batch_size": batch_size, "image_size": cfg["classifier_image_size"],
                     "head_epochs": 1 if mode == "smoke" else cfg["classifier_head_epochs"],
                     "fine_epochs": cfg["classifier_fine_epochs"], "head_lr": cfg["classifier_head_lr"],
                     "fine_lr": cfg["classifier_fine_lr"], "patience": cfg["classifier_patience"],
                     "workers": cfg["classifier_workers"], "seed": cfg["seed"],
                     "max_minutes": max_minutes, "class_weights_train_only": True,
                     "augmentation": "resize; horizontal flip; rotation ±10°; brightness/contrast ±8%; ImageNet normalization",
                     "determinism": "fixed seeds; cudnn deterministic; some GPU kernels may still vary"}
    if not resume:
        write_json(run_dir / "training_config.json", configuration)
        write_json(run_dir / "dataset_split_metadata.json", split_info)
        write_json(run_dir / "environment_metadata.json", {"torch": torch.__version__,
                   "torchvision": __import__("torchvision").__version__, "cuda_build": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory})
        model = build_model(pretrained=True).to(device)
        checkpoint = None
    else:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint["split_fingerprint"] != split_info["split_fingerprint"]:
            raise ValueError("Resume checkpoint belongs to a different split")
        old = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        if old["image_size"] != cfg["classifier_image_size"]:
            raise ValueError("Cannot change image size during resume")
        if mode != old["mode"] and not (old["mode"] == "head" and mode == "fine"):
            raise ValueError("Resume mode does not match existing run")
        old["batch_size"] = batch_size
        old["workers"] = cfg["classifier_workers"]
        write_json(run_dir / "training_config.json", old)
        configuration = old
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state"])
        _restore_rng(checkpoint.get("rng_state"))
    weights = class_weights(splits["train"], device)
    print("TRAIN-only class weights [Normal, Blast]:", weights.tolist(), flush=True)
    criterion = nn.CrossEntropyLoss(weight=weights)
    plan = (["head"] if mode in ("head", "smoke") else ["fine"] if mode == "fine" else ["head", "fine"])
    best_score = float(checkpoint["best_score"]) if checkpoint else -1.0
    prior_history = run_dir / "training_history.csv"
    elapsed_offset = (float(pd.read_csv(prior_history).elapsed_seconds.iloc[-1])
                      if resume and prior_history.is_file() else 0.0)
    start = time.monotonic() - elapsed_offset
    torch.cuda.reset_peak_memory_stats()
    for phase in plan:
        if checkpoint and checkpoint["phase"] == "fine" and phase == "head":
            continue
        if phase == "fine" and (checkpoint is None or checkpoint["phase"] == "head") and (run_dir / "best_model.pth").is_file():
            best_head = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=False)
            if best_head.get("phase") == "head":
                model.load_state_dict(best_head["model_state"])
        set_trainable(model, phase)
        model.features.eval()  # frozen BatchNorm statistics remain fixed
        if phase == "fine":
            for block in model.features[-2:]:
                block.train()
        lr = cfg["classifier_head_lr"] if phase == "head" else cfg["classifier_fine_lr"]
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        phase_start = checkpoint["next_epoch"] if checkpoint and checkpoint["phase"] == phase else 0
        if checkpoint and checkpoint["phase"] == phase:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
        max_epochs = configuration["head_epochs"] if phase == "head" else configuration["fine_epochs"]
        stalled = 0
        loader = DataLoader(CNMCDataset(splits["train"], configuration["image_size"], train=True),
                            batch_size=batch_size, shuffle=True, num_workers=configuration["workers"],
                            pin_memory=True, persistent_workers=configuration["workers"] > 0)
        val_loader = DataLoader(CNMCDataset(splits["val"], configuration["image_size"]),
                                batch_size=batch_size, num_workers=configuration["workers"], pin_memory=True,
                                persistent_workers=configuration["workers"] > 0)
        for epoch in range(phase_start, max_epochs):
            model.train()
            model.features.eval()
            if phase == "fine":
                for block in model.features[-2:]:
                    block.train()
            loss_total = examples = 0
            for step, (images, labels) in enumerate(loader):
                images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images); loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                loss_total += loss.item() * len(labels); examples += len(labels)
                if mode == "smoke" and step >= 1:
                    break
            model.eval()
            val_loss = val_total = 0
            scores, actual = [], []
            with torch.no_grad():
                for step, (images, labels) in enumerate(val_loader):
                    images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images); loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels); val_total += len(labels)
                    scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
                    actual.extend(labels.cpu().tolist())
                    if mode == "smoke" and step >= 1:
                        break
            metrics = calculate_metrics(actual, scores)
            row = {"phase": phase, "epoch": epoch + 1, "train_loss": loss_total / examples,
                   "val_loss": val_loss / val_total, "val_accuracy": metrics["accuracy"],
                   "val_macro_f1": metrics["macro_f1"], "val_blast_sensitivity": metrics["blast_sensitivity"],
                   "elapsed_seconds": round(time.monotonic() - start, 2), "batch_size": batch_size,
                   "peak_vram_bytes": torch.cuda.max_memory_allocated(0)}
            history = run_dir / "training_history.csv"
            with history.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            print(json.dumps(row), flush=True)
            payload = {"model_state": model.state_dict(), "metadata": metadata(configuration["image_size"]),
                       "phase": phase, "epoch": epoch + 1, "next_epoch": epoch + 1,
                       "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                       "scheduler_state": None, "best_score": max(best_score, metrics["macro_f1"]),
                       "split_fingerprint": split_info["split_fingerprint"], "rng_state": _rng_state()}
            _save(run_dir / "last_checkpoint.pth", payload)
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]; stalled = 0
                _save(run_dir / "best_model.pth", {"model_state": model.state_dict(),
                      "metadata": payload["metadata"], "validation_macro_f1": best_score,
                      "validation_blast_sensitivity": metrics["blast_sensitivity"],
                      "split_fingerprint": split_info["split_fingerprint"], "phase": phase, "epoch": epoch + 1})
            else:
                stalled += 1
            _plots(run_dir)
            if max_minutes and (time.monotonic() - start) / 60 >= max_minutes:
                print("Time budget reached; resume from last_checkpoint.pth", flush=True)
                return run_dir
            if stalled >= configuration["patience"]:
                print(f"Early stopping {phase} after {epoch + 1} epochs", flush=True)
                break
        checkpoint = None
    write_json(run_dir / "completed.json", {"mode": mode, "completed": True, "best_validation_macro_f1": best_score})
    return run_dir
