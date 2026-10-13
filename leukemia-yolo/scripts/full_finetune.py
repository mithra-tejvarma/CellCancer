"""Separate, resumable EfficientNet-B3 backbone experiment on saved C-NMC splits.

Run this script with the leukemia-yolo Conda interpreter, never inside a notebook
cell with DataLoader workers. Existing baseline artifacts are read only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.classifier_config import classifier_config
from src.classifier_evaluation import calculate_metrics, predict_manifest, _plot_evaluation
from src.classifier_model import load_checkpoint, metadata, set_trainable
from src.classifier_training import require_classifier_cuda, seed_all, class_weights
from src.classifier_transforms import CNMCDataset, MEAN, STD
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits
from src.config import run_id, write_json


def select_threshold(truth, scores):
    """Maximize validation macro F1; ties favor balanced accuracy then 0.5."""
    candidates = np.unique(np.r_[0.05, 0.5, 0.95, np.linspace(.1, .9, 81)])
    candidates = candidates[(candidates >= .05) & (candidates <= .95)]
    return max((calculate_metrics(truth, scores, float(t)) for t in candidates),
               key=lambda m: (m["macro_f1"], m["balanced_accuracy"], -abs(m["threshold"] - .5)))


def save_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    truth, scores, loss_sum = [], [], 0.
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * len(labels)
            truth.extend(labels.cpu().tolist())
            scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
    return truth, scores, loss_sum / len(truth)


def train_epoch(model, loader, optimizer, scaler, criterion, device, accumulation, smoke=False, progress=None):
    model.train()
    # Small physical batches produce unstable running statistics; keep pretrained BN stats.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = samples = steps = 0
    total = min(len(loader), 2) if smoke else len(loader)
    started = time.monotonic()
    for step, (images, labels) in enumerate(loader):
        if step >= total:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        group_size = min(accumulation, total - (step // accumulation) * accumulation)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(images), labels)
        scaler.scale(loss / group_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == total:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.item()) * len(labels)
        samples += len(labels)
        steps += 1
        if progress and (steps % 50 == 0 or steps == total):
            elapsed = time.monotonic() - started
            progress({"completed_batches": steps, "total_batches": total,
                      "seconds_per_batch_observed": elapsed / steps,
                      "estimated_remaining_epoch_seconds": (total - steps) * elapsed / steps,
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0)})
    return {"train_loss": loss_sum / samples, "batches": steps,
            "seconds_per_batch": (time.monotonic() - started) / steps,
            "samples_per_second": samples / max(time.monotonic() - started, 1e-6)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("full", "partial"), default="full")
    parser.add_argument("--init", type=Path, help="Verified baseline best_model.pth; default saved baseline")
    parser.add_argument("--resume", type=Path, help="Exact last_checkpoint.pth from this experiment")
    parser.add_argument("--smoke", action="store_true", help="Two training batches, no test exposure")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-minutes", type=float)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.patience) < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("Invalid training settings")
    cfg = classifier_config(args.config)
    device = require_classifier_cuda()
    seed_all(cfg["seed"])
    source, _ = inspect_cnmc(cfg, save=False)
    splits, split_info = create_or_load_splits(cfg, source)
    for name, frame in splits.items():
        if not all(Path(p).is_file() for p in frame.absolute_path):
            raise FileNotFoundError(f"{name} manifest contains missing images")
    baseline = args.init or cfg["output_path"] / "classifier/runs/efficientnet_b3_20261013T110511Z_67988092/best_model.pth"
    resume = args.resume.resolve() if args.resume else None
    if resume and resume.name != "last_checkpoint.pth":
        raise ValueError("Resume requires last_checkpoint.pth")
    run = resume.parent if resume else cfg["output_path"] / "classifier/full_finetune" / run_id("smoke" if args.smoke else args.mode)
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
        model, info, baseline_payload = load_checkpoint(baseline, device)
        if baseline_payload.get("split_fingerprint") != split_info["split_fingerprint"]:
            raise ValueError("Baseline split fingerprint differs from saved manifests")
        if info["image_size"] != cfg["classifier_image_size"]:
            raise ValueError("Baseline image size differs from configuration")
        config = {"fine_tune_mode": args.mode, "init_mode": "baseline_best_weights", "init_checkpoint": str(baseline.resolve()),
                  "init_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(), "epochs": args.epochs,
                  "batch_size": args.batch_size, "accumulation": args.accumulation, "workers": args.workers,
                  "lr": args.lr, "weight_decay": 1e-4, "patience": args.patience, "image_size": info["image_size"],
                  "seed": cfg["seed"], "smoke": args.smoke, "batchnorm_running_stats": "frozen",
                  "selection_objective": "validation_macro_f1", "test_exposure": "fold_2 previously evaluated in baseline"}
        write_json(run / "training_config.json", config)
        write_json(run / "class_mapping.json", info["class_mapping"])
        write_json(run / "preprocessing_config.json", {"version": info["preprocessing_version"], "image_size": info["image_size"], "mean": MEAN, "std": STD})
        write_json(run / "split_fingerprint.json", {"split_fingerprint": split_info["split_fingerprint"], "counts": split_info["counts"]})
        write_json(run / "environment_report.json", {"python": sys.executable, "torch": torch.__version__,
                   "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "vram_bytes": torch.cuda.get_device_properties(0).total_memory})
        start_epoch, best_f1, stalled = 0, -1., 0
    else:
        config = json.loads((run / "training_config.json").read_text(encoding="utf-8"))
        for key, requested in (("fine_tune_mode", args.mode), ("batch_size", args.batch_size),
                               ("accumulation", args.accumulation), ("lr", args.lr), ("epochs", args.epochs)):
            if config[key] != requested:
                raise ValueError(f"Resume setting changed: {key}")
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["split_fingerprint"] != split_info["split_fingerprint"]:
            raise ValueError("Resume split fingerprint differs")
        model, _, _ = load_checkpoint(resume, device)
        start_epoch, best_f1, stalled = saved["epoch"], saved["best_f1"], saved["stalled"]
        torch.set_rng_state(saved["torch_rng"])
        if saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    set_trainable(model, config["fine_tune_mode"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    if resume:
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"], device))
    train_loader = DataLoader(CNMCDataset(splits["train"], config["image_size"], train=True),
                              batch_size=config["batch_size"], shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(CNMCDataset(splits["val"], config["image_size"]),
                            batch_size=config["batch_size"], num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    history = run / "training_history.csv"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with (run / "training_log.txt").open("a", encoding="utf-8") as log:
            for epoch in range(start_epoch, config["epochs"]):
                epoch_start = time.monotonic()
                def progress(payload):
                    message = json.dumps({"epoch": epoch + 1, "progress": payload})
                    print(message, flush=True)
                    log.write(message + "\n"); log.flush()
                train_stats = train_epoch(model, train_loader, optimizer, scaler, criterion, device,
                                          config["accumulation"], config["smoke"], progress)
                if config["smoke"]:
                    row = {"epoch": epoch + 1, **train_stats, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "smoke": True}
                    write_json(run / "smoke_result.json", row)
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + "\n")
                    return
                truth, scores, val_loss = evaluate_loader(model, val_loader, criterion, device)
                selected = select_threshold(truth, scores)
                epoch_seconds = time.monotonic() - epoch_start
                row = {"epoch": epoch + 1, **train_stats, "val_loss": val_loss,
                       "val_macro_f1": selected["macro_f1"], "val_balanced_accuracy": selected["balanced_accuracy"],
                       "val_blast_sensitivity": selected["blast_sensitivity"], "val_normal_specificity": selected["normal_specificity"],
                       "val_roc_auc": selected["roc_auc"], "threshold": selected["threshold"],
                       "epoch_seconds": epoch_seconds, "estimated_remaining_seconds": epoch_seconds * (config["epochs"] - epoch - 1),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                with history.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                improved = selected["macro_f1"] > best_f1
                if improved:
                    best_f1, stalled = selected["macro_f1"], 0
                    best_info = metadata(config["image_size"], selected["threshold"])
                    save_atomic(run / "best_model.pth", {"model_state": model.state_dict(), "metadata": best_info,
                                "split_fingerprint": split_info["split_fingerprint"], "epoch": epoch + 1,
                                "validation_metrics": selected, "fine_tune_mode": config["fine_tune_mode"]})
                    write_json(run / "threshold_config.json", {"threshold": selected["threshold"],
                               "source": "validation", "objective": config["selection_objective"], "epoch": epoch + 1})
                else:
                    stalled += 1
                save_atomic(run / "last_checkpoint.pth", {"model_state": model.state_dict(),
                            "metadata": metadata(config["image_size"], selected["threshold"]),
                            "split_fingerprint": split_info["split_fingerprint"], "epoch": epoch + 1,
                            "best_f1": best_f1, "stalled": stalled, "optimizer_state": optimizer.state_dict(),
                            "scaler_state": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate()})
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n"); log.flush()
                if args.max_minutes and (time.monotonic() - started) / 60 >= args.max_minutes:
                    print(f"Time budget reached. Resume from {run / 'last_checkpoint.pth'}", flush=True)
                    return
                if stalled >= config["patience"]:
                    break
        # Evaluate the selected model once, with the validation-selected threshold.
        best, best_info, _ = load_checkpoint(run / "best_model.pth", device)
        for split in ("val", "test"):
            frame = splits[split]
            scores = predict_manifest(best, frame, best_info["image_size"], device, config["batch_size"], args.workers)
            metrics = calculate_metrics(frame.class_id, scores, best_info["decision_threshold"])
            metrics.update({"split": split, "subjects": int(frame.subject_id.nunique()),
                            "split_fingerprint": split_info["split_fingerprint"],
                            "checkpoint_sha256": hashlib.sha256((run / "best_model.pth").read_bytes()).hexdigest(),
                            "fold_2_prior_exposure": split == "test"})
            write_json(run / f"{split}_metrics.json", metrics)
            pd.DataFrame([metrics]).to_csv(run / f"{split}_metrics.csv", index=False)
            figures = run / split
            figures.mkdir(exist_ok=True)
            _plot_evaluation(frame.class_id.to_numpy(), scores, metrics, figures,
                             "Validation" if split == "val" else "Held-out test")
        write_json(run / "completed.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                                              "fine_tune_mode": config["fine_tune_mode"], "best_validation_macro_f1": best_f1})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(f"CUDA OOM. Resume {run / 'last_checkpoint.pth'} with the original batch settings, "
                           "or start a separate run with --batch-size 2 --accumulation 4")


if __name__ == "__main__":
    main()
