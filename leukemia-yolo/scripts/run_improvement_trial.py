"""Controlled C-NMC continuation trials; validation-only model selection.

All arms start from the same verified full checkpoint and keep fold_2 unused.
Run in a guarded Windows process, one trial at a time on the 4 GB GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.classifier_config import classifier_config
from src.classifier_evaluation import calculate_metrics
from src.classifier_model import load_checkpoint, metadata, set_trainable
from src.classifier_training import class_weights, require_classifier_cuda, seed_all
from src.classifier_transforms import CNMCDataset, MEAN, STD
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits
from src.config import run_id, write_json
from scripts.full_finetune import train_epoch, evaluate_loader, save_atomic


ARMS = ("control", "augmentation_trial", "class_weight_trial")


def augmentation_transform(size):
    return T.Compose([T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(12),
                      T.ColorJitter(brightness=.10, contrast=.10, saturation=.04, hue=.01),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def operating_point(truth, scores):
    """Specificity-maximizing validation threshold with sensitivity >= 0.95."""
    truth, scores = np.asarray(truth), np.asarray(scores)
    grid = np.unique(np.r_[np.linspace(.001, .1, 100), np.linspace(.105, .995, 179), .5])
    pred = scores[None, :] >= grid[:, None]
    negatives, positives = truth == 0, truth == 1
    specificity = (~pred[:, negatives]).mean(axis=1)
    sensitivity = pred[:, positives].mean(axis=1)
    eligible = np.flatnonzero(sensitivity >= .95)
    if len(eligible) == 0:
        return None
    # Higher threshold among equal points avoids a needlessly high FP count.
    index = max(eligible, key=lambda i: (specificity[i], sensitivity[i], grid[i]))
    return calculate_metrics(truth, scores, float(grid[index]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.accumulation) < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("Invalid settings")
    cfg = classifier_config()
    device = require_classifier_cuda()
    seed_all(cfg["seed"])
    source, _ = inspect_cnmc(cfg, save=False)
    splits, split_info = create_or_load_splits(cfg, source)
    init = cfg["output_path"] / "classifier/full_finetune/full_20261013T142418Z_d2b3a8ab/best_model.pth"
    model, info, initial = load_checkpoint(init, device)
    if initial["split_fingerprint"] != split_info["split_fingerprint"] or initial.get("fine_tune_mode") != "full":
        raise ValueError("Initial checkpoint split or training mode mismatch")
    set_trainable(model, "full")
    base = cfg["output_path"] / "classifier/improvements" / args.arm
    run = base / run_id("smoke" if args.smoke else "trial")
    run.mkdir(parents=True, exist_ok=False)
    config = {"arm": args.arm, "init_checkpoint": str(init.resolve()),
              "init_sha256": hashlib.sha256(init.read_bytes()).hexdigest(),
              "split_fingerprint": split_info["split_fingerprint"], "image_size": info["image_size"],
              "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
              "accumulation": args.accumulation, "workers": args.workers, "lr": args.lr,
              "weight_decay": 1e-4, "seed": cfg["seed"], "batchnorm_running_stats": "frozen",
              "augmentation": ("vertical flip, rotation ±12°, brightness/contrast ±10%, saturation ±4%, hue ±0.01"
                               if args.arm == "augmentation_trial" else "existing baseline policy"),
              "loss_weighting": "none" if args.arm == "class_weight_trial" else "inverse frequency, train only",
              "sampling": "shuffle only", "selection_objective": "maximize validation specificity subject to blast sensitivity >= 0.95",
              "test_set_used": False, "smoke": args.smoke}
    write_json(run / "training_config.json", config)
    write_json(run / "class_mapping.json", info["class_mapping"])
    write_json(run / "preprocessing_config.json", {"image_size": info["image_size"],
               "version": info["preprocessing_version"], "mean": MEAN, "std": STD})
    write_json(run / "split_fingerprint.json", {"split_fingerprint": split_info["split_fingerprint"]})
    write_json(run / "environment_report.json", {"python": sys.executable,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0),
               "dedicated_vram_bytes": torch.cuda.get_device_properties(0).total_memory})
    train_data = CNMCDataset(splits["train"], info["image_size"], train=True)
    if args.arm == "augmentation_trial":
        train_data.transform = augmentation_transform(info["image_size"])
        # Store a small morphology check without copying the source dataset.
        preview = Image.new("RGB", (info["image_size"] * 4, info["image_size"] * 2))
        for index, path in enumerate(splits["train"].absolute_path.head(4)):
            with Image.open(path) as source_image:
                original = source_image.convert("RGB").resize((info["image_size"], info["image_size"]))
                tensor = train_data.transform(source_image.convert("RGB"))
            augmented = ((tensor.permute(1, 2, 0).numpy() * np.asarray(STD) + np.asarray(MEAN)) * 255).clip(0, 255).astype(np.uint8)
            preview.paste(original, (index * info["image_size"], 0))
            preview.paste(Image.fromarray(augmented), (index * info["image_size"], info["image_size"]))
        preview.save(run / "augmentation_preview.png")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(CNMCDataset(splits["val"], info["image_size"]),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    weights = None if args.arm == "class_weight_trial" else class_weights(splits["train"], device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    best = None
    stalled = 0
    history_path = run / "training_history.csv"
    started = time.monotonic()
    try:
        with (run / "training_log.txt").open("a", encoding="utf-8") as log:
            for epoch in range(args.epochs):
                epoch_start = time.monotonic()
                def progress(value):
                    message = json.dumps({"epoch": epoch + 1, "progress": value})
                    print(message, flush=True); log.write(message + "\n"); log.flush()
                train_stats = train_epoch(model, train_loader, optimizer, scaler, criterion, device,
                                          args.accumulation, args.smoke, progress)
                if args.smoke:
                    write_json(run / "smoke_result.json", {"arm": args.arm, **train_stats,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)})
                    return
                truth, scores, val_loss = evaluate_loader(model, val_loader, criterion, device)
                selected = operating_point(truth, scores)
                reference = calculate_metrics(truth, scores, .5)
                current = selected or reference
                row = {"epoch": epoch + 1, **train_stats, "val_loss": val_loss,
                       "eligible": selected is not None, "threshold": current["threshold"],
                       "val_specificity": current["normal_specificity"],
                       "val_sensitivity": current["blast_sensitivity"],
                       "val_macro_f1": current["macro_f1"],
                       "val_balanced_accuracy": current["balanced_accuracy"],
                       "val_roc_auc": current["roc_auc"],
                       "epoch_seconds": time.monotonic() - epoch_start,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                with history_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=row.keys())
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                predictions = splits["val"][["relative_path", "subject_id", "class_id"]].copy()
                predictions["blast_score"] = scores
                predictions.to_csv(run / f"validation_epoch_{epoch+1:02d}.csv", index=False)
                # Eligible runs outrank ineligible; then specificity and macro F1.
                rank = (selected is not None, current["normal_specificity"], current["macro_f1"])
                if best is None or rank > best:
                    best, stalled = rank, 0
                    save_atomic(run / "best_model.pth", {"model_state": model.state_dict(),
                        "metadata": metadata(info["image_size"], current["threshold"]),
                        "split_fingerprint": split_info["split_fingerprint"],
                        "fine_tune_mode": "full", "trial_arm": args.arm,
                        "epoch": epoch + 1, "validation_metrics": current,
                        "objective_eligible": selected is not None})
                    write_json(run / "threshold_config.json", {"threshold": current["threshold"],
                        "source": "validation", "objective": config["selection_objective"],
                        "eligible": selected is not None, "epoch": epoch + 1})
                else:
                    stalled += 1
                save_atomic(run / "last_checkpoint.pth", {"model_state": model.state_dict(),
                    "metadata": metadata(info["image_size"], current["threshold"]),
                    "epoch": epoch + 1, "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(), "split_fingerprint": split_info["split_fingerprint"]})
                message = json.dumps(row)
                print(message, flush=True); log.write(message + "\n"); log.flush()
                if stalled >= args.patience:
                    break
        write_json(run / "completed.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                   "arm": args.arm, "epochs_completed": epoch + 1, "best_rank": best,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)})
        print(run)
    except BaseException as exc:
        with (run / "training_log.txt").open("a", encoding="utf-8") as log:
            log.write(f"ERROR {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
