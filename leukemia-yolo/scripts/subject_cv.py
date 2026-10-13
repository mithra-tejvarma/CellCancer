"""Optional subject-grouped C-NMC cross-validation; fold_2 never participates.

Each --run fold starts from fresh ImageNet EfficientNet-B3 weights. The previous
full checkpoint cannot initialize CV because it trained on future fold subjects.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from src.classifier_config import classifier_config
from src.classifier_evaluation import calculate_metrics
from src.classifier_model import build_model, metadata, set_trainable
from src.classifier_training import class_weights, require_classifier_cuda, seed_all
from src.classifier_transforms import CNMCDataset
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import _subject_groups, create_or_load_splits
from src.config import run_id, write_json
from scripts.full_finetune import train_epoch, evaluate_loader, save_atomic
from scripts.run_improvement_trial import operating_point


def folds_for(splits, n_splits=5, seed=42):
    frame = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    groups = _subject_groups(frame)
    group_ids = frame.subject_id.map(groups.set_index("subject_id").group_id)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for number, (train_index, val_index) in enumerate(splitter.split(frame, frame.class_id, group_ids)):
        train, val = frame.iloc[train_index].copy(), frame.iloc[val_index].copy()
        test = splits["test"]
        for column in ("subject_id", "absolute_path", "sha256"):
            if set(train[column]) & set(val[column]) or set(val[column]) & set(test[column]):
                raise ValueError(f"Fold {number} leakage in {column}")
        if set(val.class_id) != {0, 1}:
            raise ValueError(f"Fold {number} lacks a class")
        folds.append((train, val))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Train every fold; default only writes a verified plan")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.folds < 2 or min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0:
        parser.error("Invalid cross-validation settings")
    cfg = classifier_config()
    source, _ = inspect_cnmc(cfg, save=False)
    splits, info = create_or_load_splits(cfg, source)
    folds = folds_for(splits, args.folds, cfg["seed"])
    output = cfg["output_path"] / "classifier/improvements/cross_validation" / run_id("cv")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"protocol": "StratifiedGroupKFold of development subjects; fold_2 excluded",
            "n_splits": args.folds, "seed": cfg["seed"], "split_fingerprint": info["split_fingerprint"],
            "initialization": "fresh ImageNet EfficientNet-B3 for each fold; no previously C-NMC-trained checkpoint",
            "folds": [{"fold": index, "train_subjects": sorted(train.subject_id.unique()),
                       "val_subjects": sorted(val.subject_id.unique()),
                       "train_cells": len(train), "val_cells": len(val),
                       "val_normal_cells": int((val.class_id == 0).sum()),
                       "val_blast_cells": int((val.class_id == 1).sum())}
                      for index, (train, val) in enumerate(folds)]}
    write_json(output / "cv_plan.json", plan)
    if not args.run:
        print(output / "cv_plan.json")
        return
    device = require_classifier_cuda()
    summary = []
    for index, (train, val) in enumerate(folds):
        seed_all(cfg["seed"] + index)
        model = build_model(pretrained=True).to(device)
        set_trainable(model, "full")
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-5, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.CrossEntropyLoss(weight=class_weights(train, device))
        train_loader = DataLoader(CNMCDataset(train, 224, train=True), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        val_loader = DataLoader(CNMCDataset(val, 224), batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        folder = output / f"fold_{index}"
        folder.mkdir()
        best, stalled = None, 0
        for epoch in range(args.epochs):
            started = time.monotonic()
            stats = train_epoch(model, train_loader, optimizer, scaler, criterion, device, 2)
            truth, scores, val_loss = evaluate_loader(model, val_loader, criterion, device)
            selected = operating_point(truth, scores)
            current = selected or calculate_metrics(truth, scores, .5)
            rank = (selected is not None, current["normal_specificity"], current["macro_f1"])
            row = {"fold": index, "epoch": epoch + 1, "train_loss": stats["train_loss"],
                   "val_loss": val_loss, "eligible": selected is not None,
                   "threshold": current["threshold"], "val_specificity": current["normal_specificity"],
                   "val_sensitivity": current["blast_sensitivity"], "val_macro_f1": current["macro_f1"],
                   "val_roc_auc": current["roc_auc"], "epoch_seconds": time.monotonic() - started}
            with (folder / "history.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                if handle.tell() == 0: writer.writeheader()
                writer.writerow(row)
            print(json.dumps(row), flush=True)
            if best is None or rank > best:
                best, stalled = rank, 0
                save_atomic(folder / "best_model.pth", {"model_state": model.state_dict(),
                    "metadata": metadata(224, current["threshold"]), "fold": index,
                    "epoch": epoch + 1, "validation_metrics": current,
                    "subject_cv": True, "split_fingerprint": info["split_fingerprint"]})
                write_json(folder / "best_metrics.json", {"fold": index, "epoch": epoch + 1,
                    "train_subjects": int(train.subject_id.nunique()),
                    "val_subjects": int(val.subject_id.nunique()), **current})
            else:
                stalled += 1
            if stalled >= args.patience: break
        summary.append(json.loads((folder / "best_metrics.json").read_text()))
        write_json(output / "partial_summary.json", summary)
    fields = ("normal_specificity", "blast_sensitivity", "macro_f1", "roc_auc")
    report = {"folds": summary, "mean": {name: float(np.mean([row[name] for row in summary])) for name in fields},
              "std": {name: float(np.std([row[name] for row in summary], ddof=1)) for name in fields}}
    write_json(output / "cv_summary.json", report)
    print(output / "cv_summary.json")


if __name__ == "__main__":
    main()
