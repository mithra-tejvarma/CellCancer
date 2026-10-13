"""Immutable subject-disjoint manifests referencing original C-NMC images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .classifier_config import classifier_output
from .config import read_json, write_json

SPLITS = ("train", "val", "test")


def source_fingerprint(df: pd.DataFrame) -> str:
    fields = df[["relative_path", "sha256", "class_id", "subject_id", "fold"]].sort_values("relative_path")
    return hashlib.sha256(fields.to_csv(index=False).encode()).hexdigest()


def validate_splits(splits: dict[str, pd.DataFrame], source: pd.DataFrame | None = None) -> dict:
    if set(splits) != set(SPLITS):
        raise ValueError("Require train, val and test manifests")
    subjects = {name: set(frame.subject_id) for name, frame in splits.items()}
    paths = {name: set(frame.absolute_path) for name, frame in splits.items()}
    hashes = {name: set(frame.sha256) for name, frame in splits.items()}
    for i, left in enumerate(SPLITS):
        frame = splits[left]
        if len(frame) == 0 or set(frame.class_id.astype(int)) != {0, 1}:
            raise ValueError(f"{left} is empty or missing one class")
        for right in SPLITS[i + 1:]:
            if subjects[left] & subjects[right] or paths[left] & paths[right] or hashes[left] & hashes[right]:
                raise ValueError(f"Leakage between {left} and {right}: subject, path, or image hash overlaps")
    if source is not None:
        all_paths = [p for frame in splits.values() for p in frame.absolute_path]
        if len(all_paths) != len(source) or set(all_paths) != set(source.absolute_path):
            raise ValueError("Split manifests do not cover the inspected training folds exactly")
        if set(splits["test"].fold) != {"fold_2"}:
            raise ValueError("Test must be exactly fold_2")
        if any("fold_2" in set(splits[name].fold) for name in ("train", "val")):
            raise ValueError("fold_2 leaked into training or validation")
    return {name: {"images": len(frame), "subjects": len(subjects[name]),
                   "normal": int((frame.class_id.astype(int) == 0).sum()),
                   "blast": int((frame.class_id.astype(int) == 1).sum())}
            for name, frame in splits.items()}


def create_or_load_splits(cfg: dict, source: pd.DataFrame | None = None) -> tuple[dict, dict]:
    folder = classifier_output(cfg) / "splits"
    summary_path = folder / "summary.json"
    if summary_path.exists():
        if any(not (folder / f"{name}.csv").is_file() for name in SPLITS):
            raise ValueError("Incomplete saved C-NMC split; preserve files and repair explicitly")
        splits = {name: pd.read_csv(folder / f"{name}.csv") for name in SPLITS}
        summary = read_json(summary_path)
        validate_splits(splits, source)
        for name in SPLITS:
            digest = hashlib.sha256((folder / f"{name}.csv").read_bytes()).hexdigest()
            if summary.get("manifest_hashes", {}).get(name) != digest:
                raise ValueError(f"Saved {name} split CSV changed; refusing to use it")
        if source is not None and summary["source_fingerprint"] != source_fingerprint(source):
            raise ValueError("C-NMC source changed since split creation; refusing to silently regenerate")
        return splits, summary
    if any((folder / f"{name}.csv").exists() for name in SPLITS):
        raise ValueError("Partial saved C-NMC split exists without summary.json; repair explicitly")
    if source is None:
        raise FileNotFoundError("No split manifests. Run C-NMC inspection and patient split first.")
    candidate = source[source.fold.isin(["fold_0", "fold_1"])]
    test = source[source.fold == "fold_2"].copy()
    if len(candidate) + len(test) != len(source):
        raise ValueError("Unexpected fold in C-NMC training data")
    subject_table = candidate.groupby("subject_id", as_index=False).agg(class_id=("class_id", "first"))
    if candidate.groupby("subject_id").class_id.nunique().max() != 1:
        raise ValueError("A C-NMC subject has both ALL and HEM labels")
    try:
        tr_subjects, va_subjects = train_test_split(
            subject_table.subject_id, test_size=cfg["classifier_val_subject_fraction"],
            random_state=cfg["seed"], stratify=subject_table.class_id)
    except ValueError as exc:
        raise ValueError(f"Cannot make class-stratified subject split: {exc}. Resolve the source; no image-level fallback.") from exc
    splits = {"train": candidate[candidate.subject_id.isin(tr_subjects)].copy(),
              "val": candidate[candidate.subject_id.isin(va_subjects)].copy(), "test": test}
    counts = validate_splits(splits, source)
    summary = {"protocol": "fold_2 fixed test; fold_0+fold_1 subject-stratified train/val",
               "seed": cfg["seed"], "validation_subject_fraction": cfg["classifier_val_subject_fraction"],
               "source_fingerprint": source_fingerprint(source), "counts": counts,
               "class_mapping": {"0": "Normal (HEM)", "1": "Leukemic Blast (ALL)"}}
    for name, frame in splits.items():
        frame.sort_values("relative_path").to_csv(folder / f"{name}.csv", index=False)
    summary["manifest_hashes"] = {name: hashlib.sha256((folder / f"{name}.csv").read_bytes()).hexdigest()
                                  for name in SPLITS}
    summary["split_fingerprint"] = hashlib.sha256(
        json.dumps(summary["manifest_hashes"], sort_keys=True).encode()).hexdigest()
    write_json(summary_path, summary)
    return splits, summary
