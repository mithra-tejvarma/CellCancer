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


def overlap_details(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Identify exactly which subjects, original paths, and hashes overlap."""
    columns = {"subject_id": "subjects", "absolute_path": "paths", "sha256": "hashes"}
    report = {}
    for column, name in columns.items():
        values = sorted(set(left[column]) & set(right[column]))
        examples = pd.concat([left[left[column].isin(values)], right[right[column].isin(values)]])
        report[name] = {"count": len(values), "values": values,
                        "examples": examples[["relative_path", "subject_id", "class_id", "fold", "sha256"]]
                        .head(12).to_dict(orient="records")}
    return report


def _prepare_candidate(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    candidate = source[source.fold.isin(["fold_0", "fold_1"])].copy()
    test = source[source.fold == "fold_2"].copy()
    if len(candidate) + len(test) != len(source):
        raise ValueError("Unexpected fold in C-NMC training data")
    audit = {"candidate_vs_test": overlap_details(candidate, test),
             "policy": "Keep all fold_2 test rows; exclude whole candidate subjects overlapping test by ID or image hash, "
                       "including subjects connected to them by duplicate images"}
    if source.groupby("subject_id").class_id.nunique().max() != 1:
        audit["error"] = "A C-NMC subject has both ALL and HEM labels"
        return candidate, test, source.iloc[:0].copy(), audit
    conflicts = source.groupby("sha256").class_id.nunique()
    conflicts = conflicts[conflicts > 1].index
    if len(conflicts):
        audit["error"] = "Identical image files have conflicting ALL/HEM labels; resolve the source annotations"
        audit["conflicting_labels"] = source[source.sha256.isin(conflicts)].to_dict(orient="records")
        return candidate, test, source.iloc[:0].copy(), audit
    shared_subjects = set(candidate.subject_id) & set(test.subject_id)
    shared_hashes = set(candidate.sha256) & set(test.sha256)
    duplicate_subjects = set(candidate[candidate.sha256.isin(shared_hashes)].subject_id)
    remove_subjects = shared_subjects | duplicate_subjects
    subject_groups = _subject_groups(candidate)
    blocked_groups = set(subject_groups[subject_groups.subject_id.isin(remove_subjects)].group_id)
    remove_subjects = set(subject_groups[subject_groups.group_id.isin(blocked_groups)].subject_id)
    excluded = candidate[candidate.subject_id.isin(remove_subjects)].copy()
    if not excluded.empty:
        def reason(subject):
            reasons = []
            if subject in shared_subjects:
                reasons.append("subject_id_present_in_fold_2")
            if subject in duplicate_subjects:
                reasons.append("subject_contains_image_identical_to_fold_2")
            if not reasons:
                reasons.append("duplicate_linked_to_test_overlapping_subject")
            return ";".join(reasons)
        excluded["exclusion_reason"] = excluded.subject_id.map(reason)
    else:
        excluded["exclusion_reason"] = pd.Series(dtype=str)
    candidate = candidate[~candidate.subject_id.isin(remove_subjects)].copy()
    audit["excluded_subjects"] = sorted(remove_subjects)
    audit["excluded_images"] = len(excluded)
    return candidate, test, excluded, audit


def _subject_groups(candidate: pd.DataFrame) -> pd.DataFrame:
    """Join subjects sharing identical images before assigning train/validation."""
    parent = {subject: subject for subject in sorted(candidate.subject_id.unique())}
    def find(subject):
        while parent[subject] != subject:
            parent[subject] = parent[parent[subject]]
            subject = parent[subject]
        return subject
    for _, cells in candidate.groupby("sha256"):
        members = sorted(cells.subject_id.unique())
        for subject in members[1:]:
            left, right = find(members[0]), find(subject)
            parent[max(left, right)] = min(left, right)
    groups = candidate.groupby("subject_id", as_index=False).agg(class_id=("class_id", "first"))
    groups["group_id"] = groups.subject_id.map(find)
    return groups


def source_fingerprint(df: pd.DataFrame) -> str:
    fields = df[["relative_path", "sha256", "class_id", "subject_id", "fold"]].sort_values("relative_path")
    return hashlib.sha256(fields.to_csv(index=False).encode()).hexdigest()


def validate_splits(splits: dict[str, pd.DataFrame], source: pd.DataFrame | None = None,
                    *, excluded: pd.DataFrame | None = None) -> dict:
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
                details = overlap_details(splits[left], splits[right])
                raise ValueError(
                    f"Leakage between {left} and {right}: "
                    f"subjects={details['subjects']['count']} {details['subjects']['values'][:5]}, "
                    f"paths={details['paths']['count']}, hashes={details['hashes']['count']}. "
                    f"Examples: {details['subjects']['examples'][:2] or details['hashes']['examples'][:2]}")
    if source is not None:
        all_paths = [p for frame in splits.values() for p in frame.absolute_path]
        excluded_paths = [] if excluded is None else excluded.absolute_path.tolist()
        if set(all_paths) & set(excluded_paths):
            raise ValueError("An excluded image was included in an experiment split")
        all_paths += excluded_paths
        if len(all_paths) != len(source) or set(all_paths) != set(source.absolute_path):
            raise ValueError("Split manifests plus documented exclusions must cover the inspected source exactly")
        if set(splits["test"].fold) != {"fold_2"}:
            raise ValueError("Test must be exactly fold_2")
        if any("fold_2" in set(splits[name].fold) for name in ("train", "val")):
            raise ValueError("fold_2 leaked into training or validation")
        if set(splits["test"].absolute_path) != set(source[source.fold == "fold_2"].absolute_path):
            raise ValueError("Test must retain every original fold_2 image")
        if excluded is not None and not excluded.empty:
            if not set(excluded.fold).issubset({"fold_0", "fold_1"}):
                raise ValueError("Only training/validation candidate folds may have exclusions")
            _, _, expected_excluded, audit = _prepare_candidate(source)
            if audit.get("error") or set(expected_excluded.absolute_path) != set(excluded.absolute_path):
                raise ValueError("Saved exclusions disagree with the source leakage policy")
            if set(excluded.subject_id) & (subjects["train"] | subjects["val"]):
                raise ValueError("An excluded subject still appears in train or validation")
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
        excluded = pd.read_csv(folder / "excluded.csv") if (folder / "excluded.csv").is_file() else None
        if excluded is not None and "excluded" not in summary.get("manifest_hashes", {}):
            raise ValueError("excluded.csv is not covered by the saved summary hashes")
        if summary.get("excluded_images", 0) and excluded is None:
            raise ValueError("Missing documented excluded.csv; refusing to use incomplete manifests")
        validate_splits(splits, source, excluded=excluded)
        for name in summary.get("manifest_hashes", {}):
            if name not in (*SPLITS, "excluded") or not (folder / f"{name}.csv").is_file():
                raise ValueError(f"Invalid or missing saved manifest {name}")
            digest = hashlib.sha256((folder / f"{name}.csv").read_bytes()).hexdigest()
            if summary.get("manifest_hashes", {}).get(name) != digest:
                raise ValueError(f"Saved {name} split CSV changed; refusing to use it")
        if not set(SPLITS).issubset(summary.get("manifest_hashes", {})):
            raise ValueError("Saved split summary is missing manifest hashes")
        if source is not None and summary["source_fingerprint"] != source_fingerprint(source):
            raise ValueError("C-NMC source changed since split creation; refusing to silently regenerate")
        return splits, summary
    if any((folder / f"{name}.csv").exists() for name in (*SPLITS, "excluded")):
        raise ValueError("Partial saved C-NMC split exists without summary.json; repair explicitly")
    if source is None:
        raise FileNotFoundError("No split manifests. Run C-NMC inspection and patient split first.")
    candidate, test, excluded, audit = _prepare_candidate(source)
    report_path = folder / "leakage_report.json"
    write_json(report_path, audit)
    if audit.get("error"):
        raise ValueError(f"{audit['error']}. Details: {report_path}")
    subject_table = _subject_groups(candidate)
    group_table = subject_table.groupby("group_id", as_index=False).agg(
        class_id=("class_id", "first"), subjects=("subject_id", "size"))
    try:
        tr_groups, va_groups = train_test_split(
            group_table.group_id, test_size=cfg["classifier_val_subject_fraction"],
            random_state=cfg["seed"], stratify=group_table.class_id)
    except ValueError as exc:
        audit["error"] = f"Insufficient independent subject groups after leakage exclusions: {exc}"
        write_json(report_path, audit)
        raise ValueError(f"Cannot make class-stratified subject split: {exc}. Details: {report_path}. "
                         "No image-level fallback is permitted.") from exc
    tr_subjects = subject_table[subject_table.group_id.isin(tr_groups)].subject_id
    va_subjects = subject_table[subject_table.group_id.isin(va_groups)].subject_id
    splits = {"train": candidate[candidate.subject_id.isin(tr_subjects)].copy(),
              "val": candidate[candidate.subject_id.isin(va_subjects)].copy(), "test": test}
    counts = validate_splits(splits, source, excluded=excluded)
    audit["final_overlap"] = {f"{left}_vs_{right}": overlap_details(splits[left], splits[right])
                              for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]}
    audit["duplicate_linked_subject_groups"] = subject_table.groupby("group_id").subject_id.apply(list).to_dict()
    write_json(report_path, audit)
    summary = {"protocol": "fold_2 fixed test; fold_0+fold_1 subject-stratified train/val; "
                           "test-overlapping candidate subjects excluded; duplicate-linked subjects grouped",
               "seed": cfg["seed"], "validation_subject_fraction": cfg["classifier_val_subject_fraction"],
               "actual_validation_subject_fraction": len(va_subjects) / len(subject_table),
               "excluded_images": len(excluded), "excluded_subjects": audit["excluded_subjects"],
               "exclusion_policy": audit["policy"], "leakage_report": str(report_path),
               "source_fingerprint": source_fingerprint(source), "counts": counts,
               "class_mapping": {"0": "Normal (HEM)", "1": "Leukemic Blast (ALL)"}}
    for name, frame in splits.items():
        frame.sort_values("relative_path").to_csv(folder / f"{name}.csv", index=False)
    excluded.sort_values("relative_path").to_csv(folder / "excluded.csv", index=False)
    summary["manifest_hashes"] = {name: hashlib.sha256((folder / f"{name}.csv").read_bytes()).hexdigest()
                                  for name in (*SPLITS, "excluded")}
    summary["split_fingerprint"] = hashlib.sha256(
        json.dumps(summary["manifest_hashes"], sort_keys=True).encode()).hexdigest()
    write_json(summary_path, summary)
    if len(excluded):
        print(f"WARNING — excluded {len(excluded)} candidate cells from {len(audit['excluded_subjects'])} subjects "
              f"to prevent test leakage. fold_2 unchanged. Audit: {report_path}")
    return splits, summary
