import pandas as pd
import pytest
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits, validate_splits
from src.config import read_json


def test_subject_split_and_immutable_reuse(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    frame, _ = inspect_cnmc(cfg)
    first, summary = create_or_load_splits(cfg, frame)
    second, again = create_or_load_splits(cfg, frame)
    assert summary == again
    assert summary["counts"]["test"]["images"] == 4
    assert set(first["test"].fold) == {"fold_2"}
    assert validate_splits(second, frame)


def test_subject_and_hash_leakage(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    frame, _ = inspect_cnmc(cfg)
    splits, _ = create_or_load_splits(cfg, frame)
    broken = {key: value.copy() for key, value in splits.items()}
    broken["val"].loc[broken["val"].index[0], "subject_id"] = broken["train"].subject_id.iloc[0]
    with pytest.raises(ValueError, match="Leakage"):
        validate_splits(broken)
    broken = {key: value.copy() for key, value in splits.items()}
    broken["val"].loc[broken["val"].index[0], "sha256"] = broken["train"].sha256.iloc[0]
    with pytest.raises(ValueError, match="hashes=1"):
        validate_splits(broken)


def test_subject_overlap_with_test_excludes_candidate_subject(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    source, _ = inspect_cnmc(cfg)
    candidate_subject = source[(source.fold != "fold_2") & (source.class_id == 1)].subject_id.iloc[0]
    test_subject = source[(source.fold == "fold_2") & (source.class_id == 1)].subject_id.iloc[0]
    source.loc[source.subject_id == candidate_subject, "subject_id"] = test_subject
    original = source.copy(deep=True)
    splits, summary = create_or_load_splits(cfg, source)
    excluded = pd.read_csv(cfg["output_path"] / "classifier" / "splits" / "excluded.csv")
    assert summary["excluded_images"] == 2
    assert summary["excluded_subjects"] == [test_subject]
    assert len(splits["test"]) == 4
    assert test_subject not in set(splits["train"].subject_id) | set(splits["val"].subject_id)
    assert excluded.exclusion_reason.str.contains("subject_id_present_in_fold_2").all()
    validate_splits(splits, source, excluded=excluded)
    reused, again = create_or_load_splits(cfg, source)
    assert again == summary
    pd.testing.assert_frame_equal(source, original)


def test_test_duplicate_removes_whole_candidate_subject(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    source, _ = inspect_cnmc(cfg)
    candidate_index = source[(source.fold != "fold_2") & (source.class_id == 0)].index[0]
    subject = source.loc[candidate_index, "subject_id"]
    test_hash = source[(source.fold == "fold_2") & (source.class_id == 0)].sha256.iloc[0]
    source.loc[candidate_index, "sha256"] = test_hash
    splits, summary = create_or_load_splits(cfg, source)
    assert summary["excluded_images"] == 2 and summary["excluded_subjects"] == [subject]
    assert len(splits["test"]) == 4
    assert subject not in set(splits["train"].subject_id) | set(splits["val"].subject_id)
    audit = read_json(cfg["output_path"] / "classifier" / "splits" / "leakage_report.json")
    assert audit["candidate_vs_test"]["hashes"]["count"] == 1
    assert all(part["subjects"]["count"] == part["hashes"]["count"] == part["paths"]["count"] == 0
               for part in audit["final_overlap"].values())


def test_duplicate_linked_candidate_subjects_stay_together(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    source, _ = inspect_cnmc(cfg)
    candidates = source[(source.fold != "fold_2") & (source.class_id == 1)]
    subjects = candidates.subject_id.unique()[:2]
    first_index = candidates[candidates.subject_id == subjects[0]].index[0]
    second_index = candidates[candidates.subject_id == subjects[1]].index[0]
    source.loc[second_index, "sha256"] = source.loc[first_index, "sha256"]
    splits, summary = create_or_load_splits(cfg, source)
    assert summary["excluded_images"] == 0
    assert any(set(subjects).issubset(set(splits[name].subject_id)) for name in ("train", "val"))
    validate_splits(splits, source)


def test_conflicting_duplicate_labels_are_not_automatically_resolved(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    source, _ = inspect_cnmc(cfg)
    normal_index = source[source.class_id == 0].index[0]
    source.loc[normal_index, "sha256"] = source[source.class_id == 1].sha256.iloc[0]
    with pytest.raises(ValueError, match="conflicting ALL/HEM labels"):
        create_or_load_splits(cfg, source)
    folder = cfg["output_path"] / "classifier" / "splits"
    audit = read_json(folder / "leakage_report.json")
    assert len(audit["conflicting_labels"]) == 2
    assert not (folder / "train.csv").exists()


def test_test_exclusions_propagate_across_duplicate_subject_links(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    source, _ = inspect_cnmc(cfg)
    subjects = source[(source.fold != "fold_2") & (source.class_id == 1)].subject_id.unique()[:3]
    indices = [source[source.subject_id == subject].index.tolist() for subject in subjects]
    source.loc[indices[0][0], "sha256"] = source[(source.fold == "fold_2") & (source.class_id == 1)].sha256.iloc[0]
    source.loc[indices[1][0], "sha256"] = source.loc[indices[0][1], "sha256"]
    source.loc[indices[2][0], "sha256"] = source.loc[indices[1][1], "sha256"]
    splits, summary = create_or_load_splits(cfg, source)
    assert summary["excluded_images"] == 6
    assert set(summary["excluded_subjects"]) == set(subjects)
    assert len(splits["test"]) == 4


def test_exclusion_manifest_tampering_is_rejected(synthetic_cnmc):
    _, _, _, cfg = synthetic_cnmc
    source, _ = inspect_cnmc(cfg)
    create_or_load_splits(cfg, source)
    path = cfg["output_path"] / "classifier" / "splits" / "excluded.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="excluded split CSV changed"):
        create_or_load_splits(cfg, source)
