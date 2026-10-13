import pandas as pd
import pytest
from src.cnmc_data import inspect_cnmc
from src.cnmc_splits import create_or_load_splits, validate_splits


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
    with pytest.raises(ValueError, match="Leakage"):
        validate_splits(broken)
