from pathlib import Path
import pytest

from src.cnmc_data import discover_training_data, inspect_cnmc


def test_nested_discovery_and_inspection(synthetic_cnmc):
    root, training, _, cfg = synthetic_cnmc
    found, layout = discover_training_data(root)
    assert found == training.resolve() and len(layout) == 3
    assert discover_training_data(training)[0] == found
    frame, report = inspect_cnmc(cfg)
    assert len(frame) == 24 and report["subject_count"] == 12
    assert report["class_counts"] == {"ALL": 12, "HEM": 12}
    assert not report["matches_official_counts"]
    assert (cfg["output_path"] / "classifier" / "cnmc_manifest.csv").is_file()
    assert (training / "fold_0").is_dir()


def test_missing_and_ambiguous_root(synthetic_cnmc):
    root, training, _, _ = synthetic_cnmc
    with pytest.raises(FileNotFoundError, match="cnmc_root"):
        discover_training_data(root / "absent")
    other = root / "other" / "training_data"
    for fold in range(3):
        for label in ("all", "hem"):
            (other / f"fold_{fold}" / label).mkdir(parents=True)
    with pytest.raises(ValueError, match="Multiple valid"):
        discover_training_data(root)


def test_corruption_and_bad_filename(synthetic_cnmc):
    _, training, _, cfg = synthetic_cnmc
    (training / "fold_0" / "all" / "not_a_uid.bmp").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity failed"):
        inspect_cnmc(cfg)
