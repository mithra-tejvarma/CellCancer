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


@pytest.mark.parametrize("directory_name", ["C-NMC_training_data", "c-nmc_TRAINING_DATA"])
def test_user_archive_layout_excludes_challenge_test_images(synthetic_cnmc, directory_name):
    root, training, _, cfg = synthetic_cnmc
    actual_training = root / directory_name
    training.rename(actual_training)
    for test_name in ("C-NMC_test_final_phase_data", "C-NMC_test_prelim_phase_data"):
        challenge = root / test_name
        challenge.mkdir()
        # These files must never be decoded or added to a labeled training manifest.
        (challenge / "unlabeled_test.bmp").write_bytes(b"not a training image")
        for fold in range(3):
            for label in ("all", "hem"):
                (challenge / "training_data" / f"fold_{fold}" / label).mkdir(parents=True)
    assert discover_training_data(root)[0] == actual_training.resolve()
    assert discover_training_data(actual_training)[0] == actual_training.resolve()
    frame, report = inspect_cnmc(cfg)
    assert report["total_images"] == 24
    assert report["training_data"] == str(actual_training.resolve())
    assert all(Path(path).is_relative_to(actual_training) for path in frame.absolute_path)
    with pytest.raises(ValueError, match="challenge test data"):
        discover_training_data(root / "C-NMC_test_final_phase_data")


def test_prefixed_training_folder_in_deep_wrappers(synthetic_cnmc):
    root, training, _, _ = synthetic_cnmc
    wrapped = root / "archive" / "export" / "C-NMC_training_data"
    wrapped.parent.mkdir(parents=True)
    training.rename(wrapped)
    assert discover_training_data(root)[0] == wrapped.resolve()
