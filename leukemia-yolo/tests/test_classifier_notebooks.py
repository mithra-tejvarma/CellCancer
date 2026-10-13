import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_structure():
    for number in range(8, 15):
        matches = list((ROOT / "notebooks").glob(f"{number:02d}_*.ipynb"))
        assert len(matches) == 1
        nb = nbformat.read(matches[0], as_version=4)
        nbformat.validate(nb)
        assert nb.metadata.kernelspec.name == "leukemia-yolo"
        assert sum(cell.cell_type == "code" for cell in nb.cells) >= 3
        assert all(not cell.get("outputs") for cell in nb.cells if cell.cell_type == "code")


def test_nontraining_notebooks_with_synthetic_cnmc(synthetic_cnmc, monkeypatch):
<<<<<<< HEAD
    dataset_root, training, config_path, cfg = synthetic_cnmc
    training.rename(dataset_root / "C-NMC_training_data")
    for name in ("C-NMC_test_final_phase_data", "C-NMC_test_prelim_phase_data"):
        (dataset_root / name).mkdir()
=======
    _, _, config_path, cfg = synthetic_cnmc
>>>>>>> f5f3e3e40fcb7dd65b681e3014c49a9f3044dac3
    monkeypatch.setenv("LEUKEMIA_YOLO_CONFIG", str(config_path))
    monkeypatch.setenv("LEUKEMIA_YOLO_ROOT", str(ROOT))
    for number in (8, 9, 10, 14):
        path = next((ROOT / "notebooks").glob(f"{number:02d}_*.ipynb"))
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    base = cfg["output_path"] / "classifier"
    assert (base / "splits" / "test.csv").is_file()
    assert (base / "paper_results" / "classifier_experiment_summary.md").is_file()
