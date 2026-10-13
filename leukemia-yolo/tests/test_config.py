from pathlib import Path
import pytest
import yaml

from src.config import load_config, project_root
from src.reporting import LIMITATION, build_summary


def test_relative_paths_and_model_guard(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump({"processed_data_path":"data/processed/other", "output_path":str(tmp_path / "out")}),encoding="utf-8")
    config = load_config(path)
    assert config["processed_data_path"] == (project_root() / "data/processed/other").resolve()
    path.write_text(yaml.safe_dump({"model":"yolov8n-cls.pt"}),encoding="utf-8")
    with pytest.raises(ValueError,match="detection weights"):
        load_config(path)


def test_report_schema_without_experiment(synthetic_bccd):
    _, _, config = synthetic_bccd
    report = build_summary(config)
    assert report["test_mAP50"] == "Not available"
    assert report["limitations"] == LIMITATION
    assert (config["output_path"] / "paper_results/experiment_metadata.json").is_file()


def test_notebooks_are_valid_and_have_clean_outputs():
    import nbformat
    files = sorted((project_root() / "notebooks").glob("*.ipynb"))
    assert len(files) == 21
    for path in files:
        book = nbformat.read(path,as_version=4)
        nbformat.validate(book)
        if path.name[:2] in {"21", "22", "23", "24", "25", "26"}:
            assert book.metadata.kernelspec.name == "leukemia-yolo"
        else:
            assert book.metadata.kernelspec.name in {"leukemia-yolo", "python3"}
        assert len([c for c in book.cells if c.cell_type == "code"]) >= 3
        if path.name[:2] in {"21", "22", "23", "24", "25", "26"}:
            assert all(not c.outputs and c.execution_count is None for c in book.cells if c.cell_type == "code")
        assert "D:/Datasets/" not in path.read_text(encoding="utf-8")


def test_conda_and_yaml_contract():
    import yaml
    root = project_root()
    environment = yaml.safe_load((root / "environment.yml").read_text(encoding="utf-8"))
    template = yaml.safe_load((root / "config.example.yaml").read_text(encoding="utf-8"))
    assert environment["name"] == "leukemia-yolo"
    assert "python=3.11" in environment["dependencies"]
    assert all(any(str(d).startswith(p) for d in environment["dependencies"])
               for p in ("pip", "jupyterlab", "notebook", "ipykernel"))
    assert template["model"] == "yolov8n.pt" and template["batch"] == .75
    assert "torch" not in (root / "requirements-no-torch.txt").read_text(encoding="utf-8").splitlines()
    setup = (root / "setup_conda.bat").read_text(encoding="utf-8")
    assert "--name leukemia-yolo --display-name" in setup
    assert "torch.version.cuda" in setup


def test_nontraining_notebooks_with_synthetic_data(synthetic_kaggle,monkeypatch):
    import nbformat
    from nbclient import NotebookClient
    _, path, config = synthetic_kaggle
    monkeypatch.setenv("LEUKEMIA_YOLO_CONFIG", str(path))
    monkeypatch.setenv("LEUKEMIA_YOLO_ROOT", str(project_root()))
    for name in ("00_Environment_Check", "01_BCCD_Dataset_Inspection", "02_BCCD_Preprocessing",
                 "03_Dataset_Visualization", "07_Conference_Results"):
        book = nbformat.read(project_root() / "notebooks" / f"{name}.ipynb",as_version=4)
        NotebookClient(book,timeout=120,kernel_name="python3",resources={"metadata":{"path":str(project_root())}}).execute()
    assert (config["output_path"] / "paper_results/experiment_summary.md").is_file()
