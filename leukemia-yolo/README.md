# Explainable Deep Learning Framework for Automated Leukemia Blast Cell Detection in Blood Smears

**Current implementation: Stage 1 — Kaggle BCCD → YOLOv8n → WBC localization.** This project targets the [Orvile Kaggle BCCD archive](https://www.kaggle.com/datasets/orvile/bccd-blood-cell-count-and-detection-dataset), which already contains `train/`, `val/`, and `test/` folders with one JSON annotation per image. The original Pascal VOC release remains supported for compatibility. BCCD contains RBC, WBC and platelet detection boxes, but no dependable normal-versus-leukemic-blast labels. The model detects leukocytes; it does not diagnose leukemia or classify blasts. EfficientNet-B3, Grad-CAM and blast quantification are future stages and are not implemented here.

## Windows ASUS setup

1. Install [Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) and open **Anaconda Prompt**. Do not train from the base environment.
2. Clone this repository and enter `leukemia-yolo`:

   ```bat
   git clone https://github.com/mithra-tejvarma/CellCancer.git
   cd CellCancer\leukemia-yolo
   setup_conda.bat
   ```

3. The setup script creates `leukemia-yolo` from `environment.yml`, activates it for its own checks, and checks for a compatible CUDA-enabled PyTorch/torchvision pair. If missing, it stops. **Activation inside the batch file ends when the file exits**, so first run `conda activate leukemia-yolo` yourself. Use the **official [PyTorch installation selector](https://pytorch.org/get-started/locally/)** with Windows / Pip / Python / a CUDA wheel supported by the installed NVIDIA driver. Run its exact install command in that activated environment. No CUDA Toolkit version is guessed by this project. Check the driver, then verify the PyTorch install:

   ```bat
   nvidia-smi
   conda activate leukemia-yolo
   REM Run the exact pip command shown by the official PyTorch selector here.
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
   setup_conda.bat
   ```

   The second setup run installs remaining dependencies using constraints that pin the installed `torch` and `torchvision` versions, installs this repository as an editable package, registers `python -m ipykernel install --user --name leukemia-yolo --display-name "Python (leukemia-yolo)"`, and rechecks imports and CUDA. If dependency resolution conflicts, stop and inspect the conflict instead of replacing the CUDA build.

4. Download and extract the [Kaggle BCCD archive](https://www.kaggle.com/datasets/orvile/bccd-blood-cell-count-and-detection-dataset) outside this repository. Point the config at the extracted directory containing `meta.json` and the three split folders:

   ```text
   D:/Datasets/BCCD_Kaggle/
       meta.json
       train/img/*.jpeg       train/ann/*.jpeg.json
       val/img/*.jpeg         val/ann/*.jpeg.json
       test/img/*.jpeg        test/ann/*.jpeg.json
   ```

   Copy the template and set **one** source path:

   ```bat
   copy config.example.yaml config.local.yaml
   notepad config.local.yaml
   ```

   Set `dataset_root: "D:/Datasets/BCCD_Kaggle"` to your actual extracted directory. `config.local.yaml` is ignored by Git. Relative processed and output paths resolve against this project root. An extra one-level wrapper around the split folders is detected automatically.

5. Launch JupyterLab:

   ```bat
   launch_jupyter.bat
   ```

   **The Jupyter kernel MUST point to the `leukemia-yolo` Conda environment.** Choose **Python (leukemia-yolo)** in Jupyter. Open `notebooks/00_Environment_Check.ipynb` first. Notebook 00 prints the Python executable and Conda environment so you can verify the selection. Do not use the base Anaconda environment for model training.

## Notebook order and artifacts

| Notebook | Action | Key output |
|---|---|---|
| `00_Environment_Check` | Verify Windows, packages, RAM, disk, driver, CUDA, VRAM, kernel | `outputs/environment_report.json` |
| `01_BCCD_Dataset_Inspection` | Audit all three Kaggle image/JSON folders and source boxes | `outputs/dataset_inspection.json` |
| `02_BCCD_Preprocessing` | Convert JSON WBC boxes and preserve Kaggle splits | `data/processed/BCCD_YOLO/data.yaml` |
| `03_Dataset_Visualization` | Verify nine examples and box distributions | `outputs/figures/processed_wbc_examples.png` |
| `04_YOLOv8_Training` | One CUDA smoke epoch, then full training | `outputs/checkpoints/<run>/weights/best.pt` |
| `05_Model_Evaluation` | Evaluate best.pt on held-out **test** only | `outputs/paper_results/detection_metrics.json` |
| `06_WBC_Detection` | Predict test/custom images; save optional crops | `outputs/predictions/`, `outputs/wbc_crops/` |
| `07_Conference_Results` | Compile actual evidence for a paper | `outputs/paper_results/experiment_summary.md` |

The Kaggle archive contains **205 train, 87 validation, and 72 test images**. Their original allocation is kept exactly. The notebook files are committed with empty cell outputs. Running them locally creates figures and reports under ignored `outputs/`. Source and processed images, model weights, image caches, Conda environments and local config are ignored by Git. Do not commit downloaded BCCD content or checkpoint files.

The current Kaggle archive has two zero-area **RBC** rectangles, one in train and one in val. Inspection reports them and excludes those invalid non-target boxes while keeping both images. All 372 WBC boxes remain valid. An invalid WBC box, missing JSON file, or cross-split duplicate image stops preprocessing. The original VOC layout is still recognized, but its XML files and split text files are not required for this Kaggle release.

## Training from a terminal

The training notebook launches `scripts/train_yolo.py` in an unbuffered child process. This uses the Windows `if __name__ == "__main__"` multiprocessing guard and streams logs into both the notebook and a persistent `.log` file. The same stages are available in Anaconda Prompt after `conda activate leukemia-yolo`:

```bat
python -m src.environment
python -m src.bccd
python -m src.preprocessing
python -m src.validation
python -m src.visualization
python scripts/train_yolo.py --mode smoke
python scripts/train_yolo.py --mode main --smoke-run outputs\checkpoints\smoke_<ID>
python scripts/evaluate_yolo.py
python scripts/predict_yolo.py --limit 6
python -m src.reporting
```

The smoke command prints `RUN_DIR=...` when its one epoch and checkpoint verification complete. Supply that exact directory to `--smoke-run`. A main run cannot start without a successful smoke run of the same processed dataset. To resume an interrupted main run:

```bat
python scripts/train_yolo.py --mode main --resume outputs\checkpoints\main_<ID>\weights\last.pt
```

In notebook 04, set `RESUME_LAST` to that `last.pt` path, then run its cells in order. A completed run cannot be resumed. Each new run has its own timestamped directory and preserves previous checkpoints. `results.csv` contains epoch, box/class/DFL losses and **validation** precision/recall/mAP; `results.png` plots training progress. `run_metadata.json` records duration, completed epochs, checkpoint, GPU and peak allocated VRAM. `effective_training_config.json` records the requested settings, and `actual_batch` is recorded when Ultralytics exposes it. `best.pt` is used by notebooks 05 and 06; `last.pt` is for interrupted-run resume.

Default training settings are `epochs=50`, `imgsz=640`, `batch=0.75` (Ultralytics automatic GPU-memory fraction), `device=0`, `amp=true`, `workers=2`, `patience=15`, `seed=42`, `plots=true`, `save=true`, `save_period=5`. `cache=auto` selects disk caching only when Windows reports an SSD and enough free space; if media type is unknown it disables caching unless you explicitly set `cache: disk` after verifying the drive. A 6 GB NVIDIA GPU has about 6 GB of **dedicated** CUDA VRAM; integrated Intel graphics memory does not add to it. Full GPU utilization is not a requirement. If auto batch calibration or CUDA OOM fails, inspect the saved log. Notebook 04 explicitly enables a logged retry at batch 4 in a **new** run while retaining the failed run; set `ALLOW_BATCH_FALLBACK=False` to disable it. The CLI offers the same behavior with `--allow-batch-fallback`. If the worker process stalls on Windows, set `workers: 0` in `config.local.yaml` and start a new run. GPU training never silently falls back to CPU.

## Evaluation and research reporting

Notebook 05 invokes Ultralytics `model.val(..., split="test", plots=True)` only after training is complete. It exports precision, recall, mAP@0.5 and mAP@0.5:0.95 with a SHA-256 of `best.pt`. Test metrics are not used to choose hyperparameters. Notebook 07 includes validation metrics from training separately from held-out test metrics. If an experiment was not run, the table says **Not available**. It will reject metrics from a different checkpoint or processed dataset. `outputs/figures/test_evaluation/<timestamp>/` contains framework confusion matrix, PR and F1 curves. Prediction examples, crops and CSV/JSON metadata are stored separately. Crops have no Normal/Blast labels.

> The BCCD-based YOLOv8 experiment evaluates leukocyte localization only. Blast classification is a separate future stage trained using leukemia-labeled cell images.

## Common problems

- **CUDA unavailable / `torch.version.cuda is None`:** verify `nvidia-smi`, select a compatible CUDA wheel from the official PyTorch page, rerun `setup_conda.bat`, and select the registered Jupyter kernel. Do not proceed with GPU training until notebook 00 passes CUDA checks.
- **Wrong Python executable / base environment:** run `conda activate leukemia-yolo`, rerun kernel registration, and select **Python (leukemia-yolo)**. The notebook displays `sys.executable`.
- **Missing Kaggle files:** `dataset_root` must resolve to the folder containing `meta.json`, `train/img`, `train/ann`, `val/img`, `val/ann`, `test/img`, and `test/ann`. JSON names match image names plus `.json`, for example `BloodImage_00007.jpeg` → `BloodImage_00007.jpeg.json`. Notebook 01 reports missing/unreadable files and invalid WBC boxes. Kaggle splits are never reshuffled; if split integrity fails, repair the source download.
- **Existing processed dataset differs from source:** use a new `processed_data_path` in local config. The project intentionally does not overwrite the old dataset or its split manifest.
- **Ultralytics download fails:** verify network access or download the official `yolov8n.pt` detection checkpoint to this project root. Never substitute a `-cls` checkpoint.
- **Windows DataLoader error:** set `workers: 0` and start a new smoke run. Training itself runs as a child process.
- **CUDA OOM:** explicitly enable the batch fallback or choose a smaller positive integer batch in local config, then run a fresh smoke experiment. The failed run and log remain available.
- **Missing paper metric:** complete notebook 04, then 05. Notebook 07 never substitutes synthetic results.

## Local verification and ASUS checks

On any Python 3.11 development laptop with the nontraining requirements installed:

```bat
python -m compileall src scripts tests
python -m pytest -q
python scripts/train_yolo.py --help
python scripts/evaluate_yolo.py --help
python scripts/predict_yolo.py --help
```

The tests generate both exact-layout Kaggle JPEG/JSON fixtures and original VOC JPEG/XML fixtures, execute selected nontraining notebooks, and check conversion, split preservation, leakage, report schema and notebook validity. They do **not** establish real BCCD detection accuracy. On the ASUS laptop, verify the real dataset in notebooks 01–03, perform the short CUDA smoke training and checkpoint load in notebook 04, then full training, held-out test evaluation and prediction smoke test. Report only metrics produced by that run.

Official resources: [BCCD source](https://github.com/Shenggan/BCCD_Dataset), [Ultralytics training](https://docs.ultralytics.com/modes/train/), [PyTorch CUDA installation](https://pytorch.org/get-started/locally/).
