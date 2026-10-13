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

## Stage 2 — C-NMC EfficientNet-B3 cell classification and Grad-CAM

Stage 2 is independent of the BCCD detection experiment. Its source is the [Kaggle C-NMC mirror](https://www.kaggle.com/datasets/avk256/cnmc-leukemia) of the [C-NMC-2019 collection at TCIA](https://www.cancerimagingarchive.net/collection/c-nmc-2019/). TCIA lists the original collection under **CC BY 3.0** and requires the dataset citation: Mourya, S., Kant, S., Kumar, P., Gupta, A., & Gupta, R. (2019), *ALL Challenge dataset of ISBI 2019 (C-NMC 2019)*, DOI [10.7937/tcia.2019.dc64i46r](https://doi.org/10.7937/tcia.2019.dc64i46r). Check the Kaggle mirror's displayed terms as well. The [official C-NMC readme](https://dev.cancerimagingarchive.net/wp-content/uploads/CNMC_readme.pdf) documents subject-disjoint folds, patient filename fields, and the labeled training total of 7,272 ALL plus 3,389 HEM cells from 73 subjects. Those are reference counts; notebook 09 reports the files actually found locally.

Class **0 = Normal (HEM)**; class **1 = Leukemic Blast (ALL)**. Here ALL means *Acute Lymphoblastic Leukemia*. This classifier does not change or retrain YOLO. A YOLO WBC crop can later be passed to `predict_cell`, but a combined end-to-end pipeline and dashboard are not implemented.

### Set the existing dataset path once

Keep the already downloaded C-NMC dataset on its current drive. In the existing ignored `config.local.yaml`, add or edit `cnmc_root` to its **actual absolute path**; the example below is illustrative only. Do not overwrite an existing local config containing your BCCD path.

```yaml
cnmc_root: "D:/Datasets/C-NMC_Leukemia"
```

The discovery code accepts `C-NMC_training_data` or `training_data`, a root pointing directly to either directory, or a parent with extra nesting. It verifies `fold_0`, `fold_1`, and `fold_2`, each containing `all` and `hem` directories, case-insensitively. This downloaded layout is supported:

```text
D:/Datasets/C-NMC_Leukemia/
    C-NMC_training_data/
        fold_0/all/    fold_0/hem/
        fold_1/all/    fold_1/hem/
        fold_2/all/    fold_2/hem/
    C-NMC_test_prelim_phase_data/
    C-NMC_test_final_phase_data/
```

Use `cnmc_root: "D:/Datasets/C-NMC_Leukemia"` for this layout. Notebook 08 checks the verified training folder, and notebook 09 prints its resolved path. The preliminary and final challenge folders are excluded from discovery and training manifests. The primary experiment reserves `fold_2` inside `C-NMC_training_data` for held-out evaluation. If more than one valid training root exists beneath the configured path, select the intended one explicitly. Source BMPs are read in place; only manifests, metadata, figures and checkpoints are saved under `outputs/classifier/`. If you do not have the data yet, download manually from Kaggle or, after setting up Kaggle authentication, run `kaggle datasets download -d avk256/cnmc-leukemia -p D:\Datasets\C-NMC-download`; extract into a **new empty** directory rather than over existing data. Kaggle access may require an account. Training never downloads the dataset automatically.

### Reuse the same CUDA environment

Open **Anaconda Prompt** in `CellCancer\leukemia-yolo`:

```bat
conda activate leukemia-yolo
setup_conda.bat
conda activate leukemia-yolo
python -c "import torch,torchvision,pytorch_grad_cam,sklearn; print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
launch_jupyter.bat
```

`setup_conda.bat` adds the minimal Stage 2 packages from `requirements-no-torch.txt` while constraining the installed Torch/Torchvision versions. It stops for manual CUDA PyTorch selection if a compatible CUDA build is missing; use the [official PyTorch selector](https://pytorch.org/get-started/locally/) rather than replacing a working GPU build with a CPU wheel. **The Jupyter kernel must be Python (leukemia-yolo).** Do not use the base Conda environment.

Run these notebooks in order after the Stage 1 notebooks, or independently if Stage 1 has already been completed:

| Notebook | What to check | Main output |
|---|---|---|
| `08_CNMC_Environment_Check` | CUDA, Torch/Torchvision, driver, RAM/VRAM and source path | `outputs/classifier/environment_report.json` |
| `09_CNMC_Dataset_Inspection` | Actual fold/class counts, subjects, duplicate files and example cells | `outputs/classifier/cnmc_manifest.csv` |
| `10_CNMC_Patient_Split` | Saved subject, path and hash-disjoint manifests | `outputs/classifier/splits/` |
| `11_EfficientNet_B3_Training` | One-batch checks, CUDA smoke, head training and fine tuning | `outputs/classifier/runs/<run>/best_model.pth` |
| `12_EfficientNet_B3_Evaluation` | Untouched `fold_2` cell-level metrics | `outputs/classifier/paper_results/` |
| `13_EfficientNet_B3_GradCAM` | Correct/incorrect prediction overlays | `outputs/classifier/gradcam/` |
| `14_Classification_Paper_Results` | Actual evidence and limitations; missing results stay unavailable | `outputs/classifier/paper_results/classifier_experiment_summary.md` |

Notebook 10 fixes `fold_2` as test and makes a class-stratified **subject** split from folds 0 and 1. If the local copy contains overlapping subjects or identical image files, it preserves every fold_2 test image and excludes entire affected subjects from the candidate pool. Subjects connected to those excluded subjects by duplicate images are also excluded. Remaining subjects linked by duplicate images are assigned together to train or validation. Every exclusion is recorded in `outputs/classifier/splits/excluded.csv`; `leakage_report.json` identifies shared IDs, paths and hashes and verifies the final split has no overlap. The summary reports both the requested and actual validation subject fraction. Conflicting ALL/HEM labels for the same image, or insufficient independent subjects after exclusions, stop splitting with the report path. Original images and filenames remain unchanged. These exclusions are included in the experiment metadata and paper summary.

Saved CSVs are reused on rerun; source or manifest changes stop loading. There is no image-level random split. If notebook 10 previously failed during initial creation, copy the updated **source files as well as notebooks** to the target laptop, restart the Jupyter kernel and rerun notebooks 09 and 10. The split cell now reloads `src.cnmc_splits`, prints its absolute file path and **implementation version 2**, and rejects an outdated module before splitting. New notebook text alongside the original generic leakage traceback means the kernel is still executing old source, or the source update went to a different clone. Confirm the printed path is the intended repository. Do not bypass the leakage check or delete successful experiment manifests. Notebook 11 defaults to B3 ImageNet weights, RGB 224×224, batch 8, four head epochs and eight partial fine-tuning epochs, train-only class weights, AMP and validation macro-F1 checkpoint selection. Its 224-pixel square resize differs from the pretrained weights' nominal 300-pixel center crop; the exact preprocessing is stored in the checkpoint. Set `classifier_image_size: 300` only for a separate new experiment. If Windows DataLoader workers fail, set `classifier_workers: 0` in local config or pass `--workers 0`.

### Terminal commands and results

After notebook 10 has saved the manifests, these commands run in the **existing** activated environment:

```bat
python scripts\train_classifier.py --mode smoke
python scripts\train_classifier.py --mode full
python scripts\train_classifier.py --mode full --resume outputs\classifier\runs\YOUR_RUN\last_checkpoint.pth
python scripts\evaluate_classifier.py --checkpoint outputs\classifier\runs\YOUR_RUN\best_model.pth
python scripts\generate_gradcam.py --checkpoint outputs\classifier\runs\YOUR_RUN\best_model.pth --predictions outputs\classifier\paper_results\test_predictions.csv --output outputs\classifier\gradcam
python scripts\predict_classifier.py --checkpoint outputs\classifier\runs\YOUR_RUN\best_model.pth --input outputs\wbc_crops --output outputs\classifier\crop_predictions
```

The training script logs each completed epoch to `training_history.csv` and `training_log.txt` in a timestamped run. `best_model.pth` is selected by validation macro-F1; `last_checkpoint.pth` stores optimizer/scaler/RNG state for resumption. Stop with Ctrl+C or Jupyter Interrupt; resume from the last completed epoch. A genuine CUDA OOM retries only with a smaller batch (8→4→2), logs the change, and never switches training to CPU. If the first epoch fails before any checkpoint, the failed run is preserved and the retry gets a new run directory. `--max-minutes 360` sets an optional six-hour budget checked after each epoch. Use `--mode head` followed by `--mode fine --resume ...` if separate phases are preferred. The full training notebook performs the smoke gate before starting a new full run.

Read `training_history.csv` for loss, validation accuracy, validation macro-F1, blast sensitivity, elapsed time and peak allocated GPU memory. Test metrics, confusion matrices, ROC/PR figures, classification reports and test predictions are in `outputs/classifier/paper_results/`. Grad-CAM overlays and the multi-image figure are in `outputs/classifier/gradcam/`. Notebook 14 writes the paper table, methodology, limitations and summary. It never treats separate YOLO and classifier metrics as a measured combined-system accuracy. Classifier outputs are ignored by Git, as are C-NMC images, `.pth` checkpoints and machine-specific config.

If `cnmc_root` is wrong, notebook 09 names the missing verified fold layout. If a UID filename is unsupported, it stops instead of inventing a subject ID. If split files or source hashes change, preserve the old run for reproducibility and resolve the discrepancy before creating a new experiment. If pretrained weights are unavailable offline, place the official Torchvision weights in the standard PyTorch cache before training; checkpoint inference and evaluation never download weights. CUDA failures in notebook 08 must be fixed before running notebook 11. Grad-CAM regions are model-influential pixels, not proof of a cell feature or clinical diagnosis.

Official resources: [BCCD source](https://github.com/Shenggan/BCCD_Dataset), [Ultralytics training](https://docs.ultralytics.com/modes/train/), [PyTorch CUDA installation](https://pytorch.org/get-started/locally/).
# Current audit and full-backbone experiment

The saved classifier result in `outputs/classifier/paper_results` is the **EfficientNet-B3 baseline**. Its old training script's `full` mode trained the head and final two feature blocks; it did not unfreeze the entire backbone. The separate `scripts/full_finetune.py` experiment uses `--mode full` to train every feature block. It initializes from the verified baseline best checkpoint and writes a unique run under `outputs/classifier/full_finetune/`. The saved C-NMC subject manifests remain authoritative.

Use the existing `leukemia-yolo` Conda interpreter. From the project root:

```powershell
python scripts/audit_project.py
python scripts/full_finetune.py --smoke --batch-size 4 --accumulation 2 --workers 2
python scripts/full_finetune.py --mode full --epochs 20 --batch-size 4 --accumulation 2 --workers 2 --lr 0.00003
python scripts/full_finetune.py --resume outputs/classifier/full_finetune/<run_id>/last_checkpoint.pth --mode full --epochs 20 --batch-size 4 --accumulation 2 --workers 2 --lr 0.00003
```

The smoke mode measures two batches and does not evaluate the test set. The full run checks validation after each epoch, chooses the best checkpoint by validation macro F1, selects the blast threshold on validation predictions, and then evaluates the saved best checkpoint on fold_2 once. `--max-minutes` stops after a completed epoch and leaves a resumable `last_checkpoint.pth`. On CUDA memory exhaustion, start a new run with `--batch-size 2 --accumulation 4`; changing batch settings during exact resume is disallowed. A partial run must be named `--mode partial`.

The old fold_2 test set has already been examined, so future fold_2 results are not fully independent. A fresh subject-disjoint set of leukemia-labeled whole smears is needed for an unbiased end-to-end estimate. BCCD only evaluates WBC localization; C-NMC only evaluates cropped-cell classification.

The integrated demo can be run locally after installing `requirements-no-torch.txt` into the existing environment:

```powershell
python -m uvicorn src.dashboard:app --host 127.0.0.1 --port 8000 --workers 1
```

The dashboard loads the newest completed full-backbone checkpoint when available, otherwise the preserved baseline classifier checkpoint. Results are saved under `outputs/inference/<run_id>`; uploaded source bytes are not copied there. Use `DELETE /runs/<run_id>` to remove one result directory. The displayed model scores are uncalibrated, and the image-level blast fraction uses accepted classified WBCs only.
