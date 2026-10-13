"""Generate clean Stage 2 notebooks without touching Stage 1 notebooks."""
from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"
BOOT = '''from pathlib import Path
import os, sys
search = [Path.cwd(), *Path.cwd().parents]
if os.environ.get("LEUKEMIA_YOLO_ROOT"):
    search.insert(0, Path(os.environ["LEUKEMIA_YOLO_ROOT"]))
for candidate in search:
    if (candidate / "config.example.yaml").is_file() and (candidate / "src" / "classifier_config.py").is_file():
        sys.path.insert(0, str(candidate)); break
from src.classifier_config import classifier_config, classifier_output
cfg = classifier_config()
out = classifier_output(cfg)
print("Project:", cfg["root"])
print("Config:", cfg["config_path"])
print("C-NMC source:", cfg["cnmc_root"])'''


def book(name, title, purpose, inputs, outputs, steps, next_book):
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n**Purpose:** {purpose}\n\n**Required inputs:** {inputs}\n\n**Saved outputs:** {outputs}"),
             nbf.v4.new_markdown_cell("## 1. Load central configuration\n\nSet `cnmc_root` once in ignored `config.local.yaml`. Source images are never moved or changed."),
             nbf.v4.new_code_cell(BOOT)]
    for heading, explanation, code in steps:
        cells.extend([nbf.v4.new_markdown_cell(f"## {heading}\n\n{explanation}"), nbf.v4.new_code_cell(code)])
    cells.append(nbf.v4.new_markdown_cell(f"## Next\n\n{next_book}"))
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python (leukemia-yolo)",
        "language": "python", "name": "leukemia-yolo"}, "language_info": {"name": "python"}})
    for index, cell in enumerate(nb.cells):
        cell.id = hashlib.sha1(f"{name}:{index}:{cell.source}".encode()).hexdigest()[:8]
    nbf.write(nb, DEST / name)


book("08_CNMC_Environment_Check.ipynb", "08 — C-NMC classifier environment", "Check the existing Conda/Jupyter/CUDA environment and C-NMC path.",
     "Existing leukemia-yolo environment; `cnmc_root` in config.local.yaml.", "outputs/classifier/environment_report.json.", [
 ("2. Inspect hardware and packages", "System RAM and dedicated NVIDIA VRAM are separate resources.", '''import importlib.metadata, platform, shutil, subprocess, sys
import pandas as pd
import psutil, torch, torchvision
from src.config import write_json
checks = []
def add(check, value, status): checks.append({"check": check, "value": str(value), "status": status})
add("Python executable", sys.executable, "PASS")
add("Python version", sys.version.split()[0], "PASS" if sys.version_info[:2] == (3, 11) else "WARNING")
add("Conda environment / Jupyter kernel", os.environ.get("CONDA_DEFAULT_ENV", "Not set"), "PASS" if os.environ.get("CONDA_DEFAULT_ENV") == "leukemia-yolo" else "WARNING")
add("OS / CPU", f"{platform.platform()} / {platform.processor()}", "PASS" if sys.platform == "win32" else "WARNING")
ram = psutil.virtual_memory()
add("System RAM total / available GiB", f"{ram.total/2**30:.2f} / {ram.available/2**30:.2f}", "PASS")
add("Output disk free GiB", f"{shutil.disk_usage(out).free/2**30:.2f}", "PASS")
add("PyTorch / Torchvision", f"{torch.__version__} / {torchvision.__version__}", "PASS")
add("CUDA build / available", f"{torch.version.cuda} / {torch.cuda.is_available()}", "PASS" if torch.cuda.is_available() else "ERROR")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    add("NVIDIA GPU", torch.cuda.get_device_name(0), "PASS")
    add("Dedicated VRAM total / free GiB", f"{total/2**30:.2f} / {free/2**30:.2f}", "PASS")
else: add("NVIDIA GPU", "CUDA unavailable; GPU training blocked", "ERROR")
for package in ("Pillow", "scikit-learn", "grad-cam", "PyYAML"):
    try: add(package, importlib.metadata.version(package), "PASS")
    except importlib.metadata.PackageNotFoundError: add(package, "missing", "ERROR")
from src.cnmc_data import discover_training_data
try:
    training_root, _ = discover_training_data(cfg["cnmc_root"])
    add("C-NMC verified training directory", training_root, "PASS")
except (FileNotFoundError, ValueError) as exc:
    add("C-NMC verified training directory", exc, "ERROR")
add("Existing YOLO source", cfg["root"] / "src" / "training.py", "PASS" if (cfg["root"] / "src" / "training.py").is_file() else "ERROR")
try:
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    add("NVIDIA driver", driver.stdout.strip() or driver.stderr.strip(), "PASS" if driver.returncode == 0 else "WARNING")
except (OSError, subprocess.TimeoutExpired): add("NVIDIA driver", "nvidia-smi unavailable", "WARNING")
display(pd.DataFrame(checks))
write_json(out / "environment_report.json", {"checks": checks, "cuda_ready": torch.cuda.is_available(), "gpu_device": 0})'''),
 ("3. Gate and saved report", "A missing CUDA build is visible now and stops training later.", '''assert (out / "environment_report.json").is_file()
print("PASS — CUDA and report ready" if torch.cuda.is_available() else "ERROR — report saved; install compatible CUDA PyTorch before notebook 11")''')],
     "Open **09_CNMC_Dataset_Inspection.ipynb** after setting the real C-NMC path.")

book("09_CNMC_Dataset_Inspection.ipynb", "09 — C-NMC source inspection", "Discover labeled training folds, decode cells, audit counts and patient IDs.",
     "C-NMC `C-NMC_training_data/fold_0..2/{all,hem}` beneath cnmc_root; `training_data` is also supported. Preliminary/final challenge folders are excluded.",
     "cnmc_manifest.csv, dataset_inspection.json, sample and class-distribution figures.", [
 ("2. Verify source and metadata", "The official 10,661-image count is a comparison, not an assumed local result.", '''from src.cnmc_data import inspect_cnmc
import pandas as pd
frame, report = inspect_cnmc(cfg)
print("Verified training folder:", report["training_data"])
display(pd.DataFrame([{"metric": key, "value": str(value)} for key, value in report.items() if key != "errors"]))
display(frame.groupby(["fold", "class_name"]).size().rename("cells").reset_index())
display(frame.groupby(["subject_id", "class_name"]).size().rename("cells").reset_index().head(20))
display(frame[["image_width", "image_height", "image_channels"]].describe())'''),
 ("3. Save labeled gallery", "Eight normal and eight blast examples are selected when available.", '''from src.cnmc_data import sample_figures
from IPython.display import Image as DisplayImage, display
sample_figures(frame, out / "figures")
display(DisplayImage(filename=str(out / "figures" / "cnmc_samples.png")))
display(DisplayImage(filename=str(out / "figures" / "class_distribution.png")))'''),
 ("4. Integrity summary", "Invalid subject names, damaged images, or duplicate image content block splitting.", '''assert report["ready"] and frame.class_id.isin([0, 1]).all()
print("PASS — source untouched; manifest saved. Official count match:", report["matches_official_counts"])''')],
     "Open **10_CNMC_Patient_Split.ipynb**; never split images randomly.")

book("10_CNMC_Patient_Split.ipynb", "10 — Subject-disjoint C-NMC splits", "Freeze fold_2 as test and stratify subjects in folds 0+1 for train/validation.",
     "Validated cnmc_manifest.csv from notebook 09.", "outputs/classifier/splits/{train,val,test}.csv and summary.json.", [
 ("2. Load source manifest", "Only labeled training folds are included; preliminary/final challenge images are excluded.", '''import pandas as pd
from src.cnmc_data import inspect_cnmc
source, inspection = inspect_cnmc(cfg)
assert inspection["ready"]
display(source.groupby(["fold", "class_name"]).size().rename("cells").reset_index())'''),
 ("3. Create or reuse immutable split", "Reruns reuse saved CSVs and compare the source fingerprint.", '''from src.cnmc_splits import create_or_load_splits, validate_splits
splits, summary = create_or_load_splits(cfg, source)
display(pd.DataFrame(summary["counts"]).T)
print(summary["protocol"])
assert not (set(splits["train"].subject_id) & set(splits["val"].subject_id))'''),
 ("4. Leakage gate", "Subjects, absolute paths and SHA-256 image content must not cross splits.", '''verified = validate_splits(splits, source)
assert set(splits["test"].fold) == {"fold_2"}
print("PASS — subject/path/content disjoint; saved manifests point to original images")''')],
     "Open **11_EfficientNet_B3_Training.ipynb** for the CUDA smoke test and two-phase training.")

book("11_EfficientNet_B3_Training.ipynb", "11 — EfficientNet-B3 training", "Train a two-logit Normal/Blast classifier with a CUDA smoke test and head/fine phases.",
     "Saved subject splits; CUDA-enabled PyTorch; ImageNet weights download or cache.",
     "Timestamped outputs/classifier/runs/<run>/ checkpoints, logs, CSV and plots.", [
 ("A. GPU and data readiness", "GPU training never falls back silently to CPU.", '''from src.classifier_training import require_classifier_cuda, class_weights
from src.cnmc_splits import create_or_load_splits, validate_splits
import torch
device = require_classifier_cuda()
splits, split_info = create_or_load_splits(cfg)
display(__import__("pandas").DataFrame(validate_splits(splits)).T)
print("GPU:", torch.cuda.get_device_name(0), "free GiB:", round(torch.cuda.mem_get_info(0)[0]/2**30, 2))
print("Image size:", cfg["classifier_image_size"], "batch:", cfg["classifier_batch_size"])
print("Class weights [Normal, Blast]:", class_weights(splits["train"], device).tolist())'''),
 ("B. Model architecture", "Torchvision's DEFAULT ImageNet B3 weights; final head returns exactly two logits.", '''from src.classifier_model import build_model, set_trainable, metadata
model = build_model(pretrained=True).to(device)
set_trainable(model, "head")
print(model.classifier)
print(metadata(cfg["classifier_image_size"]))'''),
 ("C. Preview the deterministic validation batch", "At 224 pixels this differs from B3's original 300-pixel center-crop recipe; the choice is recorded.", '''from src.classifier_transforms import CNMCDataset
from torch.utils.data import DataLoader
batch = next(iter(DataLoader(CNMCDataset(splits["val"], cfg["classifier_image_size"]), batch_size=2)))
print("Batch RGB tensor:", batch[0].shape, "labels:", batch[1].tolist())
assert batch[0].shape[1:] == (3, cfg["classifier_image_size"], cfg["classifier_image_size"])'''),
 ("D. One-batch forward and backward check", "Only the new head receives gradients in phase A.", '''from torch import nn
model.train(); model.features.eval()
x, y = batch[0].to(device), batch[1].to(device)
logits = model(x); assert logits.shape == (2, 2)
loss = nn.CrossEntropyLoss()(logits, y); loss.backward()
assert model.classifier[-1].weight.grad is not None
print("PASS — forward/backward; loss:", float(loss.detach()))
model.zero_grad(set_to_none=True); del x, y, logits, loss'''),
 ("E. Launch one-epoch smoke test", "The guarded child script handles Windows workers. Logs stream live and are saved in its run folder.", '''import subprocess, sys
def stream_command(arguments):
    command = [sys.executable, "-u", *map(str, arguments)]
    process = subprocess.Popen(command, cwd=cfg["root"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in process.stdout: print(line, end="", flush=True)
    except KeyboardInterrupt:
        process.terminate(); process.wait()
        raise RuntimeError("Training stopped. Resume from the run's last_checkpoint.pth")
    code = process.wait()
    if code: raise RuntimeError(f"Child process exited {code}; inspect training_log.txt")
stream_command([cfg["root"] / "scripts" / "train_classifier.py", "--mode", "smoke"])'''),
 ("F. Verify smoke artifacts", "Full training is gated on a successful CUDA checkpoint.", '''from src.classifier_model import load_checkpoint
smokes = sorted((out / "runs").glob("smoke_*/best_model.pth"))
assert smokes, "No successful CUDA smoke checkpoint"
smoke_model, smoke_info, _ = load_checkpoint(smokes[-1], device)
print("PASS — loaded smoke checkpoint:", smokes[-1])
del smoke_model'''),
 ("G. Run full two-phase training", "Separate timestamped run. Stop with Interrupt; resume with --resume last_checkpoint.pth.", '''stream_command([cfg["root"] / "scripts" / "train_classifier.py", "--mode", "full"])
print("PASS — child process completed")'''),
 ("H. Review outputs and GPU memory", "History is written after each complete epoch; incomplete CSV lines are ignored by a fresh read after exit.", '''import pandas as pd
from src.classifier_config import latest_classifier_run
run = latest_classifier_run(cfg)
history = pd.read_csv(run / "training_history.csv")
display(history.tail(10))
print("Child-process peak allocated VRAM GiB:", round(history.peak_vram_bytes.max()/2**30, 2))
print("Training log:", run / "training_log.txt")'''),
 ("I. Inspect curves and selected checkpoint", "The best model is selected by validation macro-F1, never test metrics.", '''from IPython.display import Image as DisplayImage, display
for filename in ("loss_curve.png", "accuracy_curve.png", "f1_curve.png"):
    display(DisplayImage(filename=str(run / filename)))
best_model, best_info, selected = load_checkpoint(run / "best_model.pth", device)
print("PASS — best checkpoint:", run / "best_model.pth")
print("Validation macro-F1:", selected["validation_macro_f1"], "class mapping:", best_info["class_mapping"])''')],
     "Open **12_EfficientNet_B3_Evaluation.ipynb**. Test fold_2 is used only after model selection.")

book("12_EfficientNet_B3_Evaluation.ipynb", "12 — Held-out classifier evaluation", "Evaluate the selected checkpoint on fixed fold_2 only.",
     "A non-smoke best_model.pth and saved subject manifests.", "outputs/classifier/paper_results/classification_metrics.*, plots and predictions.", [
 ("2. Select exact checkpoint", "Pin the run you want to report; never choose it using test results.", '''from src.classifier_config import latest_classifier_run
checkpoint = latest_classifier_run(cfg) / "best_model.pth"
print("Selected checkpoint:", checkpoint)'''),
 ("3. Evaluate test fold only", "Positive class 1 is ALL/Blast. The script saves checkpoint SHA-256 to prevent result mixing.", '''from src.classifier_evaluation import evaluate
from src.cnmc_splits import create_or_load_splits
splits, summary = create_or_load_splits(cfg)
print("Test cells:", len(splits["test"]), "subjects:", splits["test"].subject_id.nunique())
metrics = evaluate(cfg, checkpoint)
display(__import__("pandas").DataFrame([metrics]).T)'''),
 ("4. Review diagnostic figures", "Include false negatives in subsequent error analysis.", '''from IPython.display import Image as DisplayImage, display
for name in ("confusion_matrix.png", "confusion_matrix_normalized.png", "roc_curve.png", "pr_curve.png", "evaluation_examples.png"):
    file = out / "paper_results" / name
    if file.is_file(): display(DisplayImage(filename=str(file)))
assert metrics["split"] == "fold_2" and metrics["positive_class"] == "Leukemic Blast (ALL)"
print("PASS — held-out cell-level metrics saved; not patient-level clinical performance")''')],
     "Open **13_EfficientNet_B3_GradCAM.ipynb** to explain selected predictions.")

book("13_EfficientNet_B3_GradCAM.ipynb", "13 — EfficientNet-B3 Grad-CAM", "Visualize pixels influencing predicted classes for real held-out examples.",
     "Selected best_model.pth, test_predictions.csv, grad-cam package.", "outputs/classifier/gradcam/ images and summary.", [
 ("2. Load checkpoint and test categories", "Correct blast/normal and FP/FN categories are drawn from actual test predictions.", '''import pandas as pd
from src.classifier_config import latest_classifier_run
checkpoint = latest_classifier_run(cfg) / "best_model.pth"
predictions = out / "paper_results" / "test_predictions.csv"
assert predictions.is_file(), "Run notebook 12 first"
frame = pd.read_csv(predictions)
display(pd.crosstab(frame.class_id, frame.predicted_class_id, rownames=["Actual"], colnames=["Predicted"]))'''),
 ("3. Generate spatial CAMs", "The final feature block has spatial maps. CAM targets the predicted logit and requires gradients in eval mode.", '''from src.classifier_gradcam import generate_gradcam
summary = generate_gradcam(checkpoint, predictions, out / "gradcam")
print(summary["category_counts"])
print("Generated examples:", len(summary["generated"]))'''),
 ("4. Review and validate saved figure", "An absent error category is reported, never fabricated. CAM does not prove morphology.", '''from IPython.display import Image as DisplayImage, display
grid = out / "gradcam" / "gradcam_publication_grid.png"
if grid.is_file(): display(DisplayImage(filename=str(grid)))
assert (out / "gradcam" / "gradcam_summary.json").is_file()
print("PASS — Grad-CAM saved; interpret with domain experts")''')],
     "Open **14_Classification_Paper_Results.ipynb** for evidence-based reporting.")

book("14_Classification_Paper_Results.ipynb", "14 — C-NMC classification paper results", "Assemble paper tables and prose from actual experiment files only.",
     "Outputs from notebooks 09–13 when available.", "outputs/classifier/paper_results/classifier_* files.", [
 ("2. Assemble evidence", "Missing experiments remain explicitly Not available.", '''from src.classifier_reporting import assemble_classifier_report
import pandas as pd
report = assemble_classifier_report(cfg)
display(pd.DataFrame([{"item": key, "value": str(value)} for key, value in report.items()]))'''),
 ("3. Review figures and limitations", "Validation and held-out test results have separate names and provenance.", '''from IPython.display import Image as DisplayImage, display
for file in (out / "paper_results" / "confusion_matrix.png", out / "paper_results" / "roc_curve.png",
             out / "gradcam" / "gradcam_publication_grid.png"):
    if file.is_file(): display(DisplayImage(filename=str(file)))
try:
    from src.classifier_config import latest_classifier_run
    latest = latest_classifier_run(cfg)
    for name in ("loss_curve.png", "f1_curve.png"):
        if (latest / name).is_file(): display(DisplayImage(filename=str(latest / name)))
except FileNotFoundError:
    print("Training curves: Not available")
print(report["limitations"])'''),
 ("4. Confirm export", "No missing metric is replaced by an invented number.", '''for name in ("classifier_experiment_summary.md", "classifier_results_table.csv", "classifier_methodology.md", "classifier_limitations.md"):
    assert (out / "paper_results" / name).is_file()
print("PASS — paper documents exported to", out / "paper_results")''')],
     "Review the generated report and figures. Integrating YOLO crops into a combined pipeline is a later stage.")
