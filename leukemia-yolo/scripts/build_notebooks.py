"""Regenerate the eight clean, executable Jupyter notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"
DEST.mkdir(exist_ok=True)

BOOT = '''from pathlib import Path
import os, sys
search = [Path.cwd(), *Path.cwd().parents]
if os.environ.get("LEUKEMIA_YOLO_ROOT"):
    search.insert(0, Path(os.environ["LEUKEMIA_YOLO_ROOT"]))
for candidate in search:
    if (candidate / "config.example.yaml").is_file() and (candidate / "src" / "config.py").is_file():
        sys.path.insert(0, str(candidate))
        break
from src.config import load_config, ensure_outputs
cfg = load_config()
out = ensure_outputs(cfg)
os.chdir(cfg["root"])
print("Project root:", cfg["root"])
print("Configuration:", cfg["config_path"])'''


def notebook(name, title, purpose, inputs, outputs, steps, next_step):
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n**Purpose:** {purpose}\n\n**Required inputs:** {inputs}\n\n**Saved outputs:** {outputs}"),
             nbf.v4.new_markdown_cell("## 1. Load the central configuration\n\nThe root is found from the notebook location or the installed package. Set `LEUKEMIA_YOLO_CONFIG` only for synthetic smoke tests or an alternate config."),
             nbf.v4.new_code_cell(BOOT)]
    for heading, explanation, code in steps:
        cells += [nbf.v4.new_markdown_cell(f"## {heading}\n\n{explanation}"), nbf.v4.new_code_cell(code)]
    cells.append(nbf.v4.new_markdown_cell(f"## Next notebook\n\n{next_step}"))
    book = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python (leukemia-yolo)",
        "language": "python", "name": "leukemia-yolo"}, "language_info": {"name": "python"}})
    nbf.write(book, DEST / name)


notebook("00_Environment_Check.ipynb", "00 — Environment and CUDA check",
         "Verify the active Conda kernel, Windows resources, PyTorch/CUDA, packages, disk and writable outputs.",
         "`config.local.yaml`; installed Python environment. No dataset is needed.",
         "`outputs/environment_report.json`, `outputs/environment_packages.txt`.", [
 ("2. Inspect environment", "PASS/WARNING/FAIL are colored. System RAM and NVIDIA dedicated VRAM are separate fields.",
  '''from src.environment import inspect_environment, status_table
report = inspect_environment(cfg)
table = status_table(report)
display(table[["check", "value", "status", "detail"]])'''),
 ("3. Verify GPU gate", "A CPU-only wheel or unavailable driver produces an explicit failure. Training also checks this gate in its child process.",
  '''from src.environment import require_cuda
try:
    gpu = require_cuda(cfg["device"])
    print("PASS — CUDA GPU:", gpu)
except RuntimeError as exc:
    print("FAIL — CUDA training unavailable:", exc)'''),
 ("4. Saved report and status", "Reports are generated even when CUDA is absent, so the development laptop can complete this notebook.",
  '''assert (out / "environment_report.json").is_file()
assert (out / "environment_packages.txt").is_file()
print("PASS — environment report saved" if report["cuda_ready"] and report["kernel_ready"] else
      "WARNING — report saved; resolve FAIL/WARNING rows before GPU training")''')],
 "Open **01_BCCD_Dataset_Inspection.ipynb** after setting the dataset path in `config.local.yaml`.")

notebook("01_BCCD_Dataset_Inspection.ipynb", "01 — BCCD source inspection",
         "Audit real microscopy files, VOC boxes, source classes and split-file contents.",
         "Configured BCCD root with `JPEGImages/`, `Annotations/`, `ImageSets/Main/` (direct or nested `BCCD/`).",
         "`outputs/dataset_inspection.json`, original gallery and class chart.", [
 ("2. Inspect source files", "Image sizes come from image pixels, not XML paths or stale XML dimensions. A standard release is near 364 images; variations are allowed.",
  '''from src.bccd import inspect_dataset, locate_bccd
from IPython.display import display
import pandas as pd
source_root = locate_bccd(cfg["dataset_root"])
print("Source:", source_root)
print("ImageSets/Main exists:", (source_root / "ImageSets" / "Main").is_dir())
inspection = inspect_dataset(cfg, strict=False)
display(pd.DataFrame([{"measure": k, "value": inspection[k]} for k in
    ("total_images", "total_xml", "total_wbc_boxes", "images_with_wbc", "images_without_wbc",
     "missing_xml", "unreadable_images", "invalid_boxes_or_xml")]))
display(pd.DataFrame(inspection["class_distribution"].items(), columns=["class", "boxes"]))
display(pd.DataFrame([{"file": k, "lines": len(v), "first_entries": v[:5]} for k,v in inspection["source_split_files"].items()]))'''),
 ("3. View real images and XML", "VOC boxes are color coded: RBC red, WBC green, platelets blue. XML examples are read from the configured dataset.",
  '''from src.visualization import source_gallery, class_chart
if inspection["records"]:
    display(source_gallery(inspection["records"], out / "figures" / "original_bccd_examples.png", count=9))
    display(class_chart(inspection["class_distribution"], out / "figures" / "source_class_counts.png"))
    xml_example = Path(inspection["records"][0]["xml"])
    print("XML example:", xml_example.name)
    print(xml_example.read_text(encoding="utf-8")[:1600])
    display(pd.DataFrame([{"image": r["id"], "pixels": f"{r['width']}x{r['height']}",
                           "xml_size": r["xml_size"], "clipped_boxes": len(r["clipping"])}
                          for r in inspection["records"][:10]]))'''),
 ("4. Integrity gate", "Stop here if an image, annotation or bounding box is missing or unreadable.",
  '''assert inspection["ready"], "FAIL — source integrity errors; inspect outputs/dataset_inspection.json"
print("PASS — source inspection complete. Next: notebook 02.")''')],
 "Open **02_BCCD_Preprocessing.ipynb**.")

notebook("02_BCCD_Preprocessing.ipynb", "02 — VOC to YOLO WBC preprocessing",
         "Create an idempotent, single-class WBC training dataset with frozen, leakage-free splits.",
         "Successful notebook 01; configured BCCD source.",
         "`data/processed/BCCD_YOLO/`, `outputs/preprocessing_report.json`, split manifests and before/after figure.", [
 ("2. Convert and preserve splits", "Only WBC objects become class 0. Coordinates are clipped to actual image bounds and transformed with continuous-edge VOC formulas. Source files are never changed.",
  '''from src.preprocessing import preprocess, conversion_table
report = preprocess(cfg)
display(report)
print("Split policy:", report["split_reason"])
print("Clipped boxes:", len(report["clipped_boxes"]))'''),
 ("3. Inspect round-trip and before/after", "The table shows source VOC coordinates, normalized YOLO values and reversible pixel error.",
  '''from src.bccd import inspect_dataset
from src.visualization import before_after
records = inspect_dataset(cfg)["records"]
selected = next((r for r in records if any(o["class"] == "WBC" for o in r["objects"])), None)
assert selected is not None, "No WBC examples in source"
table = conversion_table(selected)
display(table)
assert (table["roundtrip_max_error_px"] < 1e-4).all()
display(before_after(selected, cfg["processed_data_path"], out / "figures" / "preprocessing_examples.png"))'''),
 ("4. Verify saved dataset", "This checks labels, source provenance, disjoint IDs and cross-split image hashes.",
  '''from src.validation import validate_processed
validation = validate_processed(cfg["processed_data_path"], out / "validation_report.json")
display(pd.DataFrame(validation["splits"]).T if "pd" in globals() else validation["splits"])
assert validation["ready"] and (out / "splits" / "manifest.json").is_file()
print("PASS — processed WBC dataset ready. Next: notebook 03.")''')],
 "Open **03_Dataset_Visualization.ipynb**.")

notebook("03_Dataset_Visualization.ipynb", "03 — Processed dataset visualization",
         "Review nine labeled examples, split counts and WBC box geometry before training.",
         "Completed notebook 02 and its processed dataset.",
         "`outputs/figures/processed_wbc_examples.png`, `outputs/figures/wbc_box_dimensions.png`, validation report.", [
 ("2. Validate all images and labels", "Reject invalid classes, missing labels, source mismatch, overlapping splits and identical cross-split pixels.",
  '''from src.validation import validate_processed
import pandas as pd
validation = validate_processed(cfg["processed_data_path"], out / "validation_report.json")
display(pd.DataFrame(validation["splits"]).T)
assert all(v["wbc_boxes"] > 0 for v in validation["splits"].values())'''),
 ("3. Plot WBC examples and box sizes", "Visual review remains necessary: verify green boxes surround leukocytes and no RBC-only boxes slipped in.",
  '''from src.visualization import processed_gallery, box_distribution
display(processed_gallery(cfg["processed_data_path"], out / "figures" / "processed_wbc_examples.png", count=9, seed=cfg["seed"]))
display(box_distribution(validation["box_dimensions"], out / "figures" / "wbc_box_dimensions.png"))
display(pd.DataFrame(validation["box_dimensions"]).describe())'''),
 ("4. Training readiness", "Programmatic integrity passes; inspect the image grid before starting GPU work.",
  '''assert validation["ready"]
assert (out / "figures" / "processed_wbc_examples.png").is_file()
print("PASS — data checks complete. Proceed to notebook 04 after visually reviewing boxes.")''')],
 "Open **04_YOLOv8_Training.ipynb** on the CUDA laptop.")

notebook("04_YOLOv8_Training.ipynb", "04 — CUDA YOLOv8n WBC training",
         "Run one CUDA smoke epoch, verify checkpoints, then run or resume full transfer learning on Windows.",
         "Valid processed data, `Python (leukemia-yolo)` kernel, CUDA-enabled PyTorch and `yolov8n.pt` download/cache access.",
         "Timestamped runs in `outputs/checkpoints/` with best.pt, last.pt, results.csv, results.png, logs and effective config.", [
 ("A. Check GPU and dataset readiness", "CUDA failure stops this notebook. System RAM and dedicated NVIDIA VRAM are separate resources.",
  '''from src.training import training_readiness, stream_training, checkpoint_for_run, results_table, progress_columns
from src.environment import nvidia_smi
from IPython.display import display, Image as IPImage
import pandas as pd
ready = training_readiness(cfg)
display(pd.DataFrame([{"setting": k, "value": str(v)} for k,v in ready.items()]))
print("nvidia-smi:", nvidia_smi())'''),
 ("B. Load pretrained detection checkpoint", "This must be the YOLOv8n **detection** checkpoint. First use may download it from Ultralytics.",
  '''from ultralytics import YOLO
model = YOLO("yolov8n.pt")
assert model.task == "detect", f"Unexpected task: {model.task}"
print("PASS — pretrained object detector loaded:", cfg["model"])'''),
 ("C. Show effective plan", "Default batch 0.75 asks Ultralytics AutoBatch to target a fraction of GPU memory. Automatic disk caching requires Windows to identify an SSD and enough free space; unknown media type disables cache.",
  '''from src.training import choose_cache
plan = {"epochs": cfg["epochs"], "imgsz": cfg["imgsz"], "batch": cfg["batch"],
        "device": cfg["device"], "amp": cfg["amp"], "workers": cfg["workers"],
        "patience": cfg["patience"], "seed": cfg["seed"], "plots": True,
        "save": True, "save_period": 5, "cache": choose_cache(cfg)}
display(pd.DataFrame(plan.items(), columns=["setting", "value"]))
print("If Windows DataLoader stalls, set workers: 0 in config.local.yaml and start a new run.")'''),
 ("D. Run one smoke epoch", "Training runs in a Windows-safe child process through `sys.executable`; its stdout and stderr stream here and to a log. Set `RESUME_LAST` to an unfinished last.pt to skip a new smoke run.",
  '''RESUME_LAST = None  # Optional Path(".../weights/last.pt") from an interrupted main run.
ALLOW_BATCH_FALLBACK = True  # Logged AutoBatch/OOM retry at batch=4 in a NEW run; failed run is preserved.
smoke_run = None
if RESUME_LAST is None:
    smoke_run = stream_training(cfg, kind="smoke", allow_batch_fallback=ALLOW_BATCH_FALLBACK)
    print("Smoke run:", smoke_run)
else:
    RESUME_LAST = Path(RESUME_LAST).expanduser().resolve()
    print("Resuming existing main run from", RESUME_LAST)'''),
 ("E. Verify smoke checkpoint", "A completed one-epoch CUDA run, loadable last.pt and results.csv are required before main training.",
  '''if smoke_run:
    smoke_last = checkpoint_for_run(smoke_run, "last.pt")
    smoke_model = YOLO(str(smoke_last))
    assert smoke_model.task == "detect" and len(results_table(smoke_run)) >= 1
    print("PASS — CUDA smoke training and checkpoint load succeeded:", smoke_last)
else:
    assert RESUME_LAST.is_file()
    print("PASS — existing last.pt selected for resume")'''),
 ("F. Train or resume main experiment", "The main run requires the completed smoke run. Stopping leaves last.pt; resume by setting `RESUME_LAST` above. Reruns use new timestamped directories.",
  '''main_run = stream_training(cfg, kind="main", smoke_run=smoke_run, resume=RESUME_LAST,
                           allow_batch_fallback=ALLOW_BATCH_FALLBACK)
print("Main run:", main_run)'''),
 ("G. Review saved training outputs", "A completed run must contain both checkpoints, CSV, plot and metadata.",
  '''from src.config import read_json
metadata = read_json(main_run / "run_metadata.json")
for name in ("weights/best.pt", "weights/last.pt", "results.csv", "results.png", "effective_training_config.json"):
    assert (main_run / name).is_file(), name
display(metadata)
display(IPImage(filename=str(main_run / "results.png")))'''),
 ("H. GPU utilization and peak VRAM", "These values describe this run; 100% utilization is neither required nor promised.",
  '''display(pd.DataFrame([{"GPU": metadata["gpu_start"]["gpu_name"],
                         "peak allocated GiB": metadata.get("gpu_peak_allocated_gib"),
                         "actual batch": metadata.get("actual_batch"),
                         "nvidia-smi at end": metadata.get("nvidia_smi_end")}]))'''),
 ("I. Latest epoch progress", "The CSV reader tolerates a partially written last row during a running experiment.",
  '''progress = progress_columns(results_table(main_run))
display(progress.tail(10))
assert not progress.empty and metadata["completed_epochs"] > 0'''),
 ("J. Load best checkpoint and finish", "This checkpoint is used for later held-out evaluation and inference.",
  '''best_path = checkpoint_for_run(main_run)
best_model = YOLO(str(best_path))
assert best_model.task == "detect"
print("PASS — completed CUDA training. Best checkpoint:", best_path)
print("Validation metrics are in results.csv; held-out test metrics are produced only by notebook 05.")''')],
 "Open **05_Model_Evaluation.ipynb**. Never tune hyperparameters from its held-out test results.")

notebook("05_Model_Evaluation.ipynb", "05 — Held-out WBC localization evaluation",
         "Evaluate the completed best.pt on the test split only, with official Ultralytics detection metrics.",
         "Completed main run from notebook 04 and unchanged processed dataset.",
         "`outputs/paper_results/detection_metrics.csv/.json` and `outputs/figures/test_evaluation/<run>/`.", [
 ("2. Resolve exact checkpoint and test size", "Checkpoint provenance is checked against the frozen dataset fingerprint.",
  '''from src.training import resolve_checkpoint
from src.validation import validate_processed
checkpoint = resolve_checkpoint(cfg)
data_check = validate_processed(cfg["processed_data_path"])
print("Checkpoint:", checkpoint)
print("Held-out test images:", data_check["splits"]["test"]["images"])'''),
 ("3. Run test evaluation", "The shared evaluator invokes `YOLO(best.pt).val(data=..., split='test', plots=True)` and reads box precision/recall/mAP.",
  '''from src.evaluation import evaluate, evaluation_figure_paths
evaluation = evaluate(cfg)
import pandas as pd
display(pd.DataFrame([evaluation["metrics"]]))
print("Evaluation figures:", evaluation["figures"])'''),
 ("4. Show curves and confusion matrix", "Framework plots are generated from the held-out split. Missing optional plot files are labeled explicitly.",
  '''from IPython.display import display, Image as IPImage
for name,path in evaluation_figure_paths(evaluation).items():
    print(name, path if path.is_file() else "Not available")
    if path.is_file(): display(IPImage(filename=str(path)))'''),
 ("5. Saved result and interpretation", "Metrics measure WBC localization. They do not measure leukemia or blast classification.",
  '''assert evaluation["split"] == "test"
assert (out / "paper_results" / "detection_metrics.json").is_file()
print("PASS — test-only WBC localization results saved for this checkpoint. Next: notebook 06.")''')],
 "Open **06_WBC_Detection.ipynb**.")

notebook("06_WBC_Detection.ipynb", "06 — WBC detection and unlabeled crops",
         "Predict on held-out microscopy images, save confidence-labeled boxes and optional WBC crops.",
         "Completed best.pt; at least six held-out test images; optional custom image path.",
         "Timestamped `outputs/predictions/`, `outputs/wbc_crops/`, and `outputs/prediction_metadata.csv`.", [
 ("2. Choose inference options", "NMS uses the configured IoU threshold. Crops are WBC detections only; no Normal/Blast labels exist.",
  '''CUSTOM_IMAGE = None  # Optional Path to another microscope image.
CONFIDENCE = cfg["confidence"]
IOU_NMS = cfg["iou"]
CROP_WBCS = cfg["crop_wbcs"]
print("Confidence:", CONFIDENCE, "NMS IoU:", IOU_NMS, "Save crops:", CROP_WBCS)'''),
 ("3. Predict at least six held-out examples", "The shared function loads completed best.pt and records its SHA-256 in metadata.",
  '''from src.inference import predict
result = predict(cfg, custom=CUSTOM_IMAGE, limit=6, conf=CONFIDENCE, iou=IOU_NMS, crop=CROP_WBCS)
import pandas as pd
display(pd.DataFrame(result["examples"]))
print("Metadata:", result["metadata_csv"])
print("Crops:", result["crop_dir"])'''),
 ("4. Compare originals and detections", "Confidence values and counts are shown with each saved prediction.",
  '''import matplotlib.pyplot as plt
from PIL import Image
fig, axes = plt.subplots(len(result["examples"]), 2, figsize=(14, 5*len(result["examples"])))
for row, pair in zip(axes, result["examples"]):
    with Image.open(pair["source"]) as original, Image.open(pair["prediction"]) as pred:
        row[0].imshow(original.convert("RGB")); row[1].imshow(pred.convert("RGB"))
    row[0].set_title("Original: " + Path(pair["source"]).name)
    row[1].set_title(f"Predicted WBCs: {pair['detected_wbcs']}")
    for ax in row: ax.axis("off")
fig.tight_layout()
display(fig)
plt.close(fig)
metadata_rows = pd.read_csv(result["metadata_csv"])
display(metadata_rows.head(20))'''),
 ("5. Verify outputs", "Every metadata row has coordinates, confidence, threshold, crop filename and model identity.",
  '''assert len(result["examples"]) >= 6
assert Path(result["metadata_csv"]).is_file()
print("PASS — WBC detection artifacts saved. Next: notebook 07.")''')],
 "Open **07_Conference_Results.ipynb**.")

notebook("07_Conference_Results.ipynb", "07 — Conference results from observed artifacts",
         "Compile source counts, split sizes, hardware, training details and WBC localization results without invented values.",
         "Outputs of earlier notebooks. Missing experiments are shown as `Not available`.",
         "`outputs/paper_results/experiment_summary.md/.csv`, `experiment_metadata.json`, paper figures.", [
 ("2. Build source-backed summary", "Test metrics are accepted only if the dataset and checkpoint hashes match the current completed run.",
  '''from src.reporting import build_summary, summary_table, LIMITATION
summary = build_summary(cfg)
display(summary_table(summary))
print(LIMITATION)'''),
 ("3. Show validation training curves", "The training results.csv contains validation metrics; test metrics come only from notebook 05.",
  '''from IPython.display import display, Image as IPImage
for label in ("loss_curves", "precision_recall_curve", "confusion_matrix"):
    artifact = summary[label]
    print(label, artifact)
    if artifact != "Not available" and Path(artifact).is_file():
        display(IPImage(filename=str(artifact)))'''),
 ("4. Show example detections", "These are WBC localization predictions, without blast or normal cell labels.",
  '''examples = summary["example_detections"]
if examples == "Not available":
    print("Example detections: Not available")
else:
    for path in examples:
        display(IPImage(filename=str(path)))'''),
 ("5. Validate export schema", "The summary is suitable for drafting a paper only after real data, CUDA training and held-out evaluation are complete.",
  '''from src.config import read_json
saved = read_json(out / "paper_results" / "experiment_metadata.json")
assert saved["limitations"] == LIMITATION
assert (out / "paper_results" / "experiment_summary.csv").is_file()
assert (out / "paper_results" / "experiment_summary.md").is_file()
print("PASS — provenance-aware paper tables saved." if saved["test_mAP50"] != "Not available" else
      "WARNING — paper tables saved with unavailable results; complete notebooks 04–06 on the CUDA laptop.")''')],
 "Stage 1 ends here. EfficientNet-B3, Grad-CAM and blast quantification require separately labeled leukemia data and are future work.")

print("Created", len(list(DEST.glob("*.ipynb"))), "notebooks in", DEST)
