# Verified conference tables

All paths below are repository-relative. “fold_2” is the 3,553-cell internal C-NMC test partition. It was previously examined during development. The integrated BCCD demonstration is qualitative and has no blast ground truth.

## 1. Dataset and unit of analysis

| Dataset | Task | Train | Validation | Test | Unit and label |
|---|---|---:|---:|---:|---|
| BCCD local Kaggle JSON archive | WBC detection | 205 images, 214 WBC boxes | 87 images, 87 boxes | 72 images, 71 boxes | Smear image, WBC box; 372 WBC boxes total |
| C-NMC labeled training subset | HEM versus ALL cell classification | 4,667 cells, 30 subjects | 1,185 cells, 8 subjects | 3,553 cells, 31 subjects | Segmented cell; 0=HEM, 1=ALL |

Source: `outputs/preprocessing_report.json`, `outputs/audit/project_audit.json`, `outputs/classifier/improvement_audit/audit.json`, `outputs/classifier/splits/summary.json`. The 72 BCCD images and the 3,553 C-NMC cells are different test sets.

## 2. YOLOv8n training configuration

| Item | Recorded setting |
|---|---|
| Initialization | `yolov8n.pt` detector |
| Task/classes | Single-class WBC detection |
| Image size | 640 px |
| Epochs | 50 completed |
| Batch | Requested auto-batch fraction 0.75; actual batch 16 |
| GPU/AMP/workers | CUDA device 0; AMP true; 2 workers |
| Optimizer and augmentation | Ultralytics defaults for this run; exact resolved per-parameter policy not separately verified from saved metadata |
| Loss | YOLO box, classification and distribution-focal components in `results.csv` |
| Best checkpoint | `outputs/checkpoints/main_20261013T090007Z_a25a7934/weights/best.pt`, SHA-256 `f75aed580dcc19a180600a727d896299a3e10e4e77b027f7b912cd5cf509c439` |

Source: `outputs/checkpoints/main_20261013T090007Z_a25a7934/run_metadata.json`, `effective_training_config.json`, `results.csv`.

## 3. Detector test metrics

| Split | Images | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| BCCD test | 72 | 0.9450 | 1.0000 | 0.9646 | 0.7707 |

Source: `outputs/paper_results/detection_metrics.json`. These are **WBC localization** results, not blast detection metrics.

## 4. EfficientNet-B3 full fine-tuning configuration

| Item | Recorded setting |
|---|---|
| Initialization | Baseline best checkpoint, SHA-256 `4cef5713bbbd6e55206322a8e5d12c67ce5dc8b5f5a6f4f7e2b078a7c8fc6717`; original baseline used ImageNet initialization |
| Input | RGB, bicubic square resize to 224 × 224; ImageNet mean/std |
| Classes | 0 Normal/HEM; 1 Blast/ALL |
| Training | All feature blocks trainable; BatchNorm running statistics frozen |
| Optimizer/loss | AdamW, weighted cross-entropy; inverse-frequency weights from train partition only |
| Learning rate/weight decay | 3 × 10⁻⁵ / 10⁻⁴ |
| Batch/accumulation | 4 / 2 (effective gradient batch 8) |
| Augmentation | Horizontal flip, rotation ±10°, brightness and contrast ±8% |
| Epochs/selection | 20 completed; best epoch 16 by validation macro-F1 |
| Threshold | 0.05 selected on validation macro-F1 search; at the search floor |
| Full checkpoint | `outputs/classifier/full_finetune/full_20261013T142418Z_d2b3a8ab/best_model.pth`, SHA-256 `f9844aee67b55275c04988c062312559258158d8354ed9007126cd367a1b6a75` |

Source: full-run `training_config.json`, `training_history.csv`, `preprocessing_config.json`, `threshold_config.json`, `test_metrics.json`; `outputs/classifier/improvement_audit/audit.json`.

## 5. Classifier test comparison

| C-NMC fold_2 metric | Baseline (threshold 0.50) | Full fine-tune (threshold 0.05) |
|---|---:|---:|
| Accuracy | 0.7748 | 0.8382 |
| Balanced accuracy | 0.6790 | 0.7592 |
| Blast sensitivity | 0.9292 | 0.9654 |
| Normal specificity | 0.4288 | 0.5529 |
| Macro-F1 | 0.6956 | 0.7851 |
| ROC-AUC | 0.7205 | 0.8434 |

Source: `outputs/classifier/paper_results/classification_metrics.json` and full-run `test_metrics.json`. The different thresholds affect operating-point metrics. The improvement cannot be attributed solely to changing model weights. ROC-AUC is threshold independent.

## 6. Full model confusion matrix

| Actual / predicted | Normal | Blast | Row total |
|---|---:|---:|---:|
| Normal (HEM) | TN 606 | FP 490 | 1,096 |
| Blast (ALL) | FN 85 | TP 2,372 | 2,457 |
| Column total | 691 | 2,862 | 3,553 |

Positive class is ALL blast. `(606+2372)/3553=0.8381649`; `2372/2457=0.9654050`; `606/1096=0.5529197`.

## 7. Exploratory specificity experiments

| Arm | Split | Threshold | Normal specificity | Blast sensitivity | Macro-F1 | Status |
|---|---|---:|---:|---:|---:|---|
| Existing full checkpoint, alternative validation threshold | Validation | 0.890 | 0.900 | 0.951 | 0.916 | Analysis only |
| Unchanged-policy continuation | Validation | 0.044 | 0.926 | 0.951 | 0.924 | Exploratory |
| Mild augmentation continuation | Validation | 0.034 | 0.919 | 0.953 | 0.924 | Exploratory |
| Unweighted-loss continuation | Validation | 0.040 | 0.933 | 0.951 | 0.927 | Exploratory |
| Unweighted-loss candidate | Previously examined fold_2 | 0.040 | 0.641 | 0.921 | 0.796 | **Not deployed** |

Source: `outputs/classifier/improvements/validation_comparison.json`, `qualified_comparison.json`. The candidate lost blast sensitivity against the retained full model (0.965 to 0.921) and added 108 false negatives.

## 8. Hardware and time

| Item | Observed |
|---|---|
| Host | Windows 10 build 26200; CPU model not verified in saved reports |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU; 4 GiB dedicated VRAM |
| System RAM | 15.73 GiB |
| Python / environment | 3.11.17; executable in `leukemia-yolo` Conda environment |
| PyTorch / Torchvision / CUDA build | 2.14.1+cu126 / 0.29.1+cu126 / 12.6 |
| Ultralytics / FastAPI | 8.4.176 / 0.143.0 |
| YOLO main run | 272.76 s; 50 epochs |
| Classifier full fine-tune | 2,333.139 s measured epoch time; 20 epochs; peak allocated CUDA memory about 538 MiB |

Sources: `outputs/environment_report.json`, `outputs/audit/hardware_report.json`, detector `run_metadata.json`, full-run `training_history.csv`.

## 9. Predicted-cell quantification

| Type | Detected | Classified | Normal | Blast | Unclassified | Estimated fraction |
|---|---:|---:|---:|---:|---:|---:|
| **Illustrative arithmetic only** | 20 | 20 | 15 | 5 | 0 | 25% |
| **Saved BCCD integrated demonstration; no blast truth** | 4 | 2 | 0 | 2 | 2 | 100% |

Saved example: `outputs/inference/smear_20261013T151756Z_6c85e10b/report.json`; full-model checkpoint hash matches above. The 100% is a predicted fraction of accepted classified detections, **not** a validated blast fraction or disease severity measure.
