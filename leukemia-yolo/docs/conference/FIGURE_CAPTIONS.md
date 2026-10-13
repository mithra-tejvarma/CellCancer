# Figure captions and provenance

All `diagrams/` figures are original implementation schematics rendered by `scripts/generate_architecture_diagrams.py`; their arrows show data or computation flow, not observed diagnostic performance. All `results/` figures are copied or derived from saved experiments by `scripts/generate_result_figures.py`. Repository-relative source paths are listed in `FIGURE_MANIFEST.csv`.

| Figure | Suggested IEEE-style caption | Evidence and limitation |
|---|---|---|
| 1 | Implemented YOLOv8n–EfficientNet-B3 WBC analysis framework with Grad-CAM and estimated count output. | Architecture schematic from `src/integrated_pipeline.py`; no end-to-end labeled-smear metric. |
| 2 | YOLOv8n single-class WBC detector showing C2f/SPPF backbone, multi-scale feature fusion, and anchor-free detection head. | Derived from `yolov8n.pt` run metadata and official Ultralytics architecture; simplified layer graph. |
| 3 | EfficientNet-B3 binary cell classifier and MBConv/SE components. | `src/classifier_model.py`, `src/classifier_transforms.py`; project input 224 × 224, not the 300-pixel pretrained recipe. |
| 4 | Grad-CAM attribution workflow for a trained EfficientNet-B3 prediction. | `src/classifier_gradcam.py`; visual attribution does not verify cell morphology. |
| 5 | Independent BCCD WBC-detection and C-NMC HEM/ALL-classification training paths. | `outputs/preprocessing_report.json`, C-NMC splits; datasets are not merged. |
| 6 | Integrated smear inference sequence from input validation to JSON/dashboard output. | `src/integrated_pipeline.py`; integration is demonstrated, not quantitatively validated on leukemia-labeled smears. |
| 7 | Predicted WBC count and accepted-cell blast-fraction calculation. | `src/integrated_pipeline.py`; excludes unclassified cells, returns null for zero denominator. |
| 8 | YOLOv8n WBC predictions on actual BCCD images. | Saved prediction images under `outputs/predictions/`; qualitative only, no blast labels. |
| 9 | BCCD test WBC localization precision, recall, mAP@0.5 and mAP@0.5:0.95. | `outputs/paper_results/detection_metrics.json`; 72 images; WBC boxes only. |
| 10 | C-NMC fold_2 full-model confusion matrix at validation-selected threshold 0.05. | Full-run `test/confusion_matrix.png` and `test_metrics.json`; 3,553 cells/31 subjects, previously examined fold. |
| 11 | Baseline and full-model C-NMC fold_2 metrics at respective stored thresholds 0.50 and 0.05. | Baseline and full test JSON; operating-point differences reflect thresholds and weights. |
| 12 | C-NMC fold_2 ROC curve for the full classifier. | Copied from full-run `test/roc_curve.png`, based on saved scores; 3,553 cells, previously examined fold. |
| 13 | C-NMC fold_2 precision–recall curve for the full classifier. | Copied from full-run `test/pr_curve.png`; threshold sweep, not a whole-smear curve. |
| 14 | Recorded full-model training and validation histories. | Full-run `review/training_curves.png` derived from `training_history.csv`; no invented epochs. |
| 15 | Correct-normal, correct-blast, false-positive and false-negative Grad-CAM examples from the **baseline** classifier. | `outputs/classifier/gradcam/gradcam_publication_grid.png` and `gradcam_summary.json`; C-NMC fold_2 examples, baseline threshold 0.50, qualitative attribution only. |
| 16 | Saved integrated BCCD annotation from the full-checkpoint demonstration. | `outputs/inference/smear_20261013T151756Z_6c85e10b/annotated.png` and `report.json`; 4 detected WBCs, 2 accepted as blasts, 2 unclassified; no blast ground truth. This is an annotated output, **not a dashboard screenshot**. |

**Unavailable requested visual:** A real browser screenshot of the dashboard is not saved. The expected `integrated_dashboard_example.png` contains the actual persisted *annotated output* and is explicitly captioned as such. A screenshot can be captured in a future browser session if the paper needs to show the user interface. A ground-truth end-to-end prediction panel is unavailable because the project has no leukemia-labeled whole-smear evaluation set.
