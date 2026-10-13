"""Summarize measured development diagnostics and clearly separate hypotheses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import load_config


def main():
    root = load_config()["output_path"] / "classifier/improvement_audit"
    audit = json.loads((root / "audit.json").read_text())
    study = json.loads((root / "validation_threshold_study.json").read_text())
    subjects = pd.read_csv(root / "validation_subjects.csv")
    compare = pd.read_csv(root / "validation_model_comparison.csv")
    normal = subjects[subjects.class_id == 0]
    total_fp = int(normal.fp.sum())
    worst = normal.sort_values("fp", ascending=False).iloc[0]
    full = compare[(compare.model == "full") & (compare.policy == "stored")].iloc[0]
    lines = ["# C-NMC specificity improvement audit", "",
             "This report uses the saved subject-disjoint development split for diagnostics and threshold selection. The old fold_2 test set is only used in a separately labeled fair-comparison report, never to choose settings.", "",
             "## Verified setup", "",
             f"- Split fingerprint: `{audit['split_fingerprint']}`; train 4,667 cells/30 subjects; validation 1,185 cells/8 subjects; test 3,553 cells/31 subjects.",
             "- Class 0 is Normal (HEM); class 1 is ALL blast. RGB input, 224×224, ImageNet normalization.",
             f"- Inverse-frequency train-only class weights: normal {audit['train_class_weights'][0]:.3f}, blast {audit['train_class_weights'][1]:.3f}; shuffled batches, no weighted sampler.",
             "- Full backbone trainable; BatchNorm running statistics frozen. Existing augmentation uses horizontal flip, ±10° rotation, and ±8% brightness/contrast.",
             "", "## Confirmed development observations", "",
             f"- Only {audit['validation_normal_subjects']} normal subjects are in validation. At stored threshold 0.05, {total_fp} normal validation cells are false positives; {worst.subject_id} contributes {int(worst.fp)} ({100*worst.fp/total_fp:.1f}%).",
             f"- Stored-threshold validation specificity {full.normal_specificity:.3f}, blast sensitivity {full.blast_sensitivity:.3f}, macro F1 {full.macro_f1:.3f}.",
             f"- Densely searching validation only, macro F1 peaks at threshold {study['macro_f1_best']['threshold']:.3f}. The earlier threshold 0.05 sat at the old search floor.",
             f"- Prespecified objective (maximize validation specificity with sensitivity ≥0.95) selects threshold {study['selected']['threshold']:.3f}: specificity {study['selected']['normal_specificity']:.3f}, sensitivity {study['selected']['blast_sensitivity']:.3f}. This is not a test-set result or deployment change.",
             "- Validation images are 450×450. The selected gallery shows intact segmented cells, but some contours are irregular or partly clipped; no annotation error has been confirmed.",
             "", "## Suspected causes requiring further data", "",
             "- The large specificity gap from validation to fold_2 may reflect subject or staining distribution differences; two normal validation subjects are too few to identify the cause reliably.",
             "- Foreground color summaries differ modestly between the two normal validation subjects, but background-heavy microscopy images and different cell appearances make a stain-specific explanation unproven.",
             "- The softmax scores cluster near extremes and are not calibrated clinical probabilities. Threshold movement alone may not generalize across new subjects.",
             "", "## Artifacts", "",
             "`validation_subjects.csv`, `training_subjects.csv`, `validation_errors.csv`, `validation_threshold_curve.csv`, `validation_threshold_analysis.png`, `validation_error_gallery.png`, `validation_gradcam_examples.csv`, and the fair-comparison CSV are saved here.", ""]
    (root / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "audit.md")


if __name__ == "__main__":
    main()
