"""Build a status table from existing measured JSON artifacts only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def main():
    output = load_config()["output_path"]
    detector = json.loads((output / "paper_results/detection_metrics.json").read_text())
    baseline = json.loads((output / "classifier/paper_results/classification_metrics.json").read_text())
    full = sorted((output / "classifier/full_finetune").glob("full_*/completed.json"))
    demo = sorted((output / "inference").glob("smear_*/report.json"))
    lines = ["# Conference experiment status", "",
             "| Experiment | Dataset and unit | Measured result | Status |", "|---|---|---|---|",
             f"| YOLOv8n WBC localization | BCCD test, {detector['test_images']} images | precision {detector['metrics']['precision']:.3f}; recall {detector['metrics']['recall']:.3f}; mAP50 {detector['metrics']['mAP50']:.3f}; mAP50-95 {detector['metrics']['mAP50_95']:.3f} | EVALUATED |",
             f"| EfficientNet-B3 baseline | C-NMC fold_2, {baseline['n']} cells, {baseline['test_subjects']} subjects | accuracy {baseline['accuracy']:.3f}; balanced accuracy {baseline['balanced_accuracy']:.3f}; blast sensitivity {baseline['blast_sensitivity']:.3f}; normal specificity {baseline['normal_specificity']:.3f}; macro F1 {baseline['macro_f1']:.3f}; ROC-AUC {baseline['roc_auc']:.3f} | EVALUATED |"]
    if full:
        run = full[-1].parent
        metric = json.loads((run / "test_metrics.json").read_text())
        lines.append(f"| EfficientNet-B3 full backbone | C-NMC fold_2, {metric['n']} cells | accuracy {metric['accuracy']:.3f}; macro F1 {metric['macro_f1']:.3f}; threshold {metric['threshold']:.2f} | EVALUATED; prior test exposure |")
    else:
        lines.append("| EfficientNet-B3 full backbone | C-NMC | No completed experiment | PENDING |")
    lines += ["", "## Supporting results", "",
              f"- Validation-selected operating threshold: {'saved in completed full run' if full else 'pending full run'}.",
              f"- Grad-CAM qualitative overlays: {len(list((output / 'inference').glob('smear_*/*_gradcam.png')))} integrated demonstration files.",
              f"- Integrated BCCD functional demonstrations: {len(demo)}. These have no blast ground truth.",
              "- Hardware and preliminary timing: see `outputs/audit/hardware_report.json` and `training_benchmark.json`.",
              ""]
    if full and (run / "review/review_metrics.json").exists():
        review = json.loads((run / "review/review_metrics.json").read_text())
        config = json.loads((run / "training_config.json").read_text())
        delta = review["paired_full_minus_baseline_subject_bootstrap_95_intervals"]
        lines += ["## Full fine-tuning methods and results", "",
                  f"Initialization: saved baseline best weights (SHA256 `{config['init_sha256']}`); all feature blocks trainable, BatchNorm running statistics frozen.", "",
                  f"Train/validation/test: 4,667 cells / 30 subjects; 1,185 cells / 8 subjects; 3,553 cells / 31 subjects. Class 0 is Normal (HEM); class 1 is ALL blast.", "",
                  f"RGB, 224×224, ImageNet normalization; mild flip, rotation and brightness/contrast training augmentation. AdamW, learning rate {config['lr']}, weight decay {config['weight_decay']}, physical batch {config['batch_size']}, accumulation {config['accumulation']}; weighted cross-entropy; AMP; seed {config['seed']}.", "",
                  f"20 epochs completed in {review['total_epoch_seconds'] / 60:.1f} minutes of measured epoch time; best epoch {review['selected_epoch']}. Peak allocated CUDA memory {review['peak_cuda_allocated_bytes'] / 2**20:.0f} MiB.", "",
                  f"Validation macro F1 selected the checkpoint and the validation-selected threshold was {metric['threshold']:.2f}, the lower search boundary. Validation macro F1 on the reloaded checkpoint was {review['val_at_saved_threshold']['macro_f1']:.3f}; fold_2 macro F1 was {metric['macro_f1']:.3f}.", "",
                  "| Fold_2 classifier metric | Baseline | Full fine-tuning |", "|---|---:|---:|"]
        for name in ("accuracy", "balanced_accuracy", "blast_precision", "blast_sensitivity", "normal_specificity", "macro_f1", "roc_auc"):
            lines.append(f"| {name} | {baseline[name]:.3f} | {metric[name]:.3f} |")
        lines += ["", f"Full model confusion matrix: TN {metric['tn']}, FP {metric['fp']}, FN {metric['fn']}, TP {metric['tp']}.", "",
                  f"Paired subject-bootstrap 95% interval for macro F1 change: {delta['macro_f1']['lower_95']:+.3f} to {delta['macro_f1']['upper_95']:+.3f}; for specificity change: {delta['normal_specificity']['lower_95']:+.3f} to {delta['normal_specificity']['upper_95']:+.3f}. These are descriptive intervals from 31 test subjects and do not remove prior test-set exposure.", "",
                  "Figures: `review/training_curves.png`, `test/confusion_matrix.png`, `test/roc_curve.png`, and `test/pr_curve.png` within the full run. Full metadata and rechecked metrics: `review/review_metrics.json`.", ""]
    lines += [
              "", "## Data sources and limitations", "",
              "BCCD: https://github.com/Shenggan/BCCD_Dataset", "",
              "C-NMC 2019: https://www.cancerimagingarchive.net/collection/c-nmc-2019/ ; DOI: 10.7937/tcia.2019.dc64i46r", "",
              "The detector and classifier use different image distributions. There is no quantitatively labeled whole-smear ALL evaluation here. The fold_2 classifier test set was already examined in baseline work; a fresh subject-disjoint evaluation is needed for an unbiased estimate.", ""]
    path = output / "audit/conference_status.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
