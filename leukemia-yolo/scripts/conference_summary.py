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
              "", "## Data sources and limitations", "",
              "BCCD: https://github.com/Shenggan/BCCD_Dataset", "",
              "C-NMC 2019: https://www.cancerimagingarchive.net/collection/c-nmc-2019/ ; DOI: 10.7937/tcia.2019.dc64i46r", "",
              "The detector and classifier use different image distributions. There is no quantitatively labeled whole-smear ALL evaluation here. The fold_2 classifier test set was already examined in baseline work; a fresh subject-disjoint evaluation is needed for an unbiased estimate.", ""]
    path = output / "audit/conference_status.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
