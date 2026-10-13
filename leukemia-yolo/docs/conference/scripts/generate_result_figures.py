"""Build figures only from saved evaluations and original qualitative outputs."""
from pathlib import Path
import json, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
FULL = REPO / "outputs/classifier/full_finetune/full_20261013T142418Z_d2b3a8ab"
YOLO = REPO / "outputs/paper_results/detection_metrics.json"
BASE = REPO / "outputs/classifier/paper_results/classification_metrics.json"

def copy(source, target):
    if not source.is_file(): raise FileNotFoundError(source)
    shutil.copyfile(source, OUT/target)

def main():
    base = json.loads(BASE.read_text())
    full = json.loads((FULL/"test_metrics.json").read_text())
    det = json.loads(YOLO.read_text())
    assert full["tn"]+full["fp"]+full["fn"]+full["tp"] == full["n"] == 3553
    for source, target in [
        (FULL/"test/confusion_matrix.png", "classifier_confusion_matrix.png"),
        (FULL/"test/roc_curve.png", "classifier_roc.png"),
        (FULL/"test/pr_curve.png", "classifier_pr.png"),
        (FULL/"review/training_curves.png", "classifier_training_curves.png"),
        (REPO/"outputs/classifier/gradcam/gradcam_publication_grid.png", "gradcam_examples.png"),
    ]: copy(source,target)
    metrics = [("Accuracy","accuracy"),("Balanced accuracy","balanced_accuracy"),("Blast sensitivity","blast_sensitivity"),("Normal specificity","normal_specificity"),("Macro F1","macro_f1"),("ROC AUC","roc_auc")]
    fig, ax=plt.subplots(figsize=(10,4.8)); x=range(len(metrics))
    ax.bar([v-.18 for v in x],[base[k] for _,k in metrics],width=.36,label="Baseline, threshold 0.50",color="#91a9c6")
    ax.bar([v+.18 for v in x],[full[k] for _,k in metrics],width=.36,label="Full fine-tune, threshold 0.05",color="#167c9a")
    ax.set_xticks(list(x),[n.replace(" ","\n") for n,_ in metrics]); ax.set_ylim(0,1.08)
    ax.set_ylabel("Proportion / score"); ax.set_title("C-NMC fold_2 • 3,553 cells; operating thresholds differ")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/"baseline_vs_finetuned.png",dpi=220); plt.close(fig)
    names=["Precision","Recall","mAP@0.5","mAP@0.5:0.95"]
    vals=[det["metrics"][k] for k in ("precision","recall","mAP50","mAP50_95")]
    fig,ax=plt.subplots(figsize=(7,4)); bars=ax.bar(names,vals,color="#27778c");ax.bar_label(bars,fmt="%.3f",padding=3)
    ax.set_ylim(0,1.12);ax.set_ylabel("Score");ax.set_title("BCCD WBC localization • test • 72 images")
    fig.tight_layout();fig.savefig(OUT/"yolo_evaluation_summary.png",dpi=220);plt.close(fig)
    predictions=sorted((REPO/"outputs/predictions").glob("predictions_*/???_*_prediction.png"))
    if predictions:
        fig, axes=plt.subplots(1,min(3,len(predictions)),figsize=(12,3.7))
        for ax,p in zip(axes if hasattr(axes,"__iter__") else [axes],predictions[:3]):
            with Image.open(p) as im: ax.imshow(im.convert("RGB"))
            ax.set_title("_".join(p.name.split("_")[1:3])); ax.axis("off")
        fig.suptitle("Actual YOLO predictions on BCCD images; qualitative only")
        fig.tight_layout();fig.savefig(OUT/"yolo_detection_examples.png",dpi=200);plt.close(fig)
    reports=sorted((REPO/"outputs/inference").glob("smear_*/report.json"))
    selected=None
    for p in reports:
        report=json.loads(p.read_text())
        if report.get("model_versions",{}).get("classifier_checkpoint_sha256")==full["checkpoint_sha256"]:
            selected=p;break
    if selected:
        report=json.loads(selected.read_text()); source=selected.parent/report["annotated_filename"]
        copy(source,"integrated_dashboard_example.png")
        # This is the persisted annotated pipeline output, not a browser screenshot.
    print(f"Generated result assets in {OUT}")

if __name__=="__main__": main()
