"""Render original, implementation-specific conference schematics (no model inference)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

FIGURES = [
    ("01_complete_system_architecture", "Implemented research framework", [
        ["Blood-smear RGB image", "YOLOv8n WBC detector", "Filtered WBC boxes", "RGB WBC crops"],
        ["EfficientNet-B3\nNormal / Blast / Unclassified", "Grad-CAM on accepted cells", "Counts + estimated blast fraction", "FastAPI JSON + dashboard"]]),
    ("02_yolov8_architecture", "YOLOv8n WBC localization", [
        ["RGB smear\n640 px input", "Backbone\nConv + C2f stages", "SPPF\nmultiscale context"],
        ["Neck\nFPN top-down + PAN bottom-up\nUpsample / Concat / C2f", "P3 / P4 / P5 features", "Anchor-free Detect head\nbox + WBC score", "NMS + WBC crop"]]),
    ("03_efficientnet_b3_architecture", "EfficientNet-B3 cell classifier", [
        ["RGB crop\n224 × 224", "Convolutional stem", "MBConv stages\ndepth / width scaled", "Final features +\nglobal average pooling", "Two-logit head\n0 HEM / 1 ALL"],
        ["MBConv detail", "Expansion 1×1", "Depthwise convolution", "Squeeze-and-Excitation", "Projection 1×1\noptional residual"]]),
    ("04_gradcam_workflow", "Post-hoc Grad-CAM attribution", [
        ["Accepted cell crop", "Trained EfficientNet-B3", "Target class logit", "Last spatial feature maps A"],
        ["Gradients ∂y/∂A", "Spatial mean → weights α", "ReLU of weighted maps", "Resize heatmap", "Overlay on crop"]]),
    ("05_two_dataset_training", "Independent training data flows", [
        ["BCCD RGB smears\nWBC boxes", "Preserved train / val / test", "YOLOv8n detector\nWBC localization"],
        ["C-NMC segmented cells\nHEM / ALL", "Subject-disjoint train / val / fold_2", "EfficientNet-B3 classifier\nNormal / Blast"]]),
    ("06_complete_inference_pipeline", "Integrated inference sequence", [
        ["Validate RGB upload", "YOLO NMS + duplicate filter", "Clip boxes + save crops", "Resize 224 + normalize"],
        ["Two logits → blast score", "Threshold 0.05 + uncertainty", "Grad-CAM if classified", "Aggregate + JSON / dashboard"]]),
    ("07_quantification_workflow", "Predicted-cell quantification", [
        ["Detected WBC boxes", "Normal / Blast / Unclassified", "Valid count = Normal + Blast"],
        ["Count predicted blasts", "100 × Blast / Valid", "Null if Valid = 0", "JSON + dashboard estimate"]]),
]

def draw(name, title, rows):
    width = 15
    fig, ax = plt.subplots(figsize=(width, 5.5 if len(rows) > 1 else 3.5))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .96, title, ha="center", va="top", fontsize=17, weight="bold", color="#13253e")
    ys = [.67, .27] if len(rows) == 2 else [.5]
    colors = ["#dceafa", "#e2f3ec"]
    for ridx, (row, y) in enumerate(zip(rows, ys)):
        n = len(row); gap = .018; left = .035; usable = .93; boxw = (usable - (n-1)*gap)/n
        for i, label in enumerate(row):
            x = left+i*(boxw+gap)
            ax.add_patch(FancyBboxPatch((x,y-.105),boxw,.21,boxstyle="round,pad=0.008,rounding_size=.015",linewidth=1.4,edgecolor="#547398",facecolor=colors[ridx]))
            ax.text(x+boxw/2,y,label,ha="center",va="center",fontsize=10 if n < 5 else 9,color="#172b45")
            if i < n-1:
                ax.add_patch(FancyArrowPatch((x+boxw+.002,y),(x+boxw+gap-.002,y),arrowstyle="-|>",mutation_scale=13,color="#315b82",lw=1.4))
    if len(rows)==2 and name!="05_two_dataset_training":
        # Route the flow around the right margin, then back to the start of row 2.
        # The datasets diagram deliberately has no connector between training tasks.
        ax.plot([.97,.985,.985,.018,.018],[.67,.67,.465,.465,.27],color="#315b82",lw=1.5)
        ax.add_patch(FancyArrowPatch((.018,.27),(.035,.27),arrowstyle="-|>",mutation_scale=14,color="#315b82",lw=1.5))
    if name == "05_two_dataset_training":
        ax.text(.5,.46,"Separate supervised tasks; checkpoints meet only at inference",ha="center",va="center",fontsize=11,style="italic",color="#6b5262")
    fig.savefig(OUT/f"{name}.svg",bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{name}.png",dpi=230,bbox_inches="tight",facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    for entry in FIGURES: draw(*entry)
    print(f"Rendered {len(FIGURES)} SVG/PNG pairs to {OUT}")
