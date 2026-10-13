"""Check manuscript assets, metric arithmetic, provenance and missing evidence."""
from pathlib import Path
import csv, hashlib, json, re, xml.etree.ElementTree as ET
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
FULL=Path("outputs/classifier/full_finetune/full_20261013T142418Z_d2b3a8ab")
DETECTOR=Path("outputs/checkpoints/main_20261013T090007Z_a25a7934")
INTEGRATED=Path("outputs/inference/smear_20261013T151756Z_6c85e10b")
docs=["CONFERENCE_PAPER_COMPLETE_GUIDE.md","CONFERENCE_PAPER_DRAFT.md","FIGURE_CAPTIONS.md","RESULTS_TABLES.md","REFERENCES.md","CONFERENCE_SUBMISSION_CHECKLIST.md"]
diagrams=["complete_system_architecture","yolov8_architecture","efficientnet_b3_architecture","gradcam_workflow","two_dataset_training","complete_inference_pipeline","quantification_workflow"]
result_rows=[
 ("classifier_confusion_matrix","C-NMC full classifier confusion matrix","C-NMC","EfficientNet-B3 full","fold_2",FULL/"test/confusion_matrix.png"),
 ("baseline_vs_finetuned","Baseline versus full classifier","C-NMC","EfficientNet-B3 baseline/full","fold_2",Path("outputs/classifier/paper_results/classification_metrics.json")),
 ("classifier_roc","Full classifier ROC","C-NMC","EfficientNet-B3 full","fold_2",FULL/"test/roc_curve.png"),
 ("classifier_pr","Full classifier precision-recall","C-NMC","EfficientNet-B3 full","fold_2",FULL/"test/pr_curve.png"),
 ("classifier_training_curves","Full classifier training curves","C-NMC","EfficientNet-B3 full","train/validation",FULL/"review/training_curves.png"),
 ("yolo_detection_examples","Actual WBC prediction examples","BCCD","YOLOv8n","qualitative test",Path("outputs/predictions/predictions_20261013T090811Z_69d98a1c")),
 ("yolo_evaluation_summary","WBC detector test metrics","BCCD","YOLOv8n","test",Path("outputs/paper_results/detection_metrics.json")),
 ("gradcam_examples","Baseline Grad-CAM four-category gallery","C-NMC","EfficientNet-B3 baseline","fold_2",Path("outputs/classifier/gradcam/gradcam_publication_grid.png")),
 ("integrated_dashboard_example","Integrated annotated output (not UI screenshot)","BCCD","YOLOv8n + EfficientNet-B3 full","qualitative only",INTEGRATED/"annotated.png"),
]

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def check():
    errors=[]; notes=[]; passed=[]
    def test(condition, label):
        (passed if condition else errors).append(label)
    for doc in docs: test((HERE/doc).is_file(),f"Markdown exists: {doc}")
    refs=(HERE/"REFERENCES.md").read_text(encoding="utf-8")
    numbers=[int(x) for x in re.findall(r"(?m)^\[(\d+)\] ",refs)]
    test(numbers==list(range(1,len(numbers)+1)) and len(numbers)==9,"References numbered [1] through [9] without gaps")
    captions=(HERE/"FIGURE_CAPTIONS.md").read_text(encoding="utf-8")
    test(all(f"| {n} |" in captions for n in range(1,17)),"Captions cover Figures 1 through 16")
    manifest=[]
    for i,name in enumerate(diagrams,1):
        stem=f"{i:02d}_{name}"
        for suffix in ("svg","png"):
            generated=Path("docs/conference/diagrams")/f"{stem}.{suffix}"
            source=Path("docs/conference/scripts/generate_architecture_diagrams.py")
            manifest.append([f"F{i:02d}-{suffix.upper()}",name,"architecture", "BCCD/C-NMC" if i in (1,5,6,7) else "N/A","YOLOv8n / EfficientNet-B3" if i in (1,5,6,7) else name,"schematic",source.as_posix(),generated.as_posix(),(REPO/generated).is_file(),"Original schematic; not experiment output"])
    for i,(stem,title,dataset,model,split,source) in enumerate(result_rows,8):
        generated=Path("docs/conference/results")/(stem+".png")
        manifest.append([f"F{i:02d}",title,"result",dataset,model,split,source.as_posix(),generated.as_posix(),(REPO/source).exists() and (REPO/generated).is_file(),"Qualitative only" if "qualitative" in split else "Saved experiment or plotted saved metrics"])
    manifest.append(["MISSING-UI","Actual browser screenshot","unavailable","BCCD","Integrated application","qualitative", "", "",False,"No persisted browser screenshot; annotated output exists as F16"])
    manifest.append(["MISSING-END2END","Labeled whole-smear leukemia result","unavailable","none","Integrated application","none", "", "",False,"No whole-smear leukemia ground truth or quantitative evaluation"])
    fields=["figure_id","title","type","dataset","model","split","source_path","generated_path","verified","notes"]
    with (HERE/"FIGURE_MANIFEST.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(fields);writer.writerows(manifest)
    test(all(row[8] for row in manifest[:-2]),"Every presented figure has an existing source and generated image")
    test(all(not row[8] and not row[7] for row in manifest[-2:]),"Missing results are declared without fake files")
    for doc in docs:
        content=(HERE/doc).read_text(encoding="utf-8")
        for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)",content):
            test((HERE/img).is_file(),f"Embedded image exists: {doc} -> {img}")
    for path in (HERE/"diagrams").glob("*.svg"):
        try: test(ET.parse(path).getroot().tag.endswith("svg"),f"Valid SVG: {path.name}")
        except ET.ParseError: errors.append(f"Invalid SVG: {path.name}")
    for path in list((HERE/"diagrams").glob("*.png"))+list((HERE/"results").glob("*.png")):
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                test(im.width>=500 and im.height>=250,f"Valid PNG dimensions: {path.name}")
        except Exception as exc: errors.append(f"Invalid PNG {path.name}: {exc}")
    full=json.loads((REPO/FULL/"test_metrics.json").read_text())
    base=json.loads((REPO/"outputs/classifier/paper_results/classification_metrics.json").read_text())
    det=json.loads((REPO/"outputs/paper_results/detection_metrics.json").read_text())
    pre=json.loads((REPO/"outputs/preprocessing_report.json").read_text())
    split=json.loads((REPO/"outputs/classifier/improvement_audit/audit.json").read_text())["counts"]
    conf=(full["tn"],full["fp"],full["fn"],full["tp"])
    test(conf==(606,490,85,2372) and sum(conf)==full["n"]==3553,"Full confusion counts total 3,553")
    tol=1e-10
    test(abs((conf[0]+conf[3])/full["n"]-full["accuracy"])<tol,"Accuracy recomputed from confusion counts")
    test(abs(conf[3]/(conf[3]+conf[2])-full["blast_sensitivity"])<tol,"Sensitivity recomputed from confusion counts")
    test(abs(conf[0]/(conf[0]+conf[1])-full["normal_specificity"])<tol,"Specificity recomputed from confusion counts")
    test(abs((full["blast_sensitivity"]+full["normal_specificity"])/2-full["balanced_accuracy"])<tol,"Balanced accuracy recomputed")
    test((base["threshold"],full["threshold"])==(.5,.05),"Stored baseline and full thresholds differ")
    test(pre["splits"]=={"train":205,"val":87,"test":72} and pre["total_wbc_boxes"]==372,"BCCD split and WBC box counts")
    test(det["test_images"]==72 and det["split"]=="test" and all(0<=v<=1 for v in det["metrics"].values()),"Detector test metric fields and split")
    test([split[k]["images"] for k in ("train","val","test")]==[4667,1185,3553] and split["test"]["subjects"]==31,"C-NMC partitions and subjects")
    test(full["split_fingerprint"]==base["split_fingerprint"] and full["fold_2_prior_exposure"] is True,"Classifier split identity and prior exposure")
    test(sha(REPO/FULL/"best_model.pth")==full["checkpoint_sha256"],"Full classifier checkpoint SHA-256")
    test(sha(REPO/DETECTOR/"weights/best.pt")==det["checkpoint_sha256"],"YOLO checkpoint SHA-256")
    integrated=json.loads((REPO/INTEGRATED/"report.json").read_text())
    test(integrated["model_versions"]["classifier_checkpoint_sha256"]==full["checkpoint_sha256"] and (integrated["total_detected_wbc"],integrated["total_classified_wbc"],integrated["predicted_blast"],integrated["unclassified"])==(4,2,2,2),"Integrated example identity and counts")
    notes.extend(["No leukemia-labeled whole-smear reference evaluation is available.","No actual browser screenshot is saved; F16 is a persisted annotated output.","F15 Grad-CAM gallery belongs to the baseline checkpoint, not the full fine-tune.","fold_2 had prior development exposure and is not fresh external validation.","CPU model and fully resolved YOLO optimizer/augmentation settings are not verified in saved records."])
    lines=["# Conference package verification", "",f"**Result: {'PASS' if not errors else 'FAIL'}** — {len(passed)} checks passed; {len(errors)} failed.","","## Checks",""]
    lines += [f"- PASS: {item}" for item in passed]
    if errors: lines += ["","## Failures",""]+[f"- FAIL: {item}" for item in errors]
    lines += ["","## Missing or qualified evidence",""]+[f"- {item}" for item in notes]
    (HERE/"verification_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"{'PASS' if not errors else 'FAIL'}: {len(passed)} passed; {len(errors)} failed")
    if errors: print("\n".join(errors))
    return not errors

if __name__=="__main__": raise SystemExit(0 if check() else 1)
