"""Create six clean, executable research notebooks without altering old ones."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"


def make(filename, title, purpose, cells):
    path = DEST / filename
    if path.exists():
        raise FileExistsError(f"Preserving existing notebook: {path}")
    book = nbf.v4.new_notebook()
    book.metadata.kernelspec = {"display_name": "Python (leukemia-yolo)", "language": "python", "name": "leukemia-yolo"}
    book.metadata.language_info = {"name": "python"}
    book.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{purpose}\n\nRun from the existing `leukemia-yolo` Conda kernel. Source datasets and earlier results are read only."),
                  nbf.v4.new_code_cell("from pathlib import Path\nimport sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = Path.cwd().resolve()\nif ROOT.name != 'leukemia-yolo': ROOT = ROOT.parent\nassert (ROOT / 'config.local.yaml').is_file()\nprint(sys.executable, ROOT)")]
    for kind, content in cells:
        book.cells.append(nbf.v4.new_markdown_cell(content) if kind == "md" else nbf.v4.new_code_cell(content))
    nbf.write(book, path)
    print(path)


def main():
    make("21_Normal_Cell_Error_Analysis.ipynb", "Normal-cell error analysis",
         "Development-only subject error audit; fold_2 is never used to choose changes.", [
        ("md", "## Source and validation checks\n\nUse saved subject-disjoint manifests and regenerated predictions from the verified full checkpoint."),
        ("code", "import json\np = ROOT/'outputs/classifier/improvement_audit'\naudit = json.loads((p/'audit.json').read_text())\nprint(audit['split_fingerprint'], audit['validation_normal_subjects'])\nassert audit['validation_normal_subjects'] == 2"),
        ("code", "subjects = pd.read_csv(p/'validation_subjects.csv')\ndisplay(subjects[['subject_id','cells','fp','fn','normal_specificity','blast_sensitivity']])\nnormal = subjects[subjects.class_id==0]\nnormal.plot.bar(x='subject_id',y='fp',legend=False,title='False positives by normal validation subject'); plt.show()"),
        ("code", "from IPython.display import Image as DisplayImage, display\nfor image in sorted(p.glob('gradcam_false_positive_*.png')): display(DisplayImage(filename=str(image)))\nprint('Grad-CAM is visual attribution, not a verified morphological diagnosis.')"),
    ])
    make("22_Threshold_Analysis.ipynb", "Validation threshold study",
         "The objective is maximum validation specificity subject to blast sensitivity at least 0.95. Scores are uncalibrated.", [
        ("md", "## Validation-only threshold candidates\n\nNo threshold is chosen from fold_2."),
        ("code", "import json\np = ROOT/'outputs/classifier/improvement_audit'\nstudy = json.loads((p/'validation_threshold_study.json').read_text())\nprint(study['objective']); display(pd.DataFrame([study['selected'], study['macro_f1_best']]))"),
        ("code", "curve = pd.read_csv(p/'validation_threshold_curve.csv')\nax=curve.plot(x='threshold',y=['blast_sensitivity','normal_specificity','macro_f1'],figsize=(9,4)); ax.axhline(.95,color='gray',ls=':'); plt.show()"),
        ("code", "comparison=pd.read_csv(p/'validation_model_comparison.csv')\ndisplay(comparison[['model','policy','threshold','blast_sensitivity','normal_specificity','macro_f1','roc_auc']])\nprint('ROC-AUC does not depend on a decision threshold.')"),
    ])
    for number, arm, title in ((23, "augmentation_trial", "Mild augmentation experiment"),
                               (24, "class_weight_trial", "Class-weight experiment")):
        make(f"{number}_{'Augmentation_Experiment' if number==23 else 'Class_Weight_Experiment'}.ipynb", title,
             "Controlled continuation from the verified full checkpoint. Compare with the unchanged-policy control initialized identically.", [
            ("md", "## Launch in a Windows-safe subprocess\n\nSet `RUN_TRAINING=True` only when ready for a new run. The process streams its progress and writes a unique output directory."),
            ("code", f"import subprocess\nRUN_TRAINING=False\ncommand=[sys.executable,'-u',str(ROOT/'scripts/run_improvement_trial.py'),'--arm','{arm}','--epochs','8','--patience','3','--batch-size','4','--accumulation','2','--workers','2','--lr','0.00001']\nprint(' '.join(command))\nif RUN_TRAINING: subprocess.run(command,cwd=ROOT,check=True)"),
            ("code", f"base=ROOT/'outputs/classifier/improvements/{arm}'\nruns=sorted(base.glob('trial_*/completed.json'))\nif runs:\n    run=runs[-1].parent\n    print(run)\n    history=pd.read_csv(run/'training_history.csv')\n    display(history[['epoch','val_specificity','val_sensitivity','val_macro_f1','val_roc_auc']])\n    history.plot(x='epoch',y=['val_specificity','val_sensitivity','val_macro_f1']);plt.show()\nelse: print('Trial pending')"),
            ("code", f"from IPython.display import Image as DisplayImage, display\npreview=ROOT/'outputs/classifier/improvements/{arm}'\nimages=sorted(preview.glob('trial_*/augmentation_preview.png'))\nif images: display(DisplayImage(filename=str(images[-1])))\nprint('No fold_2 result is used for this trial.')"),
        ])
    make("25_Subject_Wise_CrossValidation.ipynb", "Subject-wise cross-validation",
         "Optional internal cross-validation. Each fold must initialize independently from ImageNet weights.", [
        ("md", "## Verified plan\n\nThe plan excludes fold_2 and checks subject, path, and image-hash separation."),
        ("code", "import subprocess, json\ncommand=[sys.executable,str(ROOT/'scripts/subject_cv.py'),'--folds','5']\nsubprocess.run(command,cwd=ROOT,check=True)\nplans=sorted((ROOT/'outputs/classifier/improvements/cross_validation').glob('cv_*/cv_plan.json'))\nplan=json.loads(plans[-1].read_text());display(pd.DataFrame(plan['folds'])[['fold','train_cells','val_cells','val_normal_cells','val_blast_cells']])"),
        ("code", "RUN_CV=False\nif RUN_CV: subprocess.run([sys.executable,'-u',str(ROOT/'scripts/subject_cv.py'),'--run','--folds','5','--epochs','8'],cwd=ROOT,check=True)\nelse: print('Five independent fold runs are optional and have not started.')"),
        ("code", "summaries=sorted((ROOT/'outputs/classifier/improvements/cross_validation').glob('cv_*/cv_summary.json'))\nif summaries:\n    result=json.loads(summaries[-1].read_text());display(pd.DataFrame(result['folds']))\nelse: print('No completed cross-validation metrics to report.')"),
    ])
    make("26_Improvement_Results.ipynb", "Improvement results and model decision",
         "Compare development trials at the prespecified objective; do not choose a deployment checkpoint from fold_2.", [
        ("code", "import json\nbase=ROOT/'outputs/classifier/improvements'\nrows=[]\nfor arm in ['control','augmentation_trial','class_weight_trial']:\n    runs=sorted((base/arm).glob('trial_*/completed.json'))\n    if runs:\n        run=runs[-1].parent\n        t=json.loads((run/'threshold_config.json').read_text())\n        h=pd.read_csv(run/'training_history.csv')\n        best=h[h.epoch==t['epoch']].iloc[0]\n        rows.append({'arm':arm,'epoch':t['epoch'],'threshold':t['threshold'],'eligible':t['eligible'],'specificity':best.val_specificity,'sensitivity':best.val_sensitivity,'macro_f1':best.val_macro_f1,'roc_auc':best.val_roc_auc})\ndisplay(pd.DataFrame(rows))"),
        ("code", "if rows:\n    data=pd.DataFrame(rows)\n    data.plot.bar(x='arm',y=['specificity','sensitivity','macro_f1'],ylim=(0,1),figsize=(9,4));plt.show()\nprint('Current full model remains the dashboard default until an independently justified replacement is approved.')"),
        ("code", "paper=ROOT/'outputs/audit/conference_status.md'\nprint(paper.read_text(encoding='utf-8')[-1800:])"),
    ])


if __name__ == "__main__":
    main()
