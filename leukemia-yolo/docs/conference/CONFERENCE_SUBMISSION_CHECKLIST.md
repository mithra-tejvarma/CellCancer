# Conference submission checklist

## Presently supported

- [x] Report BCCD **WBC localization** separately from C-NMC **segmented-cell classification**.
- [x] State class 1 is ALL blast and give split, cell and subject counts.
- [x] Report both model checkpoint hashes and the baseline/full threshold difference.
- [x] Include the full-model confusion counts and low normal specificity.
- [x] Label Grad-CAM four-category gallery as baseline checkpoint evidence.
- [x] Label BCCD integrated annotations and fractions as qualitative, without blast truth.
- [x] Correct the residual-attention paper's author attribution using its source record.

## Before submission

- [ ] Select a venue and verify its current template, page limit, figure policy and citation style.
- [ ] Confirm the current C-NMC recommended dataset citation and any local Kaggle mirror terms.
- [ ] Add verified authors/affiliations and appropriate data/governance statements; do not invent approvals.
- [ ] Obtain genuinely unseen subject-wise or external validation; disclose existing fold_2 exposure.
- [ ] Evaluate labeled leukemia whole smears before claiming end-to-end accuracy or blast-percentage validity.
- [ ] Seek expert assessment of failure cases and Grad-CAM interpretation.
- [ ] Check 55.3% normal specificity and threshold 0.05 against the intended use and calibration study.
- [ ] Match each figure's model, dataset, split and checkpoint to `FIGURE_MANIFEST.csv`.
- [ ] Run `python docs/conference/scripts/verify_conference_package.py` after any changes.

## Figure placement suggestion

Main paper: complete framework, concise dataset table, model schematic if space permits, detector metrics, classifier confusion matrix/ROC and one qualified attribution panel. Supplement or presentation: separate two-dataset diagram, full Grad-CAM gallery, inference/quantification workflows, training curves and integrated demonstration.
