# Explainable Deep Learning Framework for Automated Leukemia Blast Cell Detection in Blood Smears

**Authors/affiliations:** [Insert verified author names and institutions after approval.]  
**Venue/template:** [Insert selected conference and required format.]

## Abstract

Automated cell review could support peripheral blood-smear screening, but localization and segmented-cell classification are often evaluated on different data. We implemented a research pipeline coupling a YOLOv8n white blood cell (WBC) detector with an EfficientNet-B3 normal-versus-blast classifier, post-hoc Grad-CAM overlays, counting, and a FastAPI interface. YOLOv8n was trained on BCCD and reached precision 0.9450, recall 1.0000, mAP@0.5 0.9646, and mAP@0.5:0.95 0.7707 on 72 test images; these measure WBC localization. On 3,553 segmented C-NMC fold_2 cells from 31 subjects, the fully fine-tuned classifier reached accuracy 0.8382, blast sensitivity 0.9654, normal specificity 0.5529, macro-F1 0.7851, and ROC-AUC 0.8434. Its previous baseline reached accuracy 0.7748 and ROC-AUC 0.7205. The models used different stored thresholds (0.50 baseline; 0.05 full), so operating-point gains cannot be assigned solely to weight updates. The integrated software yields cell-linked explanations and a predicted blast fraction, but BCCD lacks blast labels and no ground-truth whole-smear leukemia evaluation is available. Low normal specificity and prior exposure of the internal test fold limit generalization claims.

**Keywords—** leukemia screening, white blood cell detection, YOLOv8n, EfficientNet-B3, Grad-CAM.

## I. Introduction

Microscopic assessment of blood cells is part of leukemia investigation, alongside additional clinical and laboratory evidence. Expert cell review is labor intensive. A computational assistant can propose WBC locations and cell-level scores for review, while keeping links to original imagery. Classification of isolated cells does not establish performance on whole smears. We therefore connect localization, classification, visual attribution and count aggregation in a research prototype. The aim is to measure its components accurately and identify what remains unvalidated, not to make a standalone ALL diagnosis.

## II. Related Work

Nunes *et al.* explored YOLOv8 **classification** of peripheral-smear cancer cells [1]. Dasaramoole Prakash *et al.* studied an altered YOLOv8 architecture with residual attention [2]. Our detector is a single-class WBC **localizer** trained on BCCD, followed by a distinct EfficientNet-B3 cell classifier. EfficientNet uses compound model scaling [4]; Grad-CAM provides gradient-based visual attribution for trained CNN outputs [3]. These studies motivate components but do not validate the combined system or permit direct numerical comparison across datasets.

## III. Proposed Methodology

![System architecture](diagrams/01_complete_system_architecture.png)

**Fig. 1.** The implemented sequence connects YOLOv8n WBC boxes, RGB crops, EfficientNet-B3 HEM/ALL scores, Grad-CAM, counts and a local dashboard.

The BCCD source's RBC, WBC and platelet rectangles were converted to WBC-only YOLO labels. Its original 205/87/72 image train/validation/test allocation was retained. The detector uses `yolov8n.pt` initialization, 640-pixel inputs, C2f and SPPF feature extraction, a multi-scale neck and anchor-free box/class head [8]. NMS and a secondary overlap filter precede crop extraction. The separately trained C-NMC classifier receives 224 × 224 bicubic RGB crops normalized with ImageNet mean/std. Torchvision EfficientNet-B3's final layer was replaced with two logits [9]: 0=HEM and 1=ALL. All feature blocks were trainable in the reported full run; BatchNorm running statistics were frozen. A softmax blast score at or above validation-selected threshold 0.05 yields the positive class unless a crop is too small or the score is within the software's uncertainty margin. For a target class $c$, Grad-CAM weights feature channel $k$ by $\alpha_k^c=Z^{-1}\sum_{ij}\partial y^c/\partial A_{ij}^k$ and forms $L^c=\mathrm{ReLU}(\sum_k\alpha_k^c A^k)$ [3]. It does not certify cellular morphology.

Accepted predictions are counted with estimated blast fraction $100N_{blast}/(N_{normal}+N_{blast})$, returning null for a zero denominator. Unclassified detections are excluded. The software exports a JSON report and annotated image. BCCD and C-NMC supervised data are not merged.

## IV. Experimental Setup

| Dataset | Role | Train | Validation | Test |
|---|---|---:|---:|---:|
| BCCD | WBC box detection | 205 images | 87 images | 72 images |
| C-NMC [5], [6] | Segmented cell HEM/ALL classification | 4,667 cells / 30 subjects | 1,185 / 8 | 3,553 / 31 |

The classifier baseline best checkpoint initialized the full run; original model initialization was ImageNet pretrained [9]. Full fine-tuning used AdamW, weighted cross-entropy, learning rate $3\times10^{-5}$, weight decay $10^{-4}$, batch 4 and gradient accumulation 2 for 20 epochs. Validation macro-F1 selected epoch 16 and threshold 0.05. The BCCD detector completed 50 epochs with actual batch 16. The host had an NVIDIA RTX 3050 Ti Laptop GPU with 4 GiB dedicated VRAM, Python 3.11.17, PyTorch 2.14.1+cu126 and Ultralytics 8.4.176. The C-NMC fold_2 test was subject-disjoint but had been viewed during baseline development.

## V. Results and Discussion

| Test outcome | Baseline classifier | Full classifier |
|---|---:|---:|
| Accuracy | 0.7748 | 0.8382 |
| Blast sensitivity | 0.9292 | 0.9654 |
| Normal specificity | 0.4288 | 0.5529 |
| Macro-F1 | 0.6956 | 0.7851 |
| ROC-AUC | 0.7205 | 0.8434 |

The baseline used threshold 0.50 and the full model used 0.05. Full-model confusion counts were TN=606, FP=490, FN=85, TP=2,372; hence 490 of 1,096 normal cells were flagged as blast. ROC-AUC comparison is threshold independent, while the other rows depend on the operating point. YOLOv8n achieved BCCD-test WBC precision 0.9450, recall 1.0000, mAP@0.5 0.9646 and mAP@0.5:0.95 0.7707. These are separate tasks.

![Full model fold_2 confusion matrix](results/classifier_confusion_matrix.png)

**Fig. 2.** C-NMC fold_2 confusion matrix for the full EfficientNet-B3 model at threshold 0.05; positive class is ALL blast.

Three exploratory continuation arms were selected by validation specificity subject to at least 0.95 blast sensitivity. The best validation unweighted-loss arm obtained fold_2 specificity 0.6414, but sensitivity fell to 0.9214 with 108 additional missed blasts relative to the full model; it was not deployed. Only two normal subjects were in validation. A four-category Grad-CAM gallery exists for the **baseline** model, and an integrated full-model BCCD example is saved; neither validates the interpretation or accuracy of a whole-smear leukemia diagnosis. The BCCD detector and C-NMC classifier encounter different image distributions; integrated leukemia-labeled whole-smear ground truth is missing. Scores and estimated fractions are uncalibrated.

## VI. Conclusion

The prototype implements WBC localization, segmented-cell binary classification, Grad-CAM visual attribution, count-based estimates and a working local interface. On their respective test sets the detector reached mAP@0.5 0.9646 for WBC boxes and the full classifier reached ROC-AUC 0.8434 for cell labels. Normal specificity of 0.5529 and missing whole-smear validation preclude clinical reliability claims. Future work requires fresh subject-wise/external evaluation and annotated whole-smear testing of both the end-to-end labels and estimated blast fraction.

## References

[1] J. C. S. Nunes *et al.*, “Cancer Cell Classification From Peripheral Blood Smear Data Using the YOLOv8 Architecture,” *IEEE Access*, vol. 13, pp. 91911–91924, 2025, doi: [10.1109/ACCESS.2025.3573277](https://doi.org/10.1109/ACCESS.2025.3573277).

[2] K. Dasaramoole Prakash, J. Khan, and K. Kim, “Lightweight and Efficient YOLOv8 With Residual Attention Mechanism for Precise Leukemia Detection and Classification,” *IEEE Access*, vol. 12, pp. 159395–159413, 2024, doi: [10.1109/ACCESS.2024.3484933](https://doi.org/10.1109/ACCESS.2024.3484933).

[3] R. R. Selvaraju *et al.*, “Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization,” *ICCV*, pp. 618–626, 2017.

[4] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *ICML*, 2019.

[5] S. Mourya *et al.*, “ALL Challenge dataset of ISBI 2019 (C-NMC 2019),” *The Cancer Imaging Archive*, 2019, doi: [10.7937/TCIA.2019.DC64I46R](https://doi.org/10.7937/tcia.2019.dc64i46r).

[6] R. Gupta, S. Gehlot, and A. Gupta, “C-NMC: B-lineage acute lymphoblastic leukaemia: A blood cancer dataset,” *Medical Engineering & Physics*, vol. 103, 2022, doi: [10.1016/j.medengphy.2022.103793](https://doi.org/10.1016/j.medengphy.2022.103793).

[7] BCCD Dataset, [source repository](https://github.com/Shenggan/BCCD_Dataset).

[8] Ultralytics, [YOLO architecture guide](https://docs.ultralytics.com/guides/yolo-architecture).

[9] PyTorch, [Torchvision EfficientNet-B3 model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b3.html).

