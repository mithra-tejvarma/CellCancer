# Verified references

Numbering is shared with the guide and paper draft. Bibliographic facts were checked against publisher, proceedings, dataset-author, or official software pages on 13 October 2026. The proposed paper is independent of these studies; their results are not this project's results.

[1] J. C. S. Nunes *et al.*, “Cancer Cell Classification From Peripheral Blood Smear Data Using the YOLOv8 Architecture,” *IEEE Access*, vol. 13, pp. 91911–91924, 2025, doi: [10.1109/ACCESS.2025.3573277](https://doi.org/10.1109/ACCESS.2025.3573277). The paper studies YOLOv8 classification; it is not evidence for the present two-stage pipeline.

[2] K. Dasaramoole Prakash, J. Khan, and K. Kim, “Lightweight and Efficient YOLOv8 With Residual Attention Mechanism for Precise Leukemia Detection and Classification,” *IEEE Access*, vol. 12, pp. 159395–159413, 2024, doi: [10.1109/ACCESS.2024.3484933](https://doi.org/10.1109/ACCESS.2024.3484933). The suggested attribution to “Ullah et al.” was incorrect and has been corrected using the [authors’ institutional record](https://cnu.elsevierpure.com/en/publications/lightweight-and-efficient-yolov8-with-residual-attention-mechanis/).

[3] R. R. Selvaraju *et al.*, “Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization,” *Proceedings of the IEEE International Conference on Computer Vision*, pp. 618–626, 2017. [CVF proceedings record](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).

[4] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *Proceedings of the 36th International Conference on Machine Learning*, 2019. [Author publication page](https://research.google/pubs/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks/).

[5] S. Mourya, S. Kant, P. Kumar, A. Gupta, and R. Gupta, “ALL Challenge dataset of ISBI 2019 (C-NMC 2019),” *The Cancer Imaging Archive*, 2019, doi: [10.7937/TCIA.2019.DC64I46R](https://doi.org/10.7937/tcia.2019.dc64i46r). [Collection citation record](https://wiki.cancerimagingarchive.net/pages/viewpage.action?navigatingVersions=true&pageId=52762893). The citation should follow the collection's current recommended wording at submission.

[6] R. Gupta, S. Gehlot, and A. Gupta, “C-NMC: B-lineage acute lymphoblastic leukaemia: A blood cancer dataset,” *Medical Engineering & Physics*, vol. 103, 2022, doi: [10.1016/j.medengphy.2022.103793](https://doi.org/10.1016/j.medengphy.2022.103793). Listed by the [dataset authors](https://www.cancerimagingarchive.net/collection/c-nmc-2019/).

[7] BCCD Dataset, [original source repository](https://github.com/Shenggan/BCCD_Dataset). This repository provides the original BCCD context. The local experiment used a Kaggle JSON-formatted archive and preserved its train/validation/test layout; see `outputs/preprocessing_report.json`.

[8] Ultralytics, [YOLO architecture guide](https://docs.ultralytics.com/guides/yolo-architecture) and [YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/). Software documentation for C2f, SPPF, FPN/PAN, and the anchor-free detection head; actual checkpoint configuration remains authoritative.

[9] PyTorch, [Torchvision EfficientNet-B3 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b3.html). Its pretrained evaluation recipe uses a 300-pixel crop; this project explicitly trained/evaluated at 224 × 224.

