<h1 align="center">Large-Scale Multi-Cancer Detection by Learning Segmentation from Reports</h1>

This is the code for the paper **Large-Scale Multi-Cancer Detection by Learning Segmentation from Reports**. It has an available preprint (see below).

Abstract:

More than 300 million computed tomography (CT) scans are performed worldwide each year, yet many early or incidental tumors in these scans remain undetected. Artificial intelligence (AI) could help: segmentation models can surpass radiologists and alternative AI models in detecting tumors, and they localize the tumors for radiologist verification. However, segmentation-based tumor detection has long been limited by the need for tumor masks: radiologist-drawn tumor outlines that are scarce, expensive, and entirely unavailable for many cancer types. In contrast, nearly every CT scan is accompanied by a radiology report with detailed tumor descriptions. Yet, these reports have not been used effectively to train tumor segmentation models. Here, we introduce R-Super, a framework that converts routine radiology and pathology reports into localized training signals for tumor segmentation. R-Super trains AI to segment tumors that match their descriptions in reports. Reports are only needed for training, not inference. We trained R-Super on 127,496 CT-Report pairs (42 million 2D images, USA) and evaluated it internally at UCSF (*N*=2,301, USA) and externally at Stanford (*N*=1,976, USA), Medipol (*N*=1,327, Turkey), and Basel (*N*=2,935, Switzerland). R-Super detects 7 tumor types for which public tumor masks are scarce or absent: spleen, gallbladder, prostate, bladder, uterus, esophagus, and adrenal tumors. Training R-Super on over 100,000 reports (no mask) outperformed mask-based segmentation models trained on 870 masks, demonstrating that large-scale report-based training surpasses smaller-scale mask-based training. Alternatively, by training R-Super on both these reports and masks together, cancer detection sensitivity increased by over +11% beyond mask-only training, and DSC by +14%. R-Super significantly surpassed 6 alternative training frameworks trained on the same dataset and 9 leading public AI models. R-Super significantly surpassed six radiologists in detecting six tumor types and matched them for uterus tumors in a reader study. On average, R-Super detected 56% more malignant tumors than the radiologists, at matched false positive rate. These results show that radiology reports are not merely clinical documentation, but a large-scale, underutilized source of localized supervision for training more accurate cancer detection AI. By effectively learning from reports, R-Super enables tumor segmentation models to scale beyond scarce radiologist-drawn tumor masks and advances automated and incidental cancer detection closer to clinical deployment. We release code, over 22,000 CT scans and reports, and the first public AI model to reach or surpass radiologist performance in detecting these tumor types on CT.

<p align="center">
  <img src="documents/rsuper_examples.png" width="100%"/>
</p>

<p align="center"><i>R-Super segments seven tumor types that have scarce or no public tumor masks. Top: input CT; bottom: R-Super tumor segmentation (orange).</i></p>

> [!NOTE]
> **The first public AI to reach or surpass radiologists on seven understudied cancers.** Trained on 100,000+ CT scans and radiology reports, R-Super segments spleen, gallbladder, prostate, bladder, uterus, esophagus, and adrenal tumors, detecting 56% more malignant tumors than radiologists at matched specificity. Read the preprint [here](https://arxiv.org/abs/2510.14803).

## Detailed Code Instructions

[**1- Use LLM to extract tumor information from reports**](report_extraction/README.md)

[**2- Create organ segmentation masks**](organ_masks/README.md)

[**3- Train (and test) R-Super for tumor segmentation using reports**](rsuper_train/README.md)



## Novel loss functions: reports supervise segmentation

<p align="center">
  <img src="documents/rsuper_method.png" width="80%"/>
</p>

<p align="center"><i>The R-Super training framework: an LLM extracts tumor information from radiology and pathology reports, and four loss functions teach the segmentation model to produce tumors that match the report descriptions.</i></p>

The **Volume Loss** enforces the volume of the segmented tumors to match the tumor volume estimated from the report. **Ball Loss** enforces each segmented tumor to match the number, rough location, and diameter described in the report. The **Attenuation Loss** enforces segmented tumors to match the attenuation (relative brightness) in reports. The **Malignancy Loss** teaches the model to distinguish malignant from benign tumors.


## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research. Paper content is covered by patents pending.

---

## Relationship to Previous Conference Paper

This paper builds on the methodology in our prior conference work "Learning Segmentation From Radiology Reports", winner of the *MICCAI 2025 Best Paper Award* (top 2 of 1,027 accepted papers). This paper presents several substantive advancements over the conference work:

1. **Scope:** In the conference work, we detected pancreatic and kidney tumors. We now detect seven different tumor types. Unlike pancreatic and kidney tumors, these tumor types have scarce or no publicly available tumor masks and cannot be accurately detected by public AI models. This paper presents the first public AI model shown to surpass radiologists in detecting these cancers on CT.
2. **Scale:** We increase the training dataset from 6,718 to 127,496 CT-Report pairs and add over 1,500 tumor masks created by 31 radiologists. This moves from a proof-of-concept dataset to hospital-scale training.
3. **Validation:** We add a reader study with six radiologists and expand external validation from one hospital with 100 CT scans to three hospitals in three countries, with 1,327 to 2,935 CT scans per test hospital.
4. **Role of reports:** Unlike the conference work, where reports only supplemented tumor masks, R-Super can now use reports to supplement or substitute masks, making reports a key to scaling tumor segmentation.
5. **Richer report supervision:** New and improved loss functions allow R-Super to learn from tumor slice and attenuation information in radiology reports, beyond the report features used in the conference work.
6. **Cancer identification:** R-Super now learns, from pathology reports, how to distinguish cancer from benign lesions, while the conference model detected lesions without distinguishing cancer.

Reference to the conference work: Bassi, P.R.A.S. et al. (2026). Learning Segmentation from Radiology Reports. In: Gee, J.C., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. MICCAI 2025. Lecture Notes in Computer Science, vol 15964. Springer, Cham. https://doi.org/10.1007/978-3-032-04971-1_29

Github: https://github.com/MrGiovanni/R-Super

> Awards: MICCAI 2025 Best Paper Award (runner-up, top 2 of 1,027 papers) · RSNA 2025 Certificate of Merit Award.

## Papers

**This repository accompanies our new preprint:**

<b>Large-Scale Multi-Cancer Detection by Learning Segmentation from Reports</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com/citations?user=NftgL6gAAAAJ&hl=en), Xinze Zhou, [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), Szymon Płotka, [Jieneng Chen](https://scholar.google.com/citations?user=yLYj88sAAAAJ&hl=zh-CN), ...,[Akshay S. Chaudhari](https://scholar.google.com/citations?user=08Y4NhMAAAAJ&hl=en), [Curtis P. Langlotz](https://scholar.google.com.vn/citations?user=WQkBYwQAAAAJ&hl=vi), [Ulas Bagci](https://scholar.google.com/citations?user=9LUdPM4AAAAJ&hl=en), [Sergio Decherchi](https://scholar.google.com/citations?user=T09qQ1IAAAAJ&hl=it), [Andrea Cavalli](https://scholar.google.com/citations?user=4xTOvaMAAAAJ&hl=en), Arkadiusz Sitek, [Kang Wang](https://radiology.ucsf.edu/people/kang-wang), [Yang Yang](https://scholar.google.com/citations?hl=en&user=6XsJUBIAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)* <br/>
*Johns Hopkins University, UCSF, Harvard Medical School / Massachusetts General Hospital, Stanford University, and collaborators* <br/>
Preprint <br/>
<a href='https://www.cs.jhu.edu/~zongwei/publication/bassi2025scaling.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='https://arxiv.org/abs/2510.14803'><img src='https://img.shields.io/badge/arXiv-2510.14803-b31b1b'></a>

#### Prior conference paper

<b>Learning Segmentation from Radiology Reports</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com/citations?user=NftgL6gAAAAJ&hl=en), [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), [Jieneng Chen](https://scholar.google.com/citations?user=yLYj88sAAAAJ&hl=zh-CN), Zheren Zhu, Tianyu Lin, [Sergio Decherchi](https://scholar.google.com/citations?user=T09qQ1IAAAAJ&hl=it), [Andrea Cavalli](https://scholar.google.com/citations?user=4xTOvaMAAAAJ&hl=en), [Kang Wang](https://radiology.ucsf.edu/people/kang-wang), [Yang Yang](https://scholar.google.com/citations?hl=en&user=6XsJUBIAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)* <br/>
*Johns Hopkins University* <br/>
MICCAI 2025, [**Best Paper Award**](https://miccai.org/index.php/about-miccai/awards/best-paper-award-and-young-scientist-award/) (runner-up, top 2 of 1,027 papers) <br/>
<a href='https://www.cs.jhu.edu/~zongwei/publication/bassi2025learning.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='https://www.cs.jhu.edu/~zongwei/poster/bassi2025miccai_rsuper.pdf'><img src='https://img.shields.io/badge/Poster-PDF-blue'></a>
<a href='https://www.cs.jhu.edu/news/for-ai-tumor-detection-a-picture-isnt-always-worth-a-thousand-words/'><img src='https://img.shields.io/badge/JHU-News-green'></a>
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/7pamG9DDSJw?si=-376z03g832UyTKB)
<a href='https://youtu.be/r11X39fH-yU?si=ZOBlHMo1CvN9aVzb'><img src='https://img.shields.io/badge/Oral-RSNA-orange'></a>

## Citation

If you use the code, data or methods in this repository, please cite:

Preprint of latest paper:
```
@article{bassi2025scaling,
  title={Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks},
  author={Bassi, Pedro RAS and Zhou, Xinze and Li, Wenxuan and P{\l}otka, Szymon and Chen, Jieneng and Chen, Qi and Zhu, Zheren and Prz{\k{a}}do, Jakub and Hamac{\i}, Ibrahim E and Er, Sezgin and others},
  journal={arXiv preprint arXiv:2510.14803},
  year={2025}
}
```

Prior conference paper:
```
@inproceedings{bassi2025learning,
  title={Learning Segmentation from Radiology Reports},
  author={Bassi, Pedro RAS and Li, Wenxuan and Chen, Jieneng and Zhu, Zheren and Lin, Tianyu and Decherchi, Sergio and Cavalli, Andrea and Wang, Kang and Yang, Yang and Yuille, Alan L and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={305--315},
  year={2025},
  organization={Springer}
}
```

## R-Super Demo with Public Data

This demo trains and evaluates R-Super (Report Supervision) with only **public data, making it fully-reproducible.** Click here: [**R-Super Merlin Demo**](rsuper_train/Merlin_demo.md)

*The data for reproducing our results on the segmentation of 7 tumor types will be released soon, as well as the model's trained weights.*

Datasets (check the demo for download instructions):
- **[PanTS](https://github.com/MrGiovanni/PanTS)**: 10K CT, 1.1K pancreatic lesion segmentation masks
- **[Merlin](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40)**: 25K CTs & reports, 2K pancreatic lesion CTs, no mask
- **[Merlin Plus](https://huggingface.co/datasets/AbdomenAtlas/MerlinPlus/)**: 25K organ segmentation masks for Merlin, 44 classes


> **Easy to Reproduce** R-Super usually needs 3 steps: LLM extracts tumor information from reports, creation of organ segmentation masks, and training R-Super. Our demo skips steps 1 and 2. You can download the LLM output, organ segmentation masks (Merlin Plus), and just train R-Super. You can also download the trained R-Super checkpoint and test.
