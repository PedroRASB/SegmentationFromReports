<h1 align="center">R-Super: Large-Scale Multi-Cancer Detection by Learning Segmentation from Reports</h1>

<div align="center">


![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/R-Super&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub stars](https://img.shields.io/github/stars/MrGiovanni/R-Super.svg?style=social)](https://github.com/MrGiovanni/R-Super/stargazers)
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>
**Subscribe us: https://groups.google.com/u/2/g/bodymaps**  

</div>


<p align="center">
  <img src="documents/r_super_pdac-8.gif" width="600"/>
</p> 



More than 300 million computed tomography (CT) scans are performed worldwide each year, yet many early or incidental tumors in these scans remain undetected. Artificial intelligence (AI) could help: segmentation models can surpass radiologists and alternative AI models in detecting tumors, and they localize the tumors for radiologist verification. However, segmentation-based tumor detection has long been limited by the need for tumor masks: radiologist-drawn tumor outlines that are scarce, expensive, and entirely unavailable for many cancer types. In contrast, nearly every CT scan is accompanied by a radiology report with detailed tumor descriptions. Yet, these reports have not been used effectively to train tumor segmentation models. Here, we introduce R-Super, a framework that converts routine radiology and pathology reports into localized training signals for tumor segmentation. R-Super trains AI to segment tumors that match their descriptions in reports. Reports are only needed for training, not inference. We trained R-Super on 127,496 CT-Report pairs (42 million 2D images, USA) and evaluated it internally at UCSF (*N*=2,301, USA) and externally at Stanford (*N*=1,976, USA), Medipol (*N*=1,327, Turkey), and Basel (*N*=2,935, Switzerland). R-Super detects 7 tumor types for which public tumor masks are scarce or absent: spleen, gallbladder, prostate, bladder, uterus, esophagus, and adrenal tumors. Training R-Super on over 100,000 reports (no mask) outperformed mask-based segmentation models trained on 870 masks, demonstrating that large-scale report-based training surpasses smaller-scale mask-based training. Alternatively, by training R-Super on both these reports and masks together, cancer detection sensitivity increased by over +11% beyond mask-only training, and DSC by +14%. R-Super significantly surpassed 6 alternative training frameworks trained on the same dataset and 9 leading public AI models. R-Super significantly surpassed six radiologists in detecting six tumor types and matched them for uterus tumors in a reader study. On average, R-Super detected 56% more malignant tumors than the radiologists, at matched false positive rate. These results show that radiology reports are not merely clinical documentation, but a large-scale, underutilized source of localized supervision for training more accurate cancer detection AI. By effectively learning from reports, R-Super enables tumor segmentation models to scale beyond scarce radiologist-drawn tumor masks and advances automated and incidental cancer detection closer to clinical deployment. We release code, over 22,000 CT scans and reports, and the first public AI model to reach or surpass radiologist performance in detecting these tumor types on CT. 


<p align="center">
  <img src="documents/rsuper_abstract.png" width="600"/>
</p> 

> [!NOTE]
> **The first public AI to reach or surpass radiologists on seven understudied cancers.** Trained on 100,000+ radiology reports — no tumor masks needed for these organs — R-Super segments spleen, gallbladder, prostate, bladder, uterus, esophagus, and adrenal tumors, detecting **56% more malignant tumors than radiologists at matched specificity**. Read the paper [here](https://arxiv.org/abs/2510.14803).

## R-Super Demo with Public Data


> [!NOTE]
> **We released Merlin Plus!**  
> The Merlin Plus dataset has 44 segmentation masks for each of the 25K CT scans in Merlin. The masks include organs, blood vessels, ducts, and organ sub-segments. Merlin Plus is a large public dataset with CT, reports, and organ masks. You can use it in the demo below to train R-Super and improve tumor segmentation.
>[Download it here!](https://huggingface.co/datasets/AbdomenAtlas/MerlinPlus/)
>


![Pancreatic metrics](documents/demo_results.png)


This demo trains and evaluates R-Super (Report Supervision) with only **public data, making it fully-reproducible.** Click here: [**R-Super Merlin Demo**](rsuper_train/Merlin_demo.md)

Datasets (check the demo for download instructions):
- **[PanTS](https://github.com/MrGiovanni/PanTS)**: 10K CT, 1.1K pancreatic lesion segmentation masks
- **[Merlin](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40)**: 25K CTs & reports, 2K pancreatic lesion CTs, no mask
- **[Merlin Plus](https://huggingface.co/datasets/AbdomenAtlas/MerlinPlus/)**: 25K organ segmentation masks for Merlin, 44 classes


> **Easy to Reproduce** R-Super usually needs 3 steps: LLM extracts tumor information from reports, creation of organ segmentation masks, and training R-Super. Our demo skips steps 1 and 2. You can download the LLM output, organ segmentation masks (Merlin Plus), and just train R-Super. You can also download the trained R-Super checkpoint and test.

## Public Trained Checkpoints

For inference instructions, check the evaluation section [here](rsuper_train/README.md) (detailed) or in our [demo](rsuper_train/Merlin_demo.md) (simplified).

| Model | Training Data | Tasks | Evaluation | Access |
|-------|---------------|-------|------------|--------|
| **R-Super (Paper)** | *16K CTs*, AbdomenAtlas 2.0 (public) & UCSF (private) | Pancreas & kidney tumor segmentation | Table 2, [MICCAI paper](https://www.cs.jhu.edu/~zongwei/publication/bassi2025learning.pdf) (R-Super) | 🤗 [Download](https://huggingface.co/AbdomenAtlas/R-SuperPancreasKidney) |
| **R-Super (Demo)** | *14K CTs*, PanTS (public) & Merlin (public) | Pancreas tumor segmentation | [Demo](rsuper_train/Merlin_demo.md) | 🤗 [Download](https://huggingface.co/AbdomenAtlas/R-SuperPanTSMerlin) |
| **Baseline (Paper, no report supervision)** | *9K CTs*, AbdomenAtlas 2.0 (beta) | Pancreas & kidney tumor segmentation | Table 2, [MICCAI paper](https://www.cs.jhu.edu/~zongwei/publication/bassi2025learning.pdf) (Segmentation) | 🤗 [Download](https://huggingface.co/AbdomenAtlas/RSuperMaskPretrained) |
| **Baseline (Demo, no report supervision)** | *10K CTs*, PanTS | Pancreas tumor segmentation | [Demo](rsuper_train/Merlin_demo.md) | 🤗 [Download](https://huggingface.co/AbdomenAtlas/MedFormerPanTS) |



> The checkpoint 'R-Super (Paper)' is the public segmentation checkpoint trained with **the largest number of lesion CT scans (5K)** that we know of: *2.2K pancreatic lesion CT-Report pairs, 344 pancreatic lesion CT-Mask pairs, 2.7K kidney lesion CT-Report pairs, 1.7K kidney lesion CT-Mask pairs, 9K controls w/o kidney or pancreas tumors.*



## Detailed Code Instructions

These instructions are not needed to follow our demo (see above), but they are useful to train R-Super with your own data, or to modify R-Super. R-Super scales tumor segmentation AI by training with radiology reports, with 3 steps (*click to open each readme*):

[**1- Use LLM to extract tumor information from reports**](report_extraction/README.md)

[**2- Create organ segmentation masks**](organ_masks/README.md)

[**3- Train (and test) R-Super for tumor segmentation, using masks & reports**](rsuper_train/README.md)

### Customizations


<details>
<summary style="margin-left: 25px;">How to use report supervision on your custom segmentation architecture?</summary>
<div style="margin-left: 25px;">

The core of R-Super is its new report supervision loss functions: the Ball Loss and the Volume Loss. To use R-Super with your own architecture, you have 2 options:
1) Just copy our loss functions to your own code. They are at: [rsuper_train/training/losses_foundation.py](rsuper_train/training/losses_foundation.py). The Volume Loss is the function volume_loss_basic, and the Ball Loss is the function ball_loss. To use the losses, first use LLMs to read reports and create organ masks (steps 1 and 2 above). You will also need to prepare your dataset to send these organ masks and report information to the losses (see our dataset at [rsuper_train/training/dataset/dim3/dataset_abdomenatlas_UFO.py](rsuper_train/training/dataset/dim3/dataset_abdomenatlas_UFO.py)).
2) **Alternatively, it may be easier to add your architecture to our code.** To do so, just substitute 'class MedFormer(nn.Module)' in [rsuper_train/model/dim3/medformer.py](rsuper_train/model/dim3/medformer.py) by your own architecture. Just format the output of your architecture like we do (check the function prepare_return). After substituting your architecture in our code, just run the 3 steps above to train it with report supervision and test it.
</details>

<details>
<summary style="margin-left: 25px;">How to develop your own report supervision loss?</summary>
<div style="margin-left: 25px;">

The core of R-Super is its new report supervision loss functions: the Ball Loss and the Volume Loss. They are at: [rsuper_train/training/losses_foundation.py](rsuper_train/training/losses_foundation.py). The Volume Loss is the function volume_loss_basic, and the Ball Loss is the function ball_loss. If you want to develop your own report supervision loss, you can begin by modifying these functions!
</details>

<details>
<summary style="margin-left: 25px;">Public Datasets</summary>
<div style="margin-left: 25px;">

R-Super trains segmentation AI with both CT-Mask pairs (potentially few) and CT-Report pairs. In our paper, our experiments used CT-Mask pairs from AbdomenAtlas 2.0 Beta, and CT-Report from a private dataset from UCSF. In our public demo, we replaced the private report dataset with the public Merlin Dataset, from Stanford, and AbdomenAtlas 2.0 by PanTS, from Johns Hopkins University, the largest dataset with pancreatic tumor masks. Both results are remarkably strong: **UCSF reports improved pancreatic tumor detection F1-Score by 16%, Merlin reports improved it by 10%.** 

PS: Merlin was not public at the time we wrote the MICCAI paper.

</details>


## Novel loss functions: reports supervise segmentation

#### Volume Loss
<div align="center">
  <img src="documents/volume_loss.png" alt="logo" width="800" />
</div>

#### Ball Loss
<div align="center">
  <img src="documents/ball_loss.png" alt="logo" width="800" />
</div>

Beyond the Volume and Ball losses, R-Super adds **report supervision for tumor attenuation** (relative brightness) and a **malignancy supervision** that learns to distinguish malignant from benign lesions using pathology reports — so R-Super not only detects tumors but identifies cancer.


## Papers

**This repository accompanies our new paper:**

<b>Large-Scale Multi-Cancer Detection by Learning Segmentation from Reports</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com/citations?user=NftgL6gAAAAJ&hl=en), Xinze Zhou, [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), Szymon Płotka, [Jieneng Chen](https://scholar.google.com/citations?user=yLYj88sAAAAJ&hl=zh-CN), [Sergio Decherchi](https://scholar.google.com/citations?user=T09qQ1IAAAAJ&hl=it), [Andrea Cavalli](https://scholar.google.com/citations?user=4xTOvaMAAAAJ&hl=en), Arkadiusz Sitek, [Kang Wang](https://radiology.ucsf.edu/people/kang-wang), [Yang Yang](https://scholar.google.com/citations?hl=en&user=6XsJUBIAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)* <br/>
*Johns Hopkins University, UC San Francisco, Harvard Medical School / Massachusetts General Hospital, Stanford University, and collaborators* <br/>
Preprint <br/>
<a href='https://www.cs.jhu.edu/~zongwei/publication/bassi2025scaling.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='https://arxiv.org/abs/2510.14803'><img src='https://img.shields.io/badge/arXiv-2510.14803-b31b1b'></a>

#### Prior conference paper (this work builds on it)

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

```
@inproceedings{bassi2025learning,
  title={Learning Segmentation from Radiology Reports},
  author={Bassi, Pedro RAS and Li, Wenxuan and Chen, Jieneng and Zhu, Zheren and Lin, Tianyu and Decherchi, Sergio and Cavalli, Andrea and Wang, Kang and Yang, Yang and Yuille, Alan L and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={305--315},
  year={2025},
  organization={Springer}
}

@article{bassi2025scaling,
  title={Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks},
  author={Bassi, Pedro RAS and Zhou, Xinze and Li, Wenxuan and P{\l}otka, Szymon and Chen, Jieneng and Chen, Qi and Zhu, Zheren and Prz{\k{a}}do, Jakub and Hamac{\i}, Ibrahim E and Er, Sezgin and others},
  journal={arXiv preprint arXiv:2510.14803},
  year={2025}
}

@article{bassi2025radgpt,
  title={Radgpt: Constructing 3d image-text tumor datasets},
  author={Bassi, Pedro RAS and Yavuz, Mehmet Can and Wang, Kang and Chen, Xiaoxi and Li, Wenxuan and Decherchi, Sergio and Cavalli, Andrea and Yang, Yang and Yuille, Alan and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2501.04678},
  year={2025}
}
```

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. We thank the funding of the Italian Institute of Technology. Paper content is covered by patents pending.

---

## Relationship to Previous Conference Paper

This paper builds on the methodology in our prior conference work "Learning Segmentation From Radiology Reports", winner of the MICCAI 2025 Best Paper Award (top 2 of 1,027 accepted papers). This paper presents several substantive advancements over the conference work:

1. **Scope:** In the conference work, we detected pancreatic and kidney tumors. We now detect seven different tumor types. Unlike pancreatic and kidney tumors, these tumor types have scarce or no publicly available tumor masks and cannot be accurately detected by public AI models. This paper presents the first public AI model shown to surpass radiologists in detecting these cancers on CT.
2. **Scale:** We increase the training dataset from 6,718 to 127,496 CT-Report pairs and add over 1,500 tumor masks created by 31 radiologists. This moves from a proof-of-concept dataset to hospital-scale training.
3. **Validation:** We add a reader study with six radiologists and expand external validation from one hospital with 100 CT scans to three hospitals in three countries, with 1,327 to 2,935 CT scans per test hospital.
4. **Role of reports:** Unlike the conference work, where reports only supplemented tumor masks, R-Super can now use reports to supplement or substitute masks, making reports a key to scaling tumor segmentation.
5. **Richer report supervision:** New and improved loss functions allow R-Super to learn from tumor slice and attenuation information in radiology reports, beyond the report features used in the conference work.
6. **Cancer identification:** R-Super now learns, from pathology reports, how to distinguish cancer from benign lesions, while the conference model detected lesions without distinguishing cancer.

Reference to the conference work: Bassi, P.R.A.S. et al. (2026). Learning Segmentation from Radiology Reports. In: Gee, J.C., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. MICCAI 2025. Lecture Notes in Computer Science, vol 15964. Springer, Cham. https://doi.org/10.1007/978-3-032-04971-1_29

<p align="center">
  <img src="documents/miccai_2025_best_paper_award.png" width="400"/>
</p>

> Recognition: MICCAI 2025 Best Paper Award (runner-up, top 2 of 1,027 papers) · RSNA 2025 Certificate of Merit Award.
