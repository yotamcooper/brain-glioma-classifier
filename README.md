# Brain Glioma Classifier


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yotamcooper/brain-glioma-classifier/blob/main/brain_glioma_classifier.ipynb)


My life changed on the 28/11/25, I was 21, I woke up from a seizure in a hospital bed and later told a tumor was found in my brain. months followed and it was determined I had a low-grade glioma. What followed was brain surgery and a life long meeting with that same strange smell of the doorstep of death and endless doctors' appointments. I spent a lot of time sitting in chairs, watching a diagnosis unfold in real time, a radiologist going through thousands of MRI slices, a pathologist handling actual tissue from my brain, every critical decision resting on a single person being available, alert, and most importently right.

That inefficiency and feeling stuck with me. Not as an abstract problem but as something that deeply changed me as a human. This project is part wanting to understand it well enough to control the dread of it, part wanting to make myself a stronger candidate for research under a PI, and part genuinely believing that the people who end up in those chairs deserve better than a sinking feeling and a system that breaks the moment the right human isn't in the room.

The pipeline mirrors how the hospital actually operates. 
Stage 1 screens MRI scans, the same first step a radiologist takes. 
Stage 2 only activates if a tumor is found, using mutation profiles and clinical data to distinguish LGG from GBM. The same distinction that drives treatment decisions. A clinician could upload a folder of MRI images and/ or a mutation CSV and the system handles the rest, removing the screening bottleneck that currently depends entirely on human bandwidth.

## Pipeline Overview

```
Unknown patient MRI scan
        |
        v
Stage 1: Healthy vs Brain Tumor Detection  <- "Does this patient have a brain tumor?"
        |
        | If TUMOR
        v
Stage 2: LGG vs GBM Grading Classifier    <- "How aggressive is the tumor?"
```

## Stage 1: Healthy vs Tumor Detection

Classifies MRI brain scans as healthy or tumor using a fine-tuned DenseNet-121 CNN.

**Dataset:** [Brain Tumor MRI Classification Dataset (Tumor vs. No Tumor)](https://data.mendeley.com/datasets/w56x9jrhxr/1) - Mendeley Data
- Brain MRI slices across two classes: 3,671 original Tumor images | 13,273 No Tumor images. The slices mix sequences (T1 with and without contrast, T2, FLAIR, diffusion) and planes (axial, coronal, sagittal)
- The dataset also ships a pre-augmented Tumor folder (13,252 rotated copies). **It is not used for splitting**: rotated copies of the same scan would land in both train and test and leak information into the evaluation. Only the original images are split.
- Class imbalance is handled with class weights in the loss, and augmentation (rotation, shift, flip, zoom) is applied on the fly to the training set only
- Images resized to 224×224 to match DenseNet-121 input requirements
- Split: 70% train / 15% val / 15% test (stratified, done in `preprocessing.py`)
- Binary classification: No Tumor / Tumor
- Two phase training: 1) head only phase with DenseNet frozen → 2) fine-tune last 50 DenseNet layers with head

**Results:**

Measured on the held-out test set (2,496 images: 1,946 No Tumor / 550 Tumor). The split is built from the original, de-duplicated images only, and the decision threshold is chosen on the validation set (Youden's J), not on the test set.

| Metric | Value |
|---|---|
| AUC | 0.9790 |
| Accuracy | 93.15% |
| Threshold (chosen on validation) | 0.2849 |
| Sensitivity (TPR) | 91.82% |
| Specificity (TNR) | 93.53% |
| False Negatives | 45 (missed tumors) |
| False Positives | 126 (healthy misclassified) |

An earlier version of this project reported AUC 0.9958 and 97.31% accuracy. Those numbers were inflated: rotated copies of training scans were in the test set, and the threshold was tuned on the test set itself. The numbers above replace them.

### Stage 2: LGG vs GBM Grading

Classifies glioma patients as Low Grade Glioma (LGG) or Glioblastoma (GBM) using mutation profiles and clinical features with a Soft Voting Ensemble (Logistic Regression + Random Forest + XGBoost).

**Dataset:** [UCI Glioma Grading Dataset](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)
- 862 patients (499 LGG, 363 GBM)
- 20 mutation features + age, gender, race

**Model selection:** 5 fold cross-validation on the training split (689 patients). AUC is computed from predicted probabilities:

| Model | CV AUC (5 fold) | CV Accuracy |
|---|---|---|
| Logistic Regression | 0.914 ± 0.023 | 0.875 ± 0.038 |
| Random Forest | 0.892 ± 0.026 | 0.834 ± 0.030 |
| XGBoost | 0.895 ± 0.023 | 0.821 ± 0.025 |
| **Soft Voting Ensemble** | 0.909 ± 0.021 | 0.848 ± 0.029 |

Rather than selecting a single best model, all three were combined into a **Soft Voting Ensemble**. Each model outputs a class probability, the ensemble averages these probabilities and predicts the class with the higher mean score. This approach reduces the risk of any one model's systematic blind spots driving the final decision which is  particularly important in a clinical context where a confidently wrong prediction has real consequences. In cross-validation the ensemble and plain logistic regression are within one standard deviation of each other, so the ensemble is a choice for robustness, not a measured gain in accuracy.

**Ensemble results** (held-out test set, 173 patients: 73 GBM / 100 LGG). GBM is the positive class, so sensitivity is the share of GBM patients caught:

| Metric | Value |
|---|---|
| AUC | 0.9218 |
| Accuracy | 87.28% |
| Sensitivity (GBM caught) | 86.30% |
| Specificity (LGG correctly called LGG) | 88.00% |
| False Negatives | 10 (GBM called LGG) |
| False Positives | 12 (LGG called GBM) |

Strongest features (largest logistic regression coefficients): IDH1, NOTCH1, Age At Diagnosis, TP53 and IDH2. IDH1, NOTCH1 and IDH2 mutations push the prediction towards LGG, older age and TP53 towards GBM.

**A note on the biology (WHO CNS5, 2021).** IDH1 coming out as the top feature is not a surprise — under the 2021 WHO classification of CNS tumours, glioblastoma is *defined* as IDH-wildtype; an IDH-mutant tumour cannot be a glioblastoma. The TCGA LGG/GBM labels used here follow the older, histology-based grading, so part of what the model learns is the molecular rule that the new classification made explicit. A natural next step is to relabel the cohort with WHO 2021 categories (IDH status, 1p/19q codeletion, CDKN2A/B, etc.) and see which features still matter.

## Limitations

- **Stage 1 and Stage 2 are trained on different datasets and different patients.** There is no cohort with both MRI and mutation data, so the two-stage pipeline has not been validated end to end.
- The Stage 1 dataset gives no patient IDs, so images are split per image, not per patient. If several slices come from the same person, some leakage between splits is still possible.
- The Stage 1 dataset is a single upload from a student thesis project, and its authors do not say which hospital or scanner the scans come from, how many patients there are, or how "No Tumor" was decided. The mix of sequences and image sizes is not the same in the two classes (some image sizes only occur in the Tumor folder), so part of the Stage 1 score may come from recognising the type of scan rather than the tumor. It should not be read as performance on scans from a new hospital.
- The Stage 2 test set is 173 patients, so its metrics move by a few points with a different split (one patient is about 0.6 points of accuracy).
- Research project only — not a medical device and not for clinical use.

## Project Structure

```text
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # data loading, encoding, image generators, patient transforms
│   ├── models.py          # model definitions, CV, tuning, save/load
│   ├── train.py           # CNN training loop (Stage 1)
│   ├── evaluate.py        # all plots and evaluation for both stages
│   ├── pipeline.py        # orchestrates training end to end
│   ├── predict.py         # inference functions for Stage 1 and Stage 2
│   └── drive_cache.py     # optional: keep trained models on Google Drive (Colab)
├── data/                  # downloaded on first run, not in git
│   ├── Brain Tumor MRI Dataset Raw/   # original images
│   ├── Brain Tumor MRI Dataset/       # train / val / test split
│   └── TCGA_GBM_LGG_Mutations_all.csv
├── models/                # saved models + metadata JSON (not in git)
├── figures/               # all output plots (not in git)
├── brain_glioma_classifier.ipynb
└── requirements.txt 
```

## How to Run

Open the notebook via the badge above and run all cells. Each cell has a title that guides you through the steps. It also runs locally: `pip install -r requirements.txt`, then open the notebook from the repo folder (or run `python -m src.pipeline`).

- **Data** is downloaded automatically on the first run — the MRI originals from Mendeley Data and the glioma table from UCI — and the train / val / test split is built from it. The split is seeded, so every run gets the same test set.
- **Models** are retrained when no up-to-date saved model is found. Each saved model carries a `pipeline_version`; a model saved by an older version of the code (for example one trained on the augmented split) is ignored, never loaded. In Colab the trained models are cached in `MyDrive/brain-glioma-classifier/models/` so later sessions skip training. Stage 1 needs a GPU runtime to train in reasonable time; Stage 2 takes seconds.

📁 [Google Drive Folder](https://drive.google.com/drive/folders/1iZKPwfNk8ybHo9LXgjmq_hX7k6h5Eyu6?usp=sharing) — pretrained models

## Acknowledgements

**Data**

- Emran A.S.H., Akter H., Nayem F.M., Shiam A.A. *Brain Tumor MRI Classification Dataset (Tumor vs. No Tumor)*. Mendeley Data, V1, 2025. [doi:10.17632/w56x9jrhxr.1](https://doi.org/10.17632/w56x9jrhxr.1) (CC BY 4.0)
- Tasci E., Camphausen K., Krauze A.V., Zhuge Y. *Glioma Grading Clinical and Mutation Features*. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset), 2022. Built from TCGA-LGG and TCGA-GBM.

**Reading that shaped this project**

- Smith H.L., Wadhwani N., Horbinski C. Major Features of the 2021 WHO Classification of CNS Tumors. *Neurotherapeutics*, 2022. [doi:10.1007/s13311-022-01249-0](https://doi.org/10.1007/s13311-022-01249-0)
- Tasci E., Zhuge Y., Kaur H., Camphausen K., Krauze A.V. Hierarchical Voting-Based Feature Selection and Ensemble Learning Model Scheme for Glioma Grading with Clinical and Molecular Characteristics. *Int. J. Mol. Sci.* 2022, 23, 14155. [doi:10.3390/ijms232214155](https://doi.org/10.3390/ijms232214155)
- Rasheed M., Jaffar M.A., Akram A., Rashid J., Alshalali T.A.N., Irshad A., Sarwar N. Improved brain tumor classification through DenseNet121 based transfer learning. *Discover Oncology* 16, 1645 (2025). [doi:10.1007/s12672-025-03501-3](https://doi.org/10.1007/s12672-025-03501-3)
- Li X., Huang X., Shen Y., et al. Machine learning for grading prediction and survival analysis in high grade glioma. *Scientific Reports* 15, 16955 (2025). [doi:10.1038/s41598-025-01413-4](https://doi.org/10.1038/s41598-025-01413-4)
- Rajakaruna P., Rios S., Elnahas H., et al. Molecular Biomarkers of Glioma. *Biomedicines* 2025, 13, 1298. [doi:10.3390/biomedicines13061298](https://doi.org/10.3390/biomedicines13061298)
- Neill S.G., Hauenstein J., Li M.M., Liu Y.J., Luo M., Saxe D.F., Ligon A.H. Copy number assessment in the genomic analysis of CNS neoplasia: An evidence-based review from the Cancer Genomics Consortium (CGC) working group on primary CNS tumors. *Cancer Genetics* 243 (2020) 19–47. [doi:10.1016/j.cancergen.2020.02.004](https://doi.org/10.1016/j.cancergen.2020.02.004)
- Ivchenko O. CNN Architectures for Medical Imaging: From ResNet to EfficientNet. Online article, 8 February 2026.
