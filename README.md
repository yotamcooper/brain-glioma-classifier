# Brain Glioma Classifier


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZKWtcXWFm8m5a00SHi5cGMu0x21xe9TI?usp=sharing)


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
- ~26,500 T1-weighted brain MRI scans across two classes
- Tumor (after augmentation): 13,252 images | No Tumor: 13,273 images
- Original dataset was imbalanced (3,671 tumor vs 13,273 no-tumor) - tumor class augmented on disk using rotation and normalization
- Images resized to 224×224 to match DenseNet-121 input requirements
- Split: 70% train / 15% val / 15% test (stratified, done in `preprocessing.py`)
- Binary classification: No Tumor / Tumor
- Two phase training: 1) head only phase with DenseNet frozen → 2) fine-tune last 50 DenseNet layers with head

**Results:**

| Metric | Value |
|---|---|
| AUC | 0.9958 |
| Accuracy | 97.31% |
| Optimal Threshold | 0.2078 |
| Sensitivity (TPR) | 96.73% |
| Specificity (TNR) | 97.89% |
| False Negatives | 65 (missed tumors) |
| False Positives | 42 (healthy misclassified) |

### Stage 2: LGG vs GBM Grading

Classifies glioma patients as Low Grade Glioma (LGG) or Glioblastoma (GBM) using mutation profiles and clinical features with a Soft Voting Ensemble (Logistic Regression + Random Forest + XGBoost).

**Dataset:** [UCI Glioma Grading Dataset](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)
- 862 patients (499 LGG, 363 GBM)
- 20 mutation features + age, gender, race

**Model selection:** 5 fold cross-validation across three candidate models:

| Model | CV AUC (5 fold) | CV Accuracy |
|---|---|---|
| Logistic Regression | 0.882 ± 0.013 | 0.876 ± 0.014 |
| Gradient Boosting | 0.866 ± 0.013 | 0.861 ± 0.013 |
| Random Forest | 0.842 ± 0.023 | 0.842 ± 0.019 |

Rather than selecting a single best model, all three were combined into a **Soft Voting Ensemble**. Each model outputs a class probability, the ensemble averages these probabilities and predicts the class with the higher mean score. This approach reduces the risk of any one model's systematic blind spots driving the final decision which is  particularly important in a clinical context where a confidently wrong prediction has real consequences.

**Ensemble results** (held-out test set):

| Metric | Value |
|---|---|
| AUC | 0.9229 |
| Accuracy | 87.28% |
| Sensitivity (TPR) | 88% |
| Specificity (TNR) | 86.3% |
| False Negatives | 12 (missed GBM) |
| False Positives | 10 (LGG misclassified) |

Top predictive features : IDH1, NOTCH1, IDH2, TP53 and Age At Diagnosis

## Project Structure

```text
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # data loading, encoding, image generators, patient transforms
│   ├── models.py          # model definitions, CV, tuning, save/load
│   ├── train.py           # CNN training loop (Stage 1)
│   ├── evaluate.py        # all plots and evaluation for both stages
│   ├── pipeline.py        # orchestrates training end to end
│   └── predict.py         # inference functions for Stage 1 and Stage 2
├── data/
│   ├── Brain Tumor MRI Dataset/
│   └── TCGA_GBM_LGG_Mutations_all.csv
├── models/                # saved models + metadata JSON
├── figures/               # all output plots
├── brain_glioma_classifier.ipynb
└── requirements.txt 
```

## How to Run

Open the notebook via the badge above and run all cells. Each cell has a title that guides you through the steps.

📁 [Google Drive Folder](https://drive.google.com/drive/folders/1iZKPwfNk8ybHo9LXgjmq_hX7k6h5Eyu6?usp=sharing) — dataset and pretrained models

