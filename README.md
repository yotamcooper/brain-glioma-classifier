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
- T1-weighted brain MRI scans across two classes: 3,671 original Tumor images | 13,273 No Tumor images
- The dataset also ships a pre-augmented Tumor folder (13,252 rotated copies). **It is not used for splitting**: rotated copies of the same scan would land in both train and test and leak information into the evaluation. Only the original images are split.
- Class imbalance is handled with class weights in the loss, and augmentation (rotation, shift, flip, zoom) is applied on the fly to the training set only
- Images resized to 224×224 to match DenseNet-121 input requirements
- Split: 70% train / 15% val / 15% test (stratified, done in `preprocessing.py`)
- Binary classification: No Tumor / Tumor
- Two phase training: 1) head only phase with DenseNet frozen → 2) fine-tune last 50 DenseNet layers with head

**Results:**

> ⚠️ **These numbers predate the leakage fix and are likely optimistic.** They were measured with augmented copies of training scans in the test set, and with the decision threshold tuned on the test set itself. The split is now built from the original, de-duplicated images and the threshold is chosen on the validation set. The table will be replaced once Stage 1 has been retrained on the new split (see *How to Run*) — expect lower numbers.

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
- The Stage 1 images come from several sources with different resolutions and formats, and the mix is not the same in the two classes (some image sizes only occur in the Tumor folder). Part of the Stage 1 score may come from recognising the source rather than the tumor, so it should not be read as performance on scans from a new hospital.
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
