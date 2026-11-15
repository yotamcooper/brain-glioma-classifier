import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.models import (
    load_stage1_model, load_stage1_metadata, load_stage2_model, load_stage2_metadata,
)
from src.preprocessing import transform_patient

IMG_SIZE = (224, 224)

# ════════════════════════════════════════════════════════════
#  STAGE 1  —  Tumor Detection from MRI folder
# ════════════════════════════════════════════════════════════



def predict_tumor(mri_folder, model=None, metadata=None, threshold=None, min_flagged=2):
    """
    Takes a folder of MRI images for one patient.
    Tumor is detected only if max_prob >= threshold AND at least min_flagged scans agree.
    Returns: dict with prediction, confidence, and per-scan results.
    """
    if model is None:
        model, metadata = load_stage1_model()

    if threshold is None:
        if metadata is None:          # model passed in without its metadata
            metadata = load_stage1_metadata()
        threshold = metadata["metrics"]["threshold"]   # chosen on the validation set

    image_files = [
        f for f in os.listdir(mri_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        raise ValueError(f"No images found in: {mri_folder}")
    min_flagged = min(min_flagged, len(image_files))   # a single scan can still be flagged

    probs = []
    scan_results = []

    for fname in sorted(image_files):
        path = os.path.join(mri_folder, fname)
        img  = load_img(path, target_size=IMG_SIZE)
        arr  = img_to_array(img) / 255.0
        arr  = np.expand_dims(arr, axis=0)
        prob = float(model.predict(arr, verbose=0)[0][0])
        probs.append(prob)
        scan_results.append({
            "scan":       fname,
            "tumor_prob": round(prob, 4),
            "flagged":    prob >= threshold
        })

    avg_prob      = float(np.mean(probs))
    max_prob      = float(np.max(probs))
    scans_flagged = sum(r["flagged"] for r in scan_results)

    # Clinical rule: max prob must exceed threshold AND
    # at least min_flagged scans must agree (avoids single artifact false positive)
    tumor_detected = (max_prob >= threshold) and (scans_flagged >= min_flagged)
    prediction     = "Tumor Detected" if tumor_detected else "No Tumor"

    print("\n" + "=" * 50)
    print("  STAGE 1 — Tumor Detection Report")
    print("=" * 50)
    print(f"  Scans analysed  : {len(image_files)}")
    print(f"  Scans flagged   : {scans_flagged} / {len(image_files)}  (min required: {min_flagged})")
    print(f"  Max probability : {max_prob:.4f}  (threshold: {threshold:.4f})")
    print(f"  Avg probability : {avg_prob:.4f}  (for reference)")
    print(f"  Result          : {prediction}")
    print("=" * 50)

    return {
        "prediction":     prediction,
        "tumor_detected": tumor_detected,
        "max_prob":       round(max_prob, 4),
        "avg_prob":       round(avg_prob, 4),
        "threshold":      round(threshold, 4),
        "scans_flagged":  scans_flagged,
        "min_flagged":    min_flagged,
        "total_scans":    len(image_files),
        "scan_results":   scan_results,
    }

# ════════════════════════════════════════════════════════════
#  STAGE 2  —  Tumor Grade prediction based on dict
# ════════════════════════════════════════════════════════════

def predict_grading(patient_data, model=None, metadata=None):
    """
    patient_data: dict of raw values, e.g. {"Age_at_diagnosis": "52 years 180 days",
    "Gender": "Male", "Race": "white", "IDH1": "MUTATED", ...}. Mutations that are
    left out count as NOT_MUTATED; a missing age / gender / race is imputed.

    The model's own pipelines impute and scale (fitted on the training split), so
    the record is passed in raw encoded form — nothing is refitted at inference.
    """
    if model is None:
        model, metadata = load_stage2_model()
    elif metadata is None:            # model passed in without its metadata
        metadata = load_stage2_metadata()
    if 'features' not in metadata or 'categories' not in metadata:
        raise ValueError("This Stage 2 model was saved by an older version of the pipeline. "
                         "Retrain it with run_stage2(load_saved=False).")

    X_patient = transform_patient(patient_data, metadata['features'], metadata['categories'])
    prob_gbm = float(model.predict_proba(X_patient)[0][1])   # class 1 = GBM
    prob_lgg = 1.0 - prob_gbm
    grade    = "GBM (High Grade)" if prob_gbm > prob_lgg else "LGG (Low Grade)"

    print("\n" + "=" * 50)
    print("  STAGE 2 — Glioma Grading Report")
    print("=" * 50)
    print(f"  Model          : {metadata.get('model_type', 'Unknown')}")
    print(f"  GBM probability: {prob_gbm:.4f}")
    print(f"  LGG probability: {prob_lgg:.4f}")
    print(f"  Result         : {grade}")
    print("=" * 50)

    return {
        "grade":    grade,
        "is_gbm":   prob_gbm > prob_lgg,
        "prob_gbm": round(prob_gbm, 4),
        "prob_lgg": round(prob_lgg, 4),
        "model":    metadata.get("model_type"),
    }
