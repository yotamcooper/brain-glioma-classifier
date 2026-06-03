import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.models import load_stage1_model, load_stage2_model
from src.preprocessing import get_mutation_columns, load_uci_glioma, encode_labels, parse_age, prepare_features, transform_patient

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
        threshold = metadata.get("metrics", {}).get("threshold", 0.5)

    image_files = [
        f for f in os.listdir(mri_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        raise ValueError(f"No images found in: {mri_folder}")

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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def predict_grading(patient_data, training_csv='data/TCGA_GBM_LGG_Mutations_all.csv', model=None, metadata=None):
    if model is None:
        model, metadata = load_stage2_model()

    df_train = load_uci_glioma(training_csv)
    X, y, features, label_encoders = encode_labels(df_train)

    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

    scaler = StandardScaler()
    scaler.fit(X_imputed)

    X_patient = transform_patient(patient_data, features, scaler, imputer, label_encoders, df_train)
    prob_gbm = float(model.predict_proba(X_patient)[0][0])
    prob_lgg = float(model.predict_proba(X_patient)[0][1])
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
