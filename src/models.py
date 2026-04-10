import os
import json
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization


# ════════════════════════════════════════════════════════════
#  STAGE 2  —  LGG vs GBM Grading
# ════════════════════════════════════════════════════════════

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'Gradient Boosting':   GradientBoostingClassifier(random_state=42)
    }


def evaluate_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        'AUC':      make_scorer(roc_auc_score),
        'accuracy': 'accuracy'
    }
    results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    return {
        'auc_mean': results['test_AUC'].mean(),
        'auc_std':  results['test_AUC'].std(),
        'acc_mean': results['test_accuracy'].mean(),
        'acc_std':  results['test_accuracy'].std()
    }


def compare_models(X, y):
    """Compare all three classifiers with StratifiedKFold CV."""
    models = get_models()
    all_results = {}
    for name, model in models.items():
        res = evaluate_model(model, X, y)
        all_results[name] = res
        print(f"{name:<25} AUC: {res['auc_mean']:.3f} +/- {res['auc_std']:.3f} | "
              f"Acc: {res['acc_mean']:.3f} +/- {res['acc_std']:.3f}")
    return all_results


def tune_best_model(X, y, best_name):
    """GridSearchCV tuning for the best model found in compare_models."""
    print(f"\n=== Tuning {best_name} ===")

    if best_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

    elif best_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}

    elif best_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2]}

    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X, y)
    print(f"Best params: {grid.best_params_}")
    print(f"Best AUC:    {grid.best_score_:.3f}")
    return grid.best_estimator_


def save_stage2_model(model, model_name, metrics, path="models/stage2_final.joblib"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    metadata = {
        "model_type": model_name,
        "params":     model.get_params(),
        "trained_at": datetime.now().isoformat(),
        "metrics":    metrics
    }
    with open("models/stage2_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Stage 2 model saved to: {path}")


def load_stage2_model(path="models/stage2_final.joblib"):
    model = joblib.load(path)
    with open("models/stage2_metadata.json") as f:
        metadata = json.load(f)
    print(f"Stage 2 model loaded | type: {metadata['model_type']} | trained at: {metadata['trained_at']}")
    return model, metadata


# ════════════════════════════════════════════════════════════
#  STAGE 1  —  Tumor Detector
# ════════════════════════════════════════════════════════════

def build_tumor_detector(input_shape=(224, 224, 3)):

    # Load DenseNet-121 with ImageNet weights, remove its 1000-class top
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # freeze all 121 layers for new head training

    # x is the new head
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def save_stage1_model(model, metrics, history=None, path="models/stage1_final.keras"):
    os.makedirs("models", exist_ok=True)
    model.save(path)
    metadata = {
        "model_type": "DenseNet121",
        "trained_at": datetime.now().isoformat(),
        "metrics": {k: float(v) for k, v in metrics.items()}   
    }
    with open("models/stage1_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    if history is not None:
        serializable_history = {
            phase: {k: [float(v) for v in vals] for k, vals in h.items()}
            for phase, h in history.items()
        }                                                        
        with open("models/stage1_history.json", "w") as f:
            json.dump(serializable_history, f, indent=2)
        print("History saved ✅")
    print(f"Stage 1 model saved to {path}")


def load_stage1_model(path="models/stage1_final.keras"):
    model = load_model(path)
    with open("models/stage1_metadata.json") as f:
        metadata = json.load(f)
    print(f"Stage 1 model loaded | trained at: {metadata['trained_at']}")
    return model, metadata