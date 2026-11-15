import os
import json
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization


# Bump when a change makes previously saved models invalid:
#   1 = Stage 1 trained/tested on the augmented (leaky) split, Stage 2 with LGG as class 1
#   2 = Stage 1 on de-duplicated originals with val-set threshold, Stage 2 with GBM as class 1
PIPELINE_VERSION = 2
STAGE2_MODEL_NAME = 'Soft Voting Ensemble (LR + RF + XGB)'


def _make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     model),
    ])


def get_base_models():
    return [
        ('lr',  _make_pipeline(LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))),
        ('rf',  _make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))),
        ('xgb', _make_pipeline(XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss'))),
    ]


def build_voting_classifier():
    return VotingClassifier(
        estimators=get_base_models(),
        voting='soft',
    )


def cross_validate_models(X, y, n_splits=5):
    """
    Stratified k-fold CV of each base model and of the soft-voting ensemble.
    AUC is computed from predicted probabilities ('roc_auc'), not from hard labels.
    Returns {name: {'auc_mean', 'auc_std', 'acc_mean', 'acc_std'}}.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {'AUC': 'roc_auc', 'accuracy': 'accuracy'}
    candidates = dict(get_base_models())
    candidates['ensemble'] = build_voting_classifier()

    results = {}
    for name, model in candidates.items():
        cv = cross_validate(model, X, y, cv=skf, scoring=scoring)
        results[name] = {
            'auc_mean': float(cv['test_AUC'].mean()),
            'auc_std':  float(cv['test_AUC'].std()),
            'acc_mean': float(cv['test_accuracy'].mean()),
            'acc_std':  float(cv['test_accuracy'].std()),
        }
    return results


def evaluate_voting_model(X, y, n_splits=5):
    return cross_validate_models(X, y, n_splits)['ensemble']


def is_current_model(metadata_path):
    """
    True if a saved model was produced by the current pipeline version. Models
    saved before a data/label fix must be retrained, not loaded — otherwise the
    notebook keeps reporting numbers from the old code.
    """
    try:
        with open(metadata_path) as f:
            return json.load(f).get('pipeline_version') == PIPELINE_VERSION
    except (OSError, ValueError):
        return False


def save_stage2_model(model, metrics, features, label_encoders, path='models/stage2_final.joblib'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, path)
    metadata = {
        'model_type': STAGE2_MODEL_NAME,
        'pipeline_version': PIPELINE_VERSION,
        'trained_at': datetime.now().isoformat(),
        'positive_class': 'GBM',
        'features':   list(features),
        'categories': {col: [str(c) for c in le.classes_] for col, le in label_encoders.items()},
        'metrics':    metrics,
    }
    with open('models/stage2_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)


def load_stage2_metadata():
    with open('models/stage2_metadata.json') as f:
        return json.load(f)


def load_stage2_model(path='models/stage2_final.joblib'):
    return joblib.load(path), load_stage2_metadata()


def build_tumor_detector(input_shape=(224, 224, 3)):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def save_stage1_model(model, metrics, history=None, path='models/stage1_final.keras'):
    os.makedirs('models', exist_ok=True)
    model.save(path)
    metadata = {
        'model_type': 'DenseNet121',
        'pipeline_version': PIPELINE_VERSION,
        'trained_at': datetime.now().isoformat(),
        'metrics':    {k: float(v) for k, v in metrics.items()},
    }
    with open('models/stage1_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    if history is not None:
        serializable_history = {
            phase: {k: [float(v) for v in vals] for k, vals in h.items()}
            for phase, h in history.items()
        }
        with open('models/stage1_history.json', 'w') as f:
            json.dump(serializable_history, f, indent=2)


def load_stage1_metadata():
    with open('models/stage1_metadata.json') as f:
        return json.load(f)


def load_stage1_model(path='models/stage1_final.keras'):
    return load_model(path), load_stage1_metadata()
