import os
import json
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization


def _make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     model),
    ])


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'Gradient Boosting':   GradientBoostingClassifier(random_state=42),
    }


def evaluate_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {'AUC': make_scorer(roc_auc_score), 'accuracy': 'accuracy'}
    results = cross_validate(_make_pipeline(model), X, y, cv=skf, scoring=scoring)
    return {
        'auc_mean': results['test_AUC'].mean(),
        'auc_std':  results['test_AUC'].std(),
        'acc_mean': results['test_accuracy'].mean(),
        'acc_std':  results['test_accuracy'].std(),
    }


def compare_models(X, y):
    all_results = {}
    for name, model in get_models().items():
        all_results[name] = evaluate_model(model, X, y)
    return all_results


def tune_best_model(X, y, best_name):
    if best_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100]}
    elif best_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [None, 5, 10]}
    elif best_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.05, 0.1, 0.2]}

    grid = GridSearchCV(_make_pipeline(model), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_


def save_stage2_model(model, model_name, metrics, path='models/stage2_final.joblib'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, path)
    metadata = {
        'model_type': model_name,
        'params':     {k: str(v) for k, v in model.get_params().items()},
        'trained_at': datetime.now().isoformat(),
        'metrics':    metrics,
    }
    with open('models/stage2_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)


def load_stage2_model(path='models/stage2_final.joblib'):
    model = joblib.load(path)
    with open('models/stage2_metadata.json') as f:
        metadata = json.load(f)
    return model, metadata


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


def load_stage1_model(path='models/stage1_final.keras'):
    model = load_model(path)
    with open('models/stage1_metadata.json') as f:
        metadata = json.load(f)
    return model, metadata
