import os
import json
from sklearn.model_selection import train_test_split
from src.preprocessing import (
    load_uci_glioma, encode_labels, create_image_generators,
    prepare_stage1_data, prepare_stage2_data,
)
from src.models import (
    build_tumor_detector, build_voting_classifier, cross_validate_models,
    save_stage1_model, load_stage1_model,
    save_stage2_model, load_stage2_model,
    is_current_model, STAGE2_MODEL_NAME,
)
from src.train import train_stage1
from src.evaluate import evaluate_tumor_detector, evaluate_grading_model

MODELS_DIR = 'models'


def _can_load(stage):
    """A saved model is only reused if it was produced by the current pipeline version."""
    return is_current_model(os.path.join(MODELS_DIR, f'{stage}_metadata.json'))


def run_stage1(load_saved=None):
    """load_saved=None (default) loads a saved model only if it is current, else retrains."""
    data_dir = prepare_stage1_data()
    train_gen, val_gen, test_gen = create_image_generators(data_dir)

    if load_saved is None:
        load_saved = _can_load('stage1')
    elif load_saved and not _can_load('stage1'):
        print("WARNING: the saved Stage 1 model predates the current pipeline version; "
              "its metrics are not comparable. Use load_saved=False to retrain.")

    if load_saved:
        model, metadata = load_stage1_model()
        print(f"Stage 1 model loaded | trained at: {metadata.get('trained_at')}")
        history_path = os.path.join(MODELS_DIR, 'stage1_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
        else:
            history = None
        evaluate_tumor_detector(model, val_gen, test_gen, history)
    else:
        model, base_model = build_tumor_detector()
        model, history = train_stage1(model, base_model, train_gen, val_gen)
        metrics = evaluate_tumor_detector(model, val_gen, test_gen, history)
        save_stage1_model(model, metrics, history)
    return model, history


def run_stage2(load_saved=None):
    """load_saved=None (default) loads a saved model only if it is current, else retrains."""
    df = load_uci_glioma(prepare_stage2_data())
    X, y, features, label_encoders = encode_labels(df)
    print(f"Dataset: {len(df)} patients, {len(features)} features "
          f"({int(y.sum())} GBM, {int((1 - y).sum())} LGG)")

    # Stratify on the grade names rather than on y: this reproduces the exact
    # held-out patients used before GBM became class 1, so results stay comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['Grade']
    )

    if load_saved is None:
        load_saved = _can_load('stage2')
    elif load_saved and not _can_load('stage2'):
        print("WARNING: the saved Stage 2 model predates the current pipeline version "
              "(different label encoding); use load_saved=False to retrain.")

    if load_saved:
        best_model, metadata = load_stage2_model()
        best_name = metadata['model_type']
        print(f"Stage 2 model loaded | type: {best_name} | trained at: {metadata.get('trained_at')}")
        return evaluate_grading_model(best_model, X_test, y_test, features, best_name)

    # Model comparison by cross-validation on the training split only.
    cv = cross_validate_models(X_train, y_train)
    print("\n5-fold cross-validation on the training split:")
    print(f"  {'Model':<10} {'AUC':<16} {'Accuracy'}")
    for name, r in cv.items():
        print(f"  {name:<10} {r['auc_mean']:.3f} ± {r['auc_std']:.3f}    "
              f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f}")

    best_model = build_voting_classifier()
    best_model.fit(X_train, y_train)
    test_metrics = evaluate_grading_model(best_model, X_test, y_test, features, STAGE2_MODEL_NAME)
    save_stage2_model(best_model, {'cv': cv, 'test': test_metrics}, features, label_encoders)
    return test_metrics


if __name__ == '__main__':
    run_stage1()
    run_stage2()
