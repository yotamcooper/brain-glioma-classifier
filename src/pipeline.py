import os
import json
from sklearn.model_selection import train_test_split
from src.preprocessing import load_uci_glioma, encode_labels, create_image_generators
from src.models import (
    build_tumor_detector, build_voting_classifier, evaluate_voting_model,
    save_stage1_model, load_stage1_model,
    save_stage2_model, load_stage2_model,
)
from src.train import train_stage1
from src.evaluate import evaluate_tumor_detector, evaluate_grading_model

MODELS_DIR = 'models'


def run_stage1(load_saved=False):
    train_gen, val_gen, test_gen = create_image_generators()

    if load_saved:
        model, metadata = load_stage1_model()
        history_path = os.path.join(MODELS_DIR, 'stage1_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
        else:
            history = None
        evaluate_tumor_detector(model, test_gen, history)
    else:
        model, base_model = build_tumor_detector()
        model, history = train_stage1(model, base_model, train_gen, val_gen)
        metrics = evaluate_tumor_detector(model, test_gen, history)
        save_stage1_model(model, metrics, history)
    return model, history


def run_stage2(load_saved=False):
    df = load_uci_glioma('data/TCGA_GBM_LGG_Mutations_all.csv')
    X, y, features, _ = encode_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if load_saved:
        best_model, metadata = load_stage2_model()
        best_name = metadata['model_type']
    else:
        best_model = build_voting_classifier()
        best_model.fit(X_train, y_train)
        metrics = evaluate_voting_model(X_train, y_train)
        save_stage2_model(best_model, metrics)
        best_name = "Soft Voting Ensemble (LR + RF + XGB)"

    evaluate_grading_model(best_model, X_test, y_test, features, best_name)



if __name__ == '__main__':
    run_stage1()
    run_stage2()
