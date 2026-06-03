import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA


FIGURES_DIR = "figures"


def _impute_for_viz(model, X):
    lr = model.named_estimators_["lr"] if hasattr(model, "named_estimators_") else model
    return lr[:-1].transform(X)


def plot_roc(y_true, y_probs, title="ROC Curve", save_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"AUC: {roc_auc:.3f} | Optimal threshold: {optimal_threshold:.3f}")
    print(f"At threshold -> FPR: {fpr[optimal_idx]:.3f}, TPR: {tpr[optimal_idx]:.3f}")
    return roc_auc


def plot_pca(X, y, class_names, title="PCA Projection", save_path=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    colors = ["steelblue", "tomato"]
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=colors[i], alpha=0.6, label=name,
                    edgecolor="k", s=40)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances),
                   key=lambda x: x[1], reverse=True)[:top_n]
    names, scores = zip(*pairs)

    print("\n--- Feature Importance ---")
    print(f"{'Feature':<20} | {'Importance':<10}")
    print("-" * 33)
    for name, score in pairs:
        print(f"{name:<20} | {score:.4f}")

    plt.figure(figsize=(10, 5))
    plt.barh(list(reversed(names)), list(reversed(scores)), color="steelblue")
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return dict(pairs)


def plot_lr_coefficients(model, feature_names, save_path=None):
    clf = model.named_steps["clf"]
    coefs = pd.Series(clf.coef_[0], index=feature_names).sort_values()
    plt.figure(figsize=(10, 6))
    coefs.plot(kind="barh", color=["tomato" if c > 0 else "steelblue" for c in coefs])
    plt.axvline(x=0, color="black", linewidth=0.8)
    plt.title("Logistic Regression Coefficients -- LGG vs GBM")
    plt.xlabel("Coefficient Value (positive = GBM, negative = LGG)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(history):
    if history is None:
        return
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stage 1 - Training Curves", fontsize=14, fontweight="bold")

    phases = ["phase1", "phase2"]
    titles = ["Phase 1: Head Training", "Phase 2: Fine-Tuning"]

    for col, (phase, title) in enumerate(zip(phases, titles)):
        h      = history[phase]
        epochs = range(1, len(h["loss"]) + 1)

        axes[0][col].plot(epochs, h.get("AUC", h.get("auc")), label="Train AUC", color="steelblue")
        axes[0][col].plot(epochs, h.get("val_AUC", h.get("val_auc")), label="Val AUC", color="tomato", linestyle="--")
        axes[0][col].set_title(f"{title} - AUC")
        axes[0][col].set_xlabel("Epoch")
        axes[0][col].set_ylabel("AUC")
        axes[0][col].legend()
        axes[0][col].grid(True, alpha=0.3)

        axes[1][col].plot(epochs, h["loss"],     label="Train Loss", color="steelblue")
        axes[1][col].plot(epochs, h["val_loss"], label="Val Loss",   color="tomato", linestyle="--")
        axes[1][col].set_title(f"{title} - Loss")
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylabel("Loss")
        axes[1][col].legend()
        axes[1][col].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "stage1_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved to: {save_path}")


# ════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════


def _plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved to: {save_path}")


def _save_table_as_png(rows, title, save_path):
    fig, ax = plt.subplots(figsize=(6, len(rows) * 0.5 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 1.6)

    for col in range(len(rows[0])):
        table[0, col].set_facecolor("#2c5f8a")
        table[0, col].set_text_props(color="white", fontweight="bold")

    for row in range(1, len(rows)):
        colour = "#f0f4f8" if row % 2 == 0 else "white"
        for col in range(len(rows[0])):
            table[row, col].set_facecolor(colour)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Results table saved to: {save_path}")


# ════════════════════════════════════════════════════════════
#  STAGE 1  —  Tumor Detector
# ════════════════════════════════════════════════════════════


def evaluate_tumor_detector(model, test_gen, history):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_training_curves(history)

    y_true  = test_gen.classes
    y_probs = model.predict(test_gen, verbose=1).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    optimal_idx       = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_probs >= optimal_threshold).astype(int)

    auc_score = roc_auc_score(y_true, y_probs)
    accuracy  = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    rows = [
        ["Metric",             "Value"],
        ["AUC Score",          f"{auc_score:.4f}"],
        ["Accuracy",           f"{accuracy * 100:.2f}%"],
        ["Optimal Threshold",  f"{optimal_threshold:.4f}"],
        ["Sensitivity (TPR)",  f"{sensitivity * 100:.2f}%"],
        ["Specificity (TNR)",  f"{specificity * 100:.2f}%"],
        ["False Negatives",    f"{fn}  (missed tumors)"],
        ["False Positives",    f"{fp}  (healthy misclassified)"],
    ]

    print("\n" + "=" * 45)
    print("  Stage 1 - Tumor Detector Results")
    print("=" * 45)
    for label, value in rows[1:]:
        print(f"  {label:<22}: {value}")
    print("=" * 45)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))

    _save_table_as_png(
        rows,
        title="Stage 1 - Tumor Detector Results",
        save_path=os.path.join(FIGURES_DIR, "results_table_stage1.png")
    )

    plot_roc(
        y_true, y_probs,
        title="ROC Curve - Tumor vs No Tumor",
        save_path=os.path.join(FIGURES_DIR, "roc_tumor_detector.png")
    )

    _plot_confusion_matrix(
        cm,
        class_names=["No Tumor", "Tumor"],
        title="Confusion Matrix - Tumor Detector",
        save_path=os.path.join(FIGURES_DIR, "confusion_matrix_stage1.png")
    )

    return {
        "auc":         auc_score,
        "accuracy":    accuracy,
        "threshold":   optimal_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ════════════════════════════════════════════════════════════
#  STAGE 2  —  LGG vs GBM Grading
# ════════════════════════════════════════════════════════════


def evaluate_grading_model(model, X_test, y_test, features, model_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    auc_score   = roc_auc_score(y_test, y_probs)
    accuracy    = accuracy_score(y_test, y_pred)
    cm_metrics  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm_metrics.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    rows = [
        ["Metric",             "Value"],
        ["Model",              model_name],
        ["AUC Score",          f"{auc_score:.4f}"],
        ["Accuracy",           f"{accuracy * 100:.2f}%"],
        ["Sensitivity (TPR)",  f"{sensitivity * 100:.2f}%"],
        ["Specificity (TNR)",  f"{specificity * 100:.2f}%"],
        ["False Negatives",    f"{fn}  (missed GBM)"],
        ["False Positives",    f"{fp}  (LGG misclassified)"],
    ]

    print("\n" + "=" * 45)
    print("  Stage 2 - Grading Model Results")
    print("=" * 45)
    for label, value in rows[1:]:
        print(f"  {label:<22}: {value}")
    print("=" * 45)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["GBM", "LGG"]))

    _save_table_as_png(
        rows,
        title="Stage 2 - Grading Model Results",
        save_path=os.path.join(FIGURES_DIR, "results_table_stage2.png")
    )

    X_transformed = _impute_for_viz(model, X_test)

    plot_pca(
        X_transformed, y_test,
        class_names=["GBM", "LGG"],
        title=f"PCA: LGG vs GBM — {model_name}",
        save_path=os.path.join(FIGURES_DIR, "pca_lgg_gbm.png")
    )

    plot_roc(
        y_test, y_probs,
        title=f"ROC Curve — LGG vs GBM ({model_name})",
        save_path=os.path.join(FIGURES_DIR, "roc_lgg_gbm.png")
    )

    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred),
        class_names=["GBM", "LGG"],
        title=f"Confusion Matrix — {model_name}",
        save_path=os.path.join(FIGURES_DIR, "confusion_matrix_stage2.png")
    )

    if hasattr(model, "feature_importances_"):
        plot_feature_importance(
            model, features,
            save_path=os.path.join(FIGURES_DIR, "feature_importance.png")
        )
    else:
        plot_lr_coefficients(
            model, features,
            save_path=os.path.join(FIGURES_DIR, "feature_importance.png")
        )
