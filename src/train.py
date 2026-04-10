import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


MODELS_DIR = "models"


# ════════════════════════════════════════════════════════════
#  STAGE 1  —  Tumor Detector
# ════════════════════════════════════════════════════════════

def train_stage1(model, base_model, train_gen, val_gen):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── PHASE 1: head only, base frozen ───────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1 — Training head only (DenseNet base frozen)")
    print("=" * 60)

    _print_param_count(model, label="Phase 1")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["AUC", "accuracy"],
    )

    print("\nStarting Phase 1 (up to 10 epochs)...")
    h1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=_get_callbacks(
            phase=1,
            checkpoint_path=os.path.join(MODELS_DIR, "stage1_phase1_best.keras"),
        ),
        verbose=1,
    )

    # ── PHASE 2: unfreeze last 50 DenseNet layers ─────────
    print("\n" + "=" * 60)
    print("  PHASE 2 — Fine-tuning last 50 DenseNet layers")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    print(f"Total DenseNet layers : {len(base_model.layers)}")
    print(f"Trainable (last 50)   : fine-tuning for MRI patterns")
    print(f"Frozen (rest)         : edges, textures — universal features")

    _print_param_count(model, label="Phase 2")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["AUC", "accuracy"],
    )

    print("\nStarting Phase 2 (up to 20 epochs)...")
    h2 = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=_get_callbacks(
            phase=2,
            checkpoint_path=os.path.join(MODELS_DIR, "stage1_phase2_best.keras"),
        ),
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Phase 1 best weights: {MODELS_DIR}/stage1_phase1_best.keras")
    print(f"  Phase 2 best weights: {MODELS_DIR}/stage1_phase2_best.keras")
    print("=" * 60)

    return model, {"phase1": h1.history, "phase2": h2.history}


# ════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════

def _get_callbacks(phase, checkpoint_path):
    callbacks = [
        EarlyStopping(
            monitor="val_AUC",
            patience=5 if phase == 1 else 7,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_AUC",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    if phase == 2:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_AUC",
                factor=0.5,
                patience=3,
                mode="max",
                min_lr=1e-7,
                verbose=1,
            )
        )

    return callbacks


def _print_param_count(model, label=""):
    trainable = sum(w.numpy().size for w in model.trainable_weights)
    total     = sum(w.numpy().size for w in model.weights)
    frozen    = total - trainable
    print(f"\n[{label}] Parameter counts:")
    print(f"  Trainable : {trainable:>10,}")
    print(f"  Frozen    : {frozen:>10,}")
    print(f"  Total     : {total:>10,}")