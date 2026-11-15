"""Keep trained models on Google Drive between Colab sessions (optional)."""
import os
import shutil
from src.models import is_current_model

STAGE_FILES = {
    'stage1': ['stage1_final.keras', 'stage1_metadata.json', 'stage1_history.json',
               'stage1_phase1_best.keras', 'stage1_phase2_best.keras'],
    'stage2': ['stage2_final.joblib', 'stage2_metadata.json'],
}


def restore(stage, drive_base, models_dir='models'):
    """
    Copy a cached model from Drive into models/ — but only if the current pipeline
    version trained it. A model from before a data/label fix is left where it is,
    so the stage retrains instead of reporting numbers from the old code.
    Returns True if a current model is now available locally.
    """
    meta = f'{stage}_metadata.json'
    if drive_base and is_current_model(os.path.join(drive_base, 'models', meta)):
        os.makedirs(models_dir, exist_ok=True)
        for name in STAGE_FILES[stage]:
            src = os.path.join(drive_base, 'models', name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(models_dir, name))
    elif drive_base and os.path.exists(os.path.join(drive_base, 'models', meta)):
        print(f"The {stage} model on Drive was trained by an older version of the "
              "pipeline — ignoring it and retraining.")
    return is_current_model(os.path.join(models_dir, meta))


def store(stage, drive_base, models_dir='models', figures_dir='figures'):
    """Copy freshly trained models and figures to Drive. An older model is moved to
    models/previous/ rather than overwritten."""
    if not drive_base:
        return
    drive_models = os.path.join(drive_base, 'models')
    os.makedirs(os.path.join(drive_models, 'previous'), exist_ok=True)
    for name in STAGE_FILES[stage]:
        src, dst = os.path.join(models_dir, name), os.path.join(drive_models, name)
        if os.path.exists(dst):
            shutil.move(dst, os.path.join(drive_models, 'previous', name))
        if os.path.exists(src):
            shutil.copy2(src, dst)
    if os.path.isdir(figures_dir):
        shutil.copytree(figures_dir, os.path.join(drive_base, 'figures'), dirs_exist_ok=True)
