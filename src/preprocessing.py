import os
import json
import shutil
import hashlib
import zipfile
import tempfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Stage 2 label convention: GBM is the positive class (1), LGG is 0.
# "Sensitivity" therefore means "share of GBM patients caught".
GRADE_NAMES = ['LGG', 'GBM']
CATEGORICAL_COLS = ['Gender', 'Race']

RAW_DIR = 'data/Brain Tumor MRI Dataset Raw'
SPLIT_DIR = 'data/Brain Tumor MRI Dataset'
SPLIT_INFO = 'split_info.json'
SPLIT_VERSION = 2   # 1 = old split of the augmented folder (leaky), 2 = originals, de-duplicated
MENDELEY_ZIP_URL = 'https://data.mendeley.com/public-api/zip/w56x9jrhxr/download/1'

GLIOMA_CSV = 'data/TCGA_GBM_LGG_Mutations_all.csv'
UCI_ZIP_URL = ('https://archive.ics.uci.edu/static/public/759/'
               'glioma+grading+clinical+and+mutation+features+dataset.zip')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def _download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def prepare_stage2_data(path=GLIOMA_CSV):
    """Return the path of the UCI glioma table, downloading it from UCI if missing."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    print("Downloading the glioma grading table from the UCI repository...")
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, 'uci.zip')
        _download(UCI_ZIP_URL, archive)
        with zipfile.ZipFile(archive) as z, z.open(os.path.basename(path)) as src, open(path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    return path


def load_uci_glioma(path=GLIOMA_CSV):
    return pd.read_csv(path)


def get_mutation_columns(df):
    non_mutation = ['Case_ID', 'Project', 'Primary_Diagnosis', 'Grade', 'Gender', 'Race', 'Age_at_diagnosis']
    return [c for c in df.columns if c not in non_mutation]


def parse_age(age_str):
    """'51 years 108 days' -> 51.3. Anything unparseable ('--', missing) -> NaN."""
    parts = str(age_str).split()
    if pd.isna(age_str) or not parts or not parts[0].isdigit():
        return np.nan
    years = int(parts[0])
    days = int(parts[2]) if len(parts) >= 4 and parts[2].isdigit() else 0
    return round(years + days / 365, 2)


def prepare_features(X, mutation_cols):
    X = X.copy()
    X['Age_at_diagnosis'] = X['Age_at_diagnosis'].apply(parse_age)
    X[mutation_cols] = (X[mutation_cols] == 'MUTATED').astype(int)
    return X


def encode_labels(df):
    y = (df['Grade'] == 'GBM').astype(int).to_numpy()   # GBM = 1 (positive), LGG = 0
    X = df.drop(columns=['Grade', 'Case_ID', 'Project', 'Primary_Diagnosis'])
    X = prepare_features(X, get_mutation_columns(df))

    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))
        label_encoders[col] = le_col

    return X, y, X.columns.tolist(), label_encoders


def _file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download_raw_dataset(dest=RAW_DIR, url=MENDELEY_ZIP_URL):
    """
    Download the ORIGINAL images from Mendeley Data into `dest`
    (folders 'Tumor(Original)' and 'No Tumor'). The pre-augmented tumor
    folder is deliberately not extracted.
    """
    os.makedirs(dest, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        outer = os.path.join(tmp, 'mendeley.zip')
        print("Downloading the MRI dataset from Mendeley Data (~285 MB)...")
        _download(url, outer)
        with zipfile.ZipFile(outer) as z:
            inner_name = next(n for n in z.namelist() if n.lower().endswith('.zip'))
            inner = z.extract(inner_name, tmp)
        n = 0
        with zipfile.ZipFile(inner) as z:
            for member in z.namelist():
                parts = member.replace('\\', '/').split('/')
                if len(parts) < 2 or not parts[-1].lower().endswith(IMAGE_EXTS):
                    continue
                folder = parts[-2]
                if 'augment' in folder.lower():
                    continue
                os.makedirs(os.path.join(dest, folder), exist_ok=True)
                with z.open(member) as src, open(os.path.join(dest, folder, parts[-1]), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    print(f"Extracted {n:,} original images to {dest}")
    return dest


def split_dataset(source_dir=RAW_DIR,
                  output_dir=SPLIT_DIR,
                  val_ratio=0.15, test_ratio=0.15, seed=42,
                  folder_name_map=None):
    """
    Split the ORIGINAL (non-augmented) images into train / val / test.

    The Mendeley dataset ships a 'Tumor (Augmented)' folder containing rotated
    copies of the 3,671 original tumor scans. Splitting that folder puts copies of
    the same scan into both train and test, so the test set is no longer unseen
    data and the reported metrics are inflated. Split the originals instead;
    training-time augmentation happens on the fly in create_image_generators(),
    and class imbalance is handled with class weights in train_stage1().

    The originals also contain byte-identical duplicates (e.g. 'MR0029 - Copy.jpg').
    Only one copy of each image is kept, so the same picture can never sit in two
    splits or be counted twice in the test set.

    Any existing train/val/test folders in output_dir are deleted first, so files
    from an older split cannot survive into the new one.

    folder_name_map maps raw folder names -> clean class names. Point it at the
    original tumor folder if yours is named differently.
    """
    if folder_name_map is None:
        folder_name_map = {'Tumor': 'Tumor', 'Tumor (Original)': 'Tumor',
                           'Tumor(Original)': 'Tumor', 'No Tumor': 'No Tumor'}

    bad = [k for k in folder_name_map if 'augment' in k.lower()]
    if bad:
        raise ValueError(
            f"Refusing to split augmented folder(s) {bad}: augmented copies of the "
            "same scan would leak into the test set. Use the original images.")

    class_names = sorted(d for d in os.listdir(source_dir)
                         if os.path.isdir(os.path.join(source_dir, d)) and d in folder_name_map)
    found = sorted(folder_name_map[d] for d in class_names)
    if found != ['No Tumor', 'Tumor']:
        raise ValueError(
            f"Expected exactly one original 'Tumor' folder and one 'No Tumor' folder in "
            f"{source_dir}, found {sorted(os.listdir(source_dir))}. Pass folder_name_map to "
            "match your folder names (original tumor images, not the augmented ones).")

    # Collect files, dropping byte-identical duplicates (within and across classes).
    files_by_class, seen, n_dupes = {}, {}, 0
    for cls in class_names:
        cls_dir = os.path.join(source_dir, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMAGE_EXTS))
        if any(f.lower().startswith('aug_') for f in files):
            raise ValueError(
                f"'{cls_dir}' contains augmented images (aug_*.jpg). Point source_dir at "
                "the original images only.")
        kept = []
        for f in files:
            digest = _file_md5(os.path.join(cls_dir, f))
            if digest in seen:
                n_dupes += 1
                if seen[digest] != folder_name_map[cls]:
                    raise ValueError(f"'{f}' appears in both classes — check the raw data.")
                continue
            seen[digest] = folder_name_map[cls]
            kept.append(f)
        files_by_class[cls] = kept

    for split in ['train', 'val', 'test']:
        shutil.rmtree(os.path.join(output_dir, split), ignore_errors=True)
        for cls in class_names:
            os.makedirs(os.path.join(output_dir, split, folder_name_map[cls]))

    rng = np.random.default_rng(seed)
    counts = {split: {} for split in ['train', 'val', 'test']}
    for cls in class_names:
        clean_name = folder_name_map[cls]
        cls_dir = os.path.join(source_dir, cls)
        files = list(files_by_class[cls])
        rng.shuffle(files)
        n = len(files)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        split_groups = {
            'val': files[:n_val],
            'test': files[n_val:n_val + n_test],
            'train': files[n_val + n_test:],
        }
        for split_name, split_files in split_groups.items():
            counts[split_name][clean_name] = len(split_files)
            for fname in split_files:
                shutil.copy2(os.path.join(cls_dir, fname),
                             os.path.join(output_dir, split_name, clean_name, fname))

    info = {'split_version': SPLIT_VERSION, 'seed': seed,
            'duplicates_removed': n_dupes, 'counts': counts}
    with open(os.path.join(output_dir, SPLIT_INFO), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Split written to {output_dir}: {counts} ({n_dupes} duplicate images removed)")
    return info


def split_is_current(data_dir=SPLIT_DIR):
    """True if data_dir holds a split made by the current (leak-free) split_dataset."""
    try:
        with open(os.path.join(data_dir, SPLIT_INFO)) as f:
            return json.load(f).get('split_version') == SPLIT_VERSION
    except (OSError, ValueError):
        return False


def _has_originals(raw_dir):
    if not os.path.isdir(raw_dir):
        return False
    names = {d.lower().replace(' ', '') for d in os.listdir(raw_dir)}
    return 'notumor' in names and bool(names & {'tumor', 'tumor(original)'})


def prepare_stage1_data(raw_dir=RAW_DIR, data_dir=SPLIT_DIR):
    """
    Make sure data_dir holds the leak-free train/val/test split, building it if needed.
    A split without the current split_info.json (e.g. the old one made from the
    augmented folder) is thrown away and rebuilt from the original images, which are
    downloaded from Mendeley Data if they are not already in raw_dir.
    """
    if split_is_current(data_dir):
        return data_dir
    print("No leak-free split found — rebuilding it from the original images.")
    try:
        if not _has_originals(raw_dir):
            raise ValueError("originals not found locally")
        split_dataset(raw_dir, data_dir)
    except ValueError as e:
        print(f"Cannot use {raw_dir} ({e}); fetching the originals instead.")
        fresh = raw_dir.rstrip('/') + ' (Mendeley originals)'
        if not _has_originals(fresh):
            download_raw_dataset(fresh)
        split_dataset(fresh, data_dir)
    return data_dir


def create_image_generators(data_dir=SPLIT_DIR):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1, fill_mode='nearest',
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=True, seed=42,
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False,
    )
    return train_gen, val_gen, test_gen


def transform_patient(patient_data, features, categories):
    """
    Turn one patient's raw record (dict) into the single-row feature frame the
    Stage 2 model expects. Imputing and scaling are NOT done here — they live
    inside the model's own sklearn pipelines, fitted on the training split.

    categories: {'Gender': [...], 'Race': [...]} — the label-encoder classes saved
    with the model. Missing age or an unseen category becomes NaN and is imputed.
    """
    row = {}
    for col in features:
        val = patient_data.get(col)
        if col == 'Age_at_diagnosis':
            row[col] = float(val) if isinstance(val, (int, float)) else parse_age(val)
        elif col in categories:
            classes = list(categories[col])
            row[col] = classes.index(str(val)) if str(val) in classes else np.nan
        else:
            row[col] = 1 if str(val).upper() in ('MUTATED', '1', 'TRUE') else 0
    return pd.DataFrame([row], columns=features).astype(float)
