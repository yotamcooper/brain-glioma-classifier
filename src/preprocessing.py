import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE   = (224, 224)   # DenseNet-121 requires exactly 224x224
BATCH_SIZE = 32           # images processed per gradient update


# ════════════════════════════════════════════════════════════
#  STAGE 2  —  LGG vs GBM Grading
# ════════════════════════════════════════════════════════════

def load_uci_glioma(path):
    df = pd.read_csv(path)
    print(f"Patients: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Grade distribution:\n{df['Grade'].value_counts()}\n")
    return df


def encode_labels(df):
    le = LabelEncoder()
    y = le.fit_transform(df['Grade'])
    X = df.drop(columns=['Grade', 'Case_ID', 'Project', 'Primary_Diagnosis'])

    X = prepare_features(X, df)          

    label_encoders = {}
    for col in ['Gender', 'Race']:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))   
        label_encoders[col] = le_col

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  

    scaler = StandardScaler()
    X['Age_at_diagnosis'] = scaler.fit_transform(X[['Age_at_diagnosis']])  

    return X, y, X.columns.tolist(), scaler, imputer, label_encoders


def get_mutation_columns(df):
    non_mutation = ['Case_ID', 'Project', 'Primary_Diagnosis',
                    'Grade', 'Gender', 'Race', 'Age_at_diagnosis']
    return [c for c in df.columns if c not in non_mutation]

def parse_age(age_str):
        if pd.isna(age_str) or not str(age_str).split()[0].isdigit():
            return np.nan
        parts = str(age_str).split()
        years = int(parts[0])
        days = int(parts[2]) if len(parts) >= 4 else 0
        return round(years + days / 365, 2)

def prepare_features(X, df):
    X = X.copy()
    X['Age_at_diagnosis'] = X['Age_at_diagnosis'].apply(parse_age)
    mutation_cols = get_mutation_columns(df)
    X[mutation_cols] = X[mutation_cols].replace({'MUTATED': 1, 'NOT_MUTATED': 0}).infer_objects(copy=False)
    return X

def transform_patient(patient_data, features, scaler, imputer, label_encoders, df_ref):
    #used for predicting patient tumor grade
    row = pd.DataFrame([patient_data])

    row = prepare_features(row, df_ref)  

    for col in ['Gender', 'Race']:
        row[col] = label_encoders[col].transform(row[col].astype(str))  

    row = row.reindex(columns=features, fill_value=0)
    row_array = imputer.transform(row)                                   
    row_df = pd.DataFrame(row_array, columns=features)
    row_df['Age_at_diagnosis'] = scaler.transform(row_df[['Age_at_diagnosis']])  
    return row_df.values


# ════════════════════════════════════════════════════════════
#  STAGE 1  —  Tumor Detector
# ════════════════════════════════════════════════════════════

# func not needed if using my secontized data if donloading raw data from website use this function
def split_dataset(
    source_dir="data/Brain Tumor MRI Dataset Raw",
    output_dir="data/Brain Tumor MRI Dataset",
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    np.random.seed(seed)

    folder_name_map = {
        "Tumor(Augmented)": "Tumor",
        "No Tumor":         "No Tumor",
    }

    class_names = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
        and d in folder_name_map
    ]
    print(f"Found classes: {class_names}")

    for split in ["train", "val", "test"]:
        for cls in class_names:
            clean_name = folder_name_map.get(cls, cls)
            os.makedirs(os.path.join(output_dir, split, clean_name), exist_ok=True)

    for cls in class_names:
        clean_name = folder_name_map.get(cls, cls)
        cls_dir = os.path.join(source_dir, cls)
        files   = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        np.random.shuffle(files)

        n      = len(files)
        n_val  = int(n * val_ratio)
        n_test = int(n * test_ratio)

        split_groups = {
            "val":   files[:n_val],
            "test":  files[n_val : n_val + n_test],
            "train": files[n_val + n_test:],
        }

        for split_name, split_files in split_groups.items():
            for fname in split_files:
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(output_dir, split_name, clean_name, fname)
                shutil.copy2(src, dst)

        print(
            f"  {clean_name:>10}: "
            f"{len(split_groups['train']):,} train | "
            f"{len(split_groups['val']):,} val | "
            f"{len(split_groups['test']):,} test"
        )

    print(f"\nDone. Split dataset saved to: {output_dir}")


def create_image_generators(data_dir="data/Brain Tumor MRI Dataset"):
    """
    TRAIN generator applies:
        rescale=1./255       pixels 0-255 -> 0.0-1.0 (always needed)
        rotation_range=15    rotate up to ±15° each batch
        width_shift_range    shift left/right up to 10%
        height_shift_range   shift up/down up to 10%
        horizontal_flip      randomly mirror left-right
        zoom_range=0.1       zoom in/out up to 10%
        fill_mode=nearest    fill exposed pixels with nearest edge value

    rotation/flip/zoom made because the dataset README confirms
    these were used to generate the augmented copies on disk —
    the training augmentation is consistent with that.

    VAL + TEST generators: rescale ONLY, no augmentation.
    because model.predict(test_gen) processes images in order, and test_gen.classes gives labels
    in that same order. Shuffling would break that match.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_gen = train_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,   # resizes every image to 224x224
        batch_size=BATCH_SIZE,
        class_mode="binary",    # labels: 0 = No Tumor, 1 = Tumor
        shuffle=True,
        seed=42,
    )

    val_gen = eval_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print(f"Train : {train_gen.samples:,} images")
    print(f"Val   : {val_gen.samples:,} images")
    print(f"Test  : {test_gen.samples:,} images")
    print(f"Class mapping: {train_gen.class_indices}")
    return train_gen, val_gen, test_gen