import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_uci_glioma(path):
    return pd.read_csv(path)


def get_mutation_columns(df):
    non_mutation = ['Case_ID', 'Project', 'Primary_Diagnosis', 'Grade', 'Gender', 'Race', 'Age_at_diagnosis']
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

    return X, y, X.columns.tolist(), label_encoders


def split_dataset(source_dir='data/Brain Tumor MRI Dataset Raw',
                  output_dir='data/Brain Tumor MRI Dataset',
                  val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
    folder_name_map = {'Tumor(Augmented)': 'Tumor', 'No Tumor': 'No Tumor'}
    class_names = [d for d in os.listdir(source_dir)
                   if os.path.isdir(os.path.join(source_dir, d)) and d in folder_name_map]

    for split in ['train', 'val', 'test']:
        for cls in class_names:
            os.makedirs(os.path.join(output_dir, split, folder_name_map.get(cls, cls)), exist_ok=True)

    for cls in class_names:
        clean_name = folder_name_map.get(cls, cls)
        cls_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(files)
        n = len(files)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        split_groups = {
            'val': files[:n_val],
            'test': files[n_val:n_val + n_test],
            'train': files[n_val + n_test:],
        }
        for split_name, split_files in split_groups.items():
            for fname in split_files:
                shutil.copy2(os.path.join(cls_dir, fname),
                             os.path.join(output_dir, split_name, clean_name, fname))


def create_image_generators(data_dir='data/Brain Tumor MRI Dataset'):
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

def transform_patient(patient_data, features, scaler, imputer, label_encoders, df_train):
# transform raw tablulr data into a digestable form
    row = {}
    mutation_cols = get_mutation_columns(df_train)

    for col in features:
        if col in mutation_cols:
            val = patient_data.get(col, 'NOT_MUTATED')
            row[col] = 1 if str(val).upper() == 'MUTATED' else 0
        elif col == 'Age_at_diagnosis':
            row[col] = parse_age(str(patient_data.get(col, np.nan)))
        elif col in label_encoders:
            raw = str(patient_data.get(col, ''))
            le = label_encoders[col]
            row[col] = le.transform([raw])[0] if raw in le.classes_ else 0
        else:
            row[col] = patient_data.get(col, 0)

    X = pd.DataFrame([row])[features]
    X = prepare_features(X, df_train)
    X = pd.DataFrame(imputer.transform(X), columns=features)
    X = pd.DataFrame(scaler.transform(X), columns=features)
    return X