
from pathlib import Path
import sqlite3

import pandas as pd
import tensorflow as tf

from src.image_loader import CLASSES, create_dataset


# ==================================================
# PROJECT PATHS AND SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs/retraining"

UPLOAD_DIR = PROJECT_ROOT / "data/uploads"
IMAGE_UPLOAD_DIR = UPLOAD_DIR / "images"
LABEL_DIR = UPLOAD_DIR / "labels"
DATABASE_PATH = UPLOAD_DIR / "roadguard_uploads.db"

# Production model is read-only.
CURRENT_MODEL = MODEL_DIR / "roadguard_best.keras"

# New candidate is saved separately.
NEW_MODEL = OUTPUT_DIR / "roadguard_candidate.keras"

COMBINED_CSV = PROJECT_ROOT / "data/processed/combined_train.csv"

EXTERNAL_TRAIN_CSV = (
    PROJECT_ROOT
    / "data/external/roaddamagevision/processed/train.csv"
)

VALIDATION_CSV = PROJECT_ROOT / "data/processed/validation.csv"

EXPECTED_ORIGINAL_IMAGES = 8223
EPOCHS = 5


# ==================================================
# IMAGE PATH NORMALIZATION
# ==================================================

def normalize_image_path(raw_path):
    """
    Resolve image paths across WSL and Docker.

    Supports:
    - Relative paths such as data/raw/image.jpg
    - Absolute paths inside the current project
    - Old absolute WSL paths containing a data/ directory
    """

    raw_path = str(raw_path).strip()

    if not raw_path:
        raise ValueError("An empty image path was encountered.")

    path = Path(raw_path).expanduser()

    # Relative paths are resolved from the project root.
    if not path.is_absolute():
        resolved = (PROJECT_ROOT / path).resolve()

    else:
        resolved = path.resolve()

        # Absolute paths already inside this environment.
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:

            # Translate old host paths using their data/ suffix.
            parts = path.parts

            data_index = next(
                (
                    index
                    for index, part in enumerate(parts)
                    if part == "data"
                ),
                None
            )

            if data_index is None:
                raise ValueError(
                    f"Cannot map external image path into project: "
                    f"{raw_path}"
                )

            relative_parts = parts[data_index:]

            resolved = (
                PROJECT_ROOT.joinpath(*relative_parts).resolve()
            )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Image does not exist in this environment: {resolved}"
        )

    return str(resolved)


# ==================================================
# GENERAL DATA VALIDATION
# ==================================================

def validate_labels(df, dataset_name):

    required_columns = ["image_path"] + CLASSES

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            f"{dataset_name} contains no images."
        )

    df = df[required_columns].copy()

    if df.isna().any().any():
        raise ValueError(
            f"{dataset_name} contains missing values."
        )

    # Normalize paths before duplicate and file checks.
    normalized_paths = []

    for raw_path in df["image_path"]:
        normalized_paths.append(
            normalize_image_path(raw_path)
        )

    df["image_path"] = normalized_paths

    if df["image_path"].duplicated().any():
        raise ValueError(
            f"{dataset_name} contains duplicate image paths."
        )

    # Validate multi-label values.
    for label in CLASSES:

        values = pd.to_numeric(
            df[label],
            errors="coerce"
        )

        if values.isna().any():
            raise ValueError(
                f"{dataset_name} has non-numeric values in {label}."
            )

        if not values.isin([0, 1]).all():
            raise ValueError(
                f"{dataset_name} has invalid values in {label}."
            )

        df[label] = values.astype("int32")

    return df


# ==================================================
# ORIGINAL DATASET
# ==================================================

def validate_original_data():

    if not COMBINED_CSV.is_file():
        raise FileNotFoundError(
            f"Original dataset CSV not found: {COMBINED_CSV}"
        )

    df = pd.read_csv(COMBINED_CSV)

    if len(df) != EXPECTED_ORIGINAL_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_ORIGINAL_IMAGES} original images, "
            f"found {len(df)}."
        )

    df = validate_labels(
        df,
        "Original training dataset"
    )

    print(
        f"Original training data validated: {len(df)} images."
    )

    return df


# ==================================================
# EXTERNAL DATASET
# ==================================================

def validate_external_data():

    if not EXTERNAL_TRAIN_CSV.is_file():
        raise FileNotFoundError(
            f"RoadDamageVision CSV not found: {EXTERNAL_TRAIN_CSV}"
        )

    df = pd.read_csv(EXTERNAL_TRAIN_CSV)

    df = validate_labels(
        df,
        "RoadDamageVision training dataset"
    )

    print(
        f"RoadDamageVision validated: {len(df)} images."
    )

    for label in CLASSES:
        print(f"  {label}: {int(df[label].sum())} positive images")

    return df


# ==================================================
# UPLOAD DATABASE
# ==================================================

def initialize_upload_database():

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_images (
                filename TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                D00 INTEGER NOT NULL,
                D10 INTEGER NOT NULL,
                D20 INTEGER NOT NULL,
                D40 INTEGER NOT NULL,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_uploads_to_database(df):

    initialize_upload_database()

    records = []

    for _, row in df.iterrows():

        records.append((
            row["filename"],
            row["image_path"],
            *[int(row[label]) for label in CLASSES]
        ))

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.executemany("""
            INSERT OR REPLACE INTO uploaded_images
            (filename, image_path, D00, D10, D20, D40)
            VALUES (?, ?, ?, ?, ?, ?)
        """, records)

    print(f"Saved {len(records)} dashboard upload records.")


# ==================================================
# DASHBOARD UPLOAD VALIDATION
# ==================================================

def validate_uploads():

    labels_file = LABEL_DIR / "labels.csv"

    if not labels_file.is_file():

        existing_images = (
            list(IMAGE_UPLOAD_DIR.iterdir())
            if IMAGE_UPLOAD_DIR.exists()
            else []
        )

        if existing_images:
            raise FileNotFoundError(
                "Dashboard images exist, but labels.csv is missing."
            )

        print("No dashboard uploads found.")

        return pd.DataFrame(
            columns=["image_path"] + CLASSES
        )

    if not IMAGE_UPLOAD_DIR.is_dir():
        raise FileNotFoundError(
            "Dashboard labels exist, but the image directory is missing."
        )

    df = pd.read_csv(labels_file)

    required_columns = ["filename"] + CLASSES

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dashboard labels CSV is missing columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "Dashboard labels CSV contains no records."
        )

    df = df[required_columns].copy()

    df["filename"] = (
        df["filename"].astype(str).str.strip()
    )

    if (df["filename"] == "").any():
        raise ValueError("Empty dashboard filenames found.")

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate dashboard filenames found.")

    for filename in df["filename"]:

        path = Path(filename)

        if (
            path.name != filename
            or filename in [".", ".."]
            or path.suffix.lower() not in [".jpg", ".jpeg"]
        ):
            raise ValueError(
                f"Invalid dashboard filename: {filename}"
            )

    image_files = {
        path.name
        for path in IMAGE_UPLOAD_DIR.iterdir()
        if path.is_file()
    }

    csv_files = set(df["filename"])

    if image_files != csv_files:
        raise ValueError(
            "Dashboard images and labels do not match. "
            f"Images without labels: {sorted(image_files - csv_files)}. "
            f"Labels without images: {sorted(csv_files - image_files)}."
        )

    for label in CLASSES:

        values = pd.to_numeric(
            df[label],
            errors="coerce"
        )

        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(
                f"Invalid dashboard labels in {label}."
            )

        df[label] = values.astype("int32")

    image_paths = []

    for filename in df["filename"]:

        image_path = IMAGE_UPLOAD_DIR / filename

        try:

            image_bytes = tf.io.read_file(str(image_path))

            image = tf.io.decode_jpeg(
                image_bytes,
                channels=3
            )

            if (
                image.shape[0] is None
                or image.shape[1] is None
                or image.shape[0] == 0
                or image.shape[1] == 0
            ):
                raise ValueError("Invalid image dimensions.")

        except Exception as error:

            raise ValueError(
                f"Cannot decode dashboard image {filename}: {error}"
            ) from error

        image_paths.append(str(image_path.resolve()))

    df["image_path"] = image_paths

    save_uploads_to_database(df)

    print(
        f"Dashboard uploads validated: {len(df)} images."
    )

    return df[["image_path"] + CLASSES].copy()


# ==================================================
# PREPARE TRAINING DATA
# ==================================================

def prepare_training_data():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nValidating original dataset...")
    original_df = validate_original_data()

    print("\nValidating RoadDamageVision...")
    external_df = validate_external_data()

    print("\nChecking dashboard uploads...")
    uploaded_df = validate_uploads()

    combined_df = pd.concat(
        [
            original_df,
            external_df,
            uploaded_df
        ],
        ignore_index=True
    )

    if combined_df["image_path"].duplicated().any():

        duplicates = combined_df.loc[
            combined_df["image_path"].duplicated(),
            "image_path"
        ].tolist()

        raise ValueError(
            "Duplicate image paths found across datasets. "
            f"First duplicate: {duplicates[0]}"
        )

    training_csv = OUTPUT_DIR / "retraining_data.csv"

    combined_df.to_csv(
        training_csv,
        index=False
    )

    print("\n========== TRAINING DATA SUMMARY ==========")

    print(f"Original images: {len(original_df)}")
    print(f"RoadDamageVision images: {len(external_df)}")
    print(f"Dashboard uploads: {len(uploaded_df)}")
    print(f"TOTAL TRAINING IMAGES: {len(combined_df)}")

    print("\nClass distribution:")

    for label in CLASSES:
        print(f"{label}: {int(combined_df[label].sum())}")

    print(f"\nTraining CSV saved: {training_csv}")

    return training_csv


# ==================================================
# RETRAIN CANDIDATE MODEL
# ==================================================

def retrain():

    print("\n========== ROADGUARD AI RETRAINING ==========")

    if not CURRENT_MODEL.is_file():
        raise FileNotFoundError(
            f"Production model not found: {CURRENT_MODEL}"
        )

    if not VALIDATION_CSV.is_file():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_CSV}"
        )

    training_csv = prepare_training_data()

    validation_df = validate_labels(
        pd.read_csv(VALIDATION_CSV),
        "Validation dataset"
    )

    print(
        f"\nValidation dataset verified: {len(validation_df)} images."
    )

    print("\nPreparing TensorFlow datasets...")

    train_dataset = create_dataset(
        str(training_csv),
        training=True
    )

    validation_dataset = create_dataset(
        str(VALIDATION_CSV)
    )

    print("\nLoading existing MobileNetV2 model...")

    model = tf.keras.models.load_model(
        CURRENT_MODEL
    )

    print("Existing model loaded successfully.")

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7
        )
    ]

    print("\n========== STARTING CANDIDATE TRAINING ==========")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    NEW_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(NEW_MODEL)

    history_path = OUTPUT_DIR / "candidate_training_history.csv"

    pd.DataFrame(
        history.history
    ).to_csv(
        history_path,
        index=False
    )

    print("\n========== RETRAINING COMPLETED ==========")

    print(f"Candidate model: {NEW_MODEL}")
    print(f"Training history: {history_path}")

    print("\nProduction model remains unchanged:")
    print(CURRENT_MODEL)

    return history


if __name__ == "__main__":
    retrain()
