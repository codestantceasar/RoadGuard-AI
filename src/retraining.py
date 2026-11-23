
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.image_loader import CLASSES, create_dataset


MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/retraining")

CURRENT_MODEL = MODEL_DIR / "roadguard_best.keras"
NEW_MODEL = OUTPUT_DIR / "roadguard_candidate.keras"

COMBINED_CSV = Path("data/processed/combined_train.csv")
VALIDATION_CSV = Path("data/processed/validation.csv")

EPOCHS = 5
EXPECTED_IMAGES = 8223


def validate_combined_data():
    """Validate the prepared India + Czech training dataset."""

    if not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"Combined dataset not found: {COMBINED_CSV}"
        )

    df = pd.read_csv(COMBINED_CSV)

    required_columns = ["image_path"] + CLASSES

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Combined training dataset is empty.")

    if len(df) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, found {len(df)}."
        )

    if df["image_path"].isna().any():
        raise ValueError("Missing image paths detected.")

    if df["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths detected.")

    for class_name in CLASSES:
        if not df[class_name].isin([0, 1]).all():
            raise ValueError(
                f"{class_name} must contain only 0 or 1."
            )

    missing_images = [
        path for path in df["image_path"]
        if not Path(path).is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} training images are missing. "
            f"Example: {missing_images[0]}"
        )

    print("\nCombined dataset validation passed.")
    print("Total images:", len(df))

    print("\nClass-positive image counts:")
    for class_name in CLASSES:
        print(
            f"{class_name}: {int(df[class_name].sum())}"
        )

    return df


def prepare_training_data():
    """Prepare the validated combined CSV for TensorFlow."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = validate_combined_data()

    training_df = df[["image_path"] + CLASSES].copy()

    training_csv = OUTPUT_DIR / "retraining_data.csv"

    training_df.to_csv(training_csv, index=False)

    print("\nTraining data prepared:", training_csv)
    print("Training images:", len(training_df))

    return training_csv


def retrain():

    print("========== ROADGUARD RETRAINING ==========")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CURRENT_MODEL.exists():
        raise FileNotFoundError(
            f"Current model not found: {CURRENT_MODEL}"
        )

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_CSV}"
        )

    # Step 1: Validate and prepare combined training data.
    training_csv = prepare_training_data()

    # Step 2: Load training and validation datasets.
    train_dataset = create_dataset(
        str(training_csv),
        training=True
    )

    validation_dataset = create_dataset(
        str(VALIDATION_CSV)
    )

    # Step 3: Load the existing model.
    model = tf.keras.models.load_model(
        CURRENT_MODEL
    )

    # Step 4: Configure callbacks.
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

    # Step 5: Train the separate candidate model.
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Step 6: Save candidate separately.
    model.save(NEW_MODEL)

    # Step 7: Save training history.
    history_df = pd.DataFrame(history.history)

    history_path = (
        OUTPUT_DIR / "candidate_training_history.csv"
    )

    history_df.to_csv(history_path, index=False)

    print("\n========== RETRAINING COMPLETED ==========")
    print("Candidate model:", NEW_MODEL)
    print("Training history:", history_path)
    print("Original model preserved:", CURRENT_MODEL)

    return history


if __name__ == "__main__":
    retrain()
