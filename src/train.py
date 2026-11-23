
from pathlib import Path
import pandas as pd
import tensorflow as tf

from src.image_loader import create_dataset
from src.model import build_model


MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

EPOCHS = 15


def train():
    print("========== ROADGUARD MODEL TRAINING ==========")

    train_dataset = create_dataset(
        "data/processed/train.csv",
        training=True
    )

    validation_dataset = create_dataset(
        "data/processed/validation.csv"
    )

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "roadguard_best.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save training history
    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    print("\nTraining history saved to outputs/training_history.csv")

    # Save final model
    model.save(MODEL_DIR / "roadguard_final.keras")

    print("\nTRAINING COMPLETED")
    print("Best model: models/roadguard_best.keras")
    print("Final model: models/roadguard_final.keras")

    return history


if __name__ == "__main__":
    train()
