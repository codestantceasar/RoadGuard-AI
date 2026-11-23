import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_recall_curve
from src.image_loader import create_dataset, CLASSES

def tune_thresholds():
    print("========== VALIDATION THRESHOLD TUNING ==========")

    model = tf.keras.models.load_model(
        "models/roadguard_best.keras"
    )

    dataset = create_dataset(
        "data/processed/validation.csv"
    )

    y_true = np.concatenate([
        labels.numpy() for _, labels in dataset
    ])

    y_prob = model.predict(dataset)

    thresholds = {}

    for index, class_name in enumerate(CLASSES):
        precision, recall, threshold_values = (
            precision_recall_curve(
                y_true[:, index],
                y_prob[:, index]
            )
        )

        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (
                precision[:-1] + recall[:-1] + 1e-8
            )
        )

        best_index = np.argmax(f1_scores)
        best_threshold = float(
            threshold_values[best_index]
        )

        thresholds[class_name] = best_threshold

        print(f"\n{class_name}")
        print("Best threshold:", round(best_threshold, 4))
        print("Validation F1:", round(
            float(f1_scores[best_index]), 4
        ))

    with open("models/thresholds.json", "w") as file:
        json.dump(thresholds, file, indent=4)

    print("\nThresholds saved to models/thresholds.json")


if __name__ == "__main__":
    tune_thresholds()
