import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.image_loader import create_dataset, CLASSES


def evaluate():
    print("========== ROADGUARD FINAL EVALUATION ==========")

    model = tf.keras.models.load_model(
        "models/roadguard_best.keras"
    )

    with open("models/thresholds.json") as file:
        thresholds = json.load(file)

    dataset = create_dataset("data/processed/test.csv")

    y_true = np.concatenate([
        labels.numpy() for _, labels in dataset
    ])

    y_prob = model.predict(dataset)

    threshold_values = np.array([
        thresholds[name] for name in CLASSES
    ])

    y_pred = (y_prob >= threshold_values).astype(int)

    print("\nCLASSIFICATION REPORT - TUNED THRESHOLDS")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        zero_division=0
    ))

    print("\nOVERALL METRICS")
    print("Macro Precision:",
          precision_score(y_true, y_pred,
                          average="macro", zero_division=0))

    print("Macro Recall:",
          recall_score(y_true, y_pred,
                       average="macro", zero_division=0))

    print("Macro F1:",
          f1_score(y_true, y_pred,
                   average="macro", zero_division=0))

    print("Micro F1:",
          f1_score(y_true, y_pred,
                   average="micro", zero_division=0))

    print("Macro ROC-AUC:",
          roc_auc_score(y_true, y_prob, average="macro"))

    print("\nTHRESHOLDS USED")
    for name, value in thresholds.items():
        print(f"{name}: {value:.4f}")

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate()
