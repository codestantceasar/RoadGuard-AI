from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

from src.image_loader import CLASSES, create_dataset


CURRENT_MODEL = Path("models/roadguard_best.keras")
CANDIDATE_MODEL = Path("outputs/retraining/roadguard_candidate.keras")

CURRENT_THRESHOLDS = Path("models/thresholds.json")
CANDIDATE_THRESHOLDS = Path(
    "outputs/retraining/candidate_thresholds.json"
)

VALIDATION_CSV = Path("data/processed/validation.csv")
TEST_CSV = Path("data/processed/test.csv")

OUTPUT_DIR = Path("outputs/retraining")
REPORT_FILE = OUTPUT_DIR / "candidate_evaluation.json"

MAX_CLASS_F1_DROP = 0.05


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df[CLASSES].values.astype("int32")


def predict_model(model_path, csv_path):
    """Load a model and predict probabilities."""

    dataset = create_dataset(str(csv_path))

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    predictions = model.predict(
        dataset,
        verbose=1
    )

    if predictions.ndim != 2 or predictions.shape[1] != len(CLASSES):
        raise ValueError(
            f"Expected predictions with {len(CLASSES)} classes."
        )

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced invalid probabilities.")

    return predictions


def tune_candidate_thresholds():
    """Tune thresholds using validation data only."""

    print("\nTuning candidate thresholds on validation data...")

    y_true = load_labels(VALIDATION_CSV)

    probabilities = predict_model(
        CANDIDATE_MODEL,
        VALIDATION_CSV
    )

    if len(y_true) != len(probabilities):
        raise ValueError("Validation prediction count does not match labels.")

    thresholds = {}

    for index, class_name in enumerate(CLASSES):

        if len(np.unique(y_true[:, index])) < 2:
            thresholds[class_name] = 0.5
            print(
                f"{class_name}: insufficient validation classes; "
                "using threshold 0.5"
            )
            continue

        precision, recall, values = precision_recall_curve(
            y_true[:, index],
            probabilities[:, index]
        )

        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )

        best_index = int(np.argmax(f1_values))

        thresholds[class_name] = float(values[best_index])

        print(
            f"{class_name}: threshold={thresholds[class_name]:.4f}, "
            f"validation F1={f1_values[best_index]:.4f}"
        )

    CANDIDATE_THRESHOLDS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(CANDIDATE_THRESHOLDS, "w") as file:
        json.dump(thresholds, file, indent=4)

    return thresholds


def calculate_metrics(y_true, probabilities, thresholds):
    """Calculate metrics using per-class thresholds."""

    threshold_values = np.array(
        [thresholds[class_name] for class_name in CLASSES]
    )

    predictions = (
        probabilities >= threshold_values
    ).astype("int32")

    per_class = {}

    for index, class_name in enumerate(CLASSES):
        per_class[class_name] = {
            "precision": float(
                precision_score(
                    y_true[:, index],
                    predictions[:, index],
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true[:, index],
                    predictions[:, index],
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true[:, index],
                    predictions[:, index],
                    zero_division=0
                )
            ),
        }

    try:
        macro_roc_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                average="macro"
            )
        )
    except ValueError:
        macro_roc_auc = None

    return {
        "macro_precision": float(
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="micro",
                zero_division=0
            )
        ),
        "macro_roc_auc": macro_roc_auc,
        "per_class": per_class,
    }


def main():
    print("========== CANDIDATE MODEL EVALUATION ==========")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    required_files = [
        CURRENT_MODEL,
        CANDIDATE_MODEL,
        CURRENT_THRESHOLDS,
        VALIDATION_CSV,
        TEST_CSV,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    candidate_thresholds = tune_candidate_thresholds()

    with open(CURRENT_THRESHOLDS) as file:
        current_thresholds = json.load(file)

    for thresholds in [current_thresholds, candidate_thresholds]:
        if set(thresholds) != set(CLASSES):
            raise ValueError("Threshold file does not contain all four classes.")

        if not all(
            np.isfinite(float(value)) and 0 <= float(value) <= 1
            for value in thresholds.values()
        ):
            raise ValueError("Thresholds must be finite values between 0 and 1.")

    print("\nLoading test labels...")

    y_test = load_labels(TEST_CSV)

    print("\nEvaluating current model...")

    current_probabilities = predict_model(
        CURRENT_MODEL,
        TEST_CSV
    )

    print("\nEvaluating candidate model...")

    candidate_probabilities = predict_model(
        CANDIDATE_MODEL,
        TEST_CSV
    )

    if (
        len(y_test) != len(current_probabilities)
        or len(y_test) != len(candidate_probabilities)
    ):
        raise ValueError("Test predictions and labels have different lengths.")

    current_metrics = calculate_metrics(
        y_test,
        current_probabilities,
        current_thresholds
    )

    candidate_metrics = calculate_metrics(
        y_test,
        candidate_probabilities,
        candidate_thresholds
    )

    macro_f1_pass = (
        candidate_metrics["macro_f1"]
        >= current_metrics["macro_f1"]
    )

    class_results = {}
    class_f1_pass = True

    for class_name in CLASSES:
        old_f1 = current_metrics["per_class"][class_name]["f1"]
        new_f1 = candidate_metrics["per_class"][class_name]["f1"]

        difference = new_f1 - old_f1
        passed = difference >= -MAX_CLASS_F1_DROP

        if not passed:
            class_f1_pass = False

        class_results[class_name] = {
            "current_f1": old_f1,
            "candidate_f1": new_f1,
            "difference": difference,
            "passed": passed,
        }

    accepted = macro_f1_pass and class_f1_pass

    report = {
        "test_images": int(len(y_test)),
        "comparison_method": (
            "Both models evaluated on the same test images. "
            "Each model uses thresholds selected using validation data."
        ),
        "current_model": str(CURRENT_MODEL),
        "candidate_model": str(CANDIDATE_MODEL),
        "candidate_thresholds": candidate_thresholds,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "acceptance_criteria": {
            "candidate_macro_f1_must_not_be_lower": macro_f1_pass,
            "maximum_allowed_per_class_f1_drop": MAX_CLASS_F1_DROP,
            "all_classes_within_allowed_drop": class_f1_pass,
        },
        "class_comparisons": class_results,
        "accepted_for_promotion": accepted,
        "promotion_performed": False,
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print("\n========== RESULTS ==========")

    print("Current macro F1:", round(current_metrics["macro_f1"], 4))
    print("Candidate macro F1:", round(candidate_metrics["macro_f1"], 4))

    print("\nPer-class comparison:")

    for class_name, result in class_results.items():
        print(
            f"{class_name}: "
            f"current={result['current_f1']:.4f}, "
            f"candidate={result['candidate_f1']:.4f}, "
            f"passed={result['passed']}"
        )

    print("\nAccepted for promotion:", accepted)
    print("Promotion performed: False")
    print("Evaluation report:", REPORT_FILE)


if __name__ == "__main__":
    main()
