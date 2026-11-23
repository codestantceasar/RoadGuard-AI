
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from src.image_loader import IMAGE_SIZE, CLASSES


MODEL_PATH = Path("models/roadguard_best.keras")
THRESHOLD_PATH = Path("models/thresholds.json")


def load_prediction_resources():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        thresholds = json.load(file)

    return model, thresholds


def predict_road_damage(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError("The provided path is not a file.")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

    except Exception as error:
        raise ValueError(
            "Could not process the supplied image."
        ) from error

    image_array = np.expand_dims(image_array, axis=0)

    model, thresholds = load_prediction_resources()

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predictions = {}

    for index, class_name in enumerate(CLASSES):
        probability = float(probabilities[index])

        threshold = float(thresholds[class_name])

        predictions[class_name] = {
            "probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "detected": probability >= threshold
        }

    return {
        "image": image_path.name,
        "predictions": predictions
    }


if __name__ == "__main__":
    sample_image = (
        "data/raw/RDD2022_India/India/train/images/"
        "India_001000.jpg"
    )

    result = predict_road_damage(sample_image)

    print("\nROADGUARD PREDICTION")
    print(json.dumps(result, indent=4))
