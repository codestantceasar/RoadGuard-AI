import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from io import BytesIO

CLASSES = ["D00", "D10", "D20", "D40"]
IMAGE_SIZE = (160, 160)

MODEL_PATH = Path("models/roadguard_best.keras")
THRESHOLD_PATH = Path("models/thresholds.json")

model = None
thresholds = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, thresholds

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(THRESHOLD_PATH) as file:
        thresholds = json.load(file)

    print("RoadGuard model loaded successfully.")

    yield

    model = None
    thresholds = None


app = FastAPI(
    title="RoadGuard AI API",
    description="Road damage image classification API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "project": "RoadGuard AI",
        "message": "Road damage detection API is running",
        "classes": CLASSES
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None or thresholds is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG or PNG image."
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB limit."
            )

        with Image.open(BytesIO(contents)) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image."
        )

    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    results = {}

    for index, class_name in enumerate(CLASSES):
        probability = float(probabilities[index])
        threshold = float(thresholds[class_name])

        results[class_name] = {
            "probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "detected": probability >= threshold
        }

    return {
        "filename": file.filename,
        "predictions": results
    }
