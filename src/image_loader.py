
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_SIZE = (160, 160)
CLASSES = ["D00", "D10", "D20", "D40"]
BATCH_SIZE = 8


def resolve_image_path(raw_path):

    path = Path(str(raw_path)).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return str(path)


def load_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


def create_dataset(csv_path, training=False):

    csv_path = Path(csv_path)

    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = ["image_path"] + CLASSES

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset CSV is missing columns: {missing}"
        )

    paths = [
        resolve_image_path(path)
        for path in df["image_path"]
    ]

    labels = df[CLASSES].values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    def process_image(path, label):

        image = load_image(path)

        return image, label

    dataset = dataset.map(
        process_image,
        num_parallel_calls=2
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=min(len(df), 1000),
            seed=42
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(1)

    return dataset


if __name__ == "__main__":

    print("Testing RoadGuard image loader...")

    train_csv = (
        PROJECT_ROOT / "data/processed/combined_train.csv"
    )

    train_dataset = create_dataset(
        str(train_csv),
        training=True
    )

    for images, labels in train_dataset.take(1):

        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Image data type:", images.dtype)

        print(
            "Minimum pixel value:",
            float(tf.reduce_min(images))
        )

        print(
            "Maximum pixel value:",
            float(tf.reduce_max(images))
        )

    print("IMAGE LOADER TEST PASSED")
