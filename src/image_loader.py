from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


IMAGE_SIZE = (160, 160)
CLASSES = ["D00", "D10", "D20", "D40"]
BATCH_SIZE = 8


def load_image(image_path):
    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32)

    return image


def create_dataset(csv_path, training=False):
    df = pd.read_csv(csv_path)

    paths = df["image_path"].astype(str).values

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

    train_dataset = create_dataset(
        "data/processed/train.csv",
        training=True
    )

    for images, labels in train_dataset.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Image data type:", images.dtype)
        print("Minimum pixel value:", float(tf.reduce_min(images)))
        print("Maximum pixel value:", float(tf.reduce_max(images)))

    print("IMAGE LOADER TEST PASSED")
