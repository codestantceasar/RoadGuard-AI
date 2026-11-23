import tensorflow as tf
from tensorflow.keras import layers, models

from src.image_loader import IMAGE_SIZE, CLASSES


def build_model():
    print("Building RoadGuard MobileNetV2 model...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially.
    base_model.trainable = False

    inputs = layers.Input(
        shape=(*IMAGE_SIZE, 3)
    )

    # MobileNetV2 expects pixel values in [-1, 1].
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(inputs)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        len(CLASSES),
        activation="sigmoid",
        name="damage_predictions"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="RoadGuard_MobileNetV2"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy",
                threshold=0.5
            )
        ]
    )

    return model


if __name__ == "__main__":
    model = build_model()

    model.summary()

    print("\nOutput classes:", CLASSES)
    print("Output shape:", model.output_shape)
    print("\nMODEL CREATED SUCCESSFULLY")
