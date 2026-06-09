"""Build the MobileNetV2 transfer-learning classifier."""

from __future__ import annotations

import tensorflow as tf

from config import CLASS_NAMES, IMAGE_SIZE, LEARNING_RATE


def build_model(
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = len(CLASS_NAMES),
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Build and compile a frozen ImageNet MobileNetV2 classifier."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_mobilenet_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
