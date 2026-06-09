"""End-to-end MobileNetV2 training orchestration."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATASET_DIR,
    EPOCHS,
    FINAL_MODEL_PATH,
    IMAGE_SIZE,
    RANDOM_SEED,
)
from src.model_builder import build_model
from src.utils import (
    configure_training_logging,
    ensure_project_directories,
    evaluate_and_save_results,
    save_labels,
    save_training_plots,
)

LOGGER = logging.getLogger("training")


class EpochLoggingCallback(tf.keras.callbacks.Callback):
    """Write concise epoch progress and metrics to the training log."""

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Log the start of an epoch."""
        LOGGER.info("Starting epoch %d", epoch + 1)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Log metrics produced at the end of an epoch."""
        metrics = {key: round(float(value), 5) for key, value in (logs or {}).items()}
        LOGGER.info("Completed epoch %d: %s", epoch + 1, metrics)


def _validate_dataset(dataset_dir: Path) -> tuple[Path, Path, Path]:
    """Validate required train, validation, and test split directories."""
    split_paths = tuple(dataset_dir / name for name in ("train", "val", "test"))
    missing = [str(path) for path in split_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset split directories: " + ", ".join(missing))
    return split_paths


def _load_datasets(
    dataset_dir: Path,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load all split directories as optimized TensorFlow datasets."""
    train_path, val_path, test_path = _validate_dataset(dataset_dir)
    common = {
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "label_mode": "int",
    }
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path, shuffle=True, seed=RANDOM_SEED, **common
    )
    class_names = list(train_dataset.class_names)
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_path, shuffle=False, class_names=class_names, **common
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_path, shuffle=False, class_names=class_names, **common
    )
    prefetch = tf.data.AUTOTUNE
    return (
        train_dataset.prefetch(prefetch),
        val_dataset.prefetch(prefetch),
        test_dataset.prefetch(prefetch),
        class_names,
    )


def _calculate_class_weights(dataset_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Calculate balanced class weights from training directory counts."""
    counts = Counter(
        {
            index: sum(1 for path in (dataset_dir / "train" / name).iterdir() if path.is_file())
            for index, name in enumerate(class_names)
        }
    )
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Every training class must contain images. Counts: {dict(counts)}")

    labels = [index for index, count in counts.items() for _ in range(count)]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.asarray(labels),
    )
    result = {index: float(weight) for index, weight in enumerate(weights)}
    LOGGER.info("Training class counts: %s", dict(counts))
    LOGGER.info("Calculated class weights: %s", result)
    return result


def _create_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for checkpointing, stopping, LR reduction, and logging."""
    LOGGER.info("Best-model checkpoints will be saved to %s", BEST_MODEL_PATH)
    return [
        EpochLoggingCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
        ),
    ]


def train_model(
    dataset_dir: Path = DATASET_DIR,
    model_path: Path = FINAL_MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tf.keras.Model:
    """Train, save, and evaluate the MobileNetV2 plant disease classifier."""
    configure_training_logging()
    ensure_project_directories()
    LOGGER.info("Training started: dataset=%s, epochs=%d, batch_size=%d", dataset_dir, epochs, batch_size)

    try:
        train_dataset, val_dataset, test_dataset, class_names = _load_datasets(
            dataset_dir, batch_size
        )
        if len(class_names) != 6:
            raise ValueError(f"Expected 6 classes in dataset/train, found {len(class_names)}.")

        save_labels(class_names)
        class_weights = _calculate_class_weights(dataset_dir, class_names)
        model = build_model(num_classes=len(class_names))
        model.summary(print_fn=lambda line: LOGGER.info(line))

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=_create_callbacks(),
        )
        LOGGER.info("Best model saved to %s", BEST_MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        LOGGER.info("Final model saved to %s", model_path)

        save_training_plots(history)
        evaluate_and_save_results(model, test_dataset, class_names)
        LOGGER.info("Training completed successfully")
        return model
    except Exception:
        LOGGER.exception("Training failed")
        raise
