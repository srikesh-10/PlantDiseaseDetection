"""Shared utilities for training and application workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DISEASE_INFO_PATH,
    DOCS_DIR,
    LABELS_PATH,
    LOGS_DIR,
    MODELS_DIR,
    SAVED_MODELS_DIR,
    TRAINING_ACCURACY_PATH,
    TRAINING_LOG_PATH,
    TRAINING_LOSS_PATH,
    TRAINING_METRICS_PATH,
)

LOGGER = logging.getLogger("training")


def load_disease_knowledgebase(path: Path = DISEASE_INFO_PATH) -> dict[str, Any]:
    """Load disease information from a JSON knowledge-base file."""
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge-base file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_project_directories() -> None:
    """Create directories used by model training and reporting."""
    for path in (MODELS_DIR, SAVED_MODELS_DIR, DOCS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_training_logging() -> Path:
    """Configure console and file logging for a training run."""
    ensure_project_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(TRAINING_LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )
    return TRAINING_LOG_PATH


def save_labels(class_names: list[str], path: Path = LABELS_PATH) -> None:
    """Save model output indices and their class labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = {str(index): name for index, name in enumerate(class_names)}
    path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Class labels saved to %s", path)


def save_training_plots(history: tf.keras.callbacks.History) -> None:
    """Save accuracy and loss curves from a Keras training history."""
    plots = (
        ("accuracy", "val_accuracy", "Accuracy", TRAINING_ACCURACY_PATH),
        ("loss", "val_loss", "Loss", TRAINING_LOSS_PATH),
    )
    for train_key, val_key, title, path in plots:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get(train_key, []), label=f"Training {title}")
        plt.plot(history.history.get(val_key, []), label=f"Validation {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"{title} vs Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        LOGGER.info("Training plot saved to %s", path)


def evaluate_and_save_results(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
) -> dict[str, Any]:
    """Evaluate a model and save metrics, report, and confusion matrix."""
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])
    probabilities = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "test_samples": int(len(y_true)),
    }
    TRAINING_METRICS_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    CLASSIFICATION_REPORT_PATH.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row, column in np.ndindex(matrix.shape):
        plt.text(
            column,
            row,
            str(matrix[row, column]),
            ha="center",
            va="center",
            color="white" if matrix[row, column] > threshold else "black",
        )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    LOGGER.info("Evaluation results: %s", metrics)
    LOGGER.info("Evaluation artifacts saved under %s", DOCS_DIR)
    return metrics
