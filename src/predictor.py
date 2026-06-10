"""Reusable single-image plant disease prediction engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from config import (
    BEST_MODEL_PATH,
    DATABASE_PATH,
    DISEASE_INFO_PATH,
    IMAGE_SIZE,
    LABELS_PATH,
    PREDICTION_LOG_PATH,
)
from database import initialize_database, save_prediction
from src.disease_info import DiseaseInfoManager

LOGGER = logging.getLogger("prediction")


def configure_prediction_logging() -> Path:
    """Configure prediction logging without replacing other application loggers."""
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    resolved_log_path = PREDICTION_LOG_PATH.resolve()
    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved_log_path
        for handler in LOGGER.handlers
    )
    if not has_file_handler:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(PREDICTION_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
    return PREDICTION_LOG_PATH


def load_model(model_path: Path = BEST_MODEL_PATH) -> tf.keras.Model:
    """Load a trained Keras model, raising a clear error when unavailable."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {model_path}")
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load trained model {model_path}: {exc}") from exc


def load_labels(labels_path: Path = LABELS_PATH) -> list[str]:
    """Load labels in model output-index order from JSON."""
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {labels_path}")
    try:
        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(raw_labels, dict):
            labels = [raw_labels[str(index)] for index in range(len(raw_labels))]
        elif isinstance(raw_labels, list):
            labels = raw_labels
        else:
            raise ValueError("Labels JSON must contain an index mapping or list.")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unable to load labels from {labels_path}: {exc}") from exc
    if not labels or not all(isinstance(label, str) and label for label in labels):
        raise ValueError(f"Labels file contains invalid class names: {labels_path}")
    return labels


def preprocess_image(
    image_path: Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    apply_mobilenet_preprocessing: bool = True,
) -> np.ndarray:
    """Load one image as a resized RGB batch ready for prediction."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            rgb_image = ImageOps.exif_transpose(image).convert("RGB")
            resized_image = rgb_image.resize(image_size, Image.Resampling.LANCZOS)
            image_array = np.asarray(resized_image, dtype=np.float32)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unable to read image {image_path}: {exc}") from exc

    batch = np.expand_dims(image_array, axis=0)
    if apply_mobilenet_preprocessing:
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return np.asarray(batch, dtype=np.float32)


class PlantDiseasePredictor:
    """Load model assets, predict one image, and persist prediction history."""

    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        labels_path: Path = LABELS_PATH,
        database_path: Path = DATABASE_PATH,
        disease_info_path: Path = DISEASE_INFO_PATH,
        model_contains_preprocessing: bool = True,
    ) -> None:
        """Initialize prediction assets and history storage.

        Phase 3 models contain MobileNetV2 ``preprocess_input`` in their saved
        graph. Set ``model_contains_preprocessing=False`` only for a model that
        expects externally normalized input.
        """
        configure_prediction_logging()
        self.model_path = model_path
        self.labels_path = labels_path
        self.database_path = database_path
        self.model_contains_preprocessing = model_contains_preprocessing

        LOGGER.info("Loading prediction model from %s", model_path)
        self.model = load_model(model_path)
        self.labels = load_labels(labels_path)
        output_classes = int(self.model.output_shape[-1])
        if output_classes != len(self.labels):
            raise ValueError(
                f"Model outputs {output_classes} classes but labels contain "
                f"{len(self.labels)} entries."
            )
        initialize_database(database_path)

        # Load the disease knowledge base (graceful degradation on failure).
        self.disease_info_manager: DiseaseInfoManager | None = None
        try:
            self.disease_info_manager = DiseaseInfoManager(disease_info_path)
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning(
                "Disease knowledge base unavailable — predictions will "
                "not include disease details: %s",
                exc,
            )

        LOGGER.info("Prediction engine initialized with %d classes", len(self.labels))

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Predict one image and return percentage confidence values."""
        path = Path(image_path).expanduser().resolve()
        LOGGER.info("Starting prediction for %s", path)
        try:
            image_batch = preprocess_image(
                path,
                apply_mobilenet_preprocessing=not self.model_contains_preprocessing,
            )
            raw_predictions = self.model.predict(image_batch, verbose=0)
            probabilities = np.asarray(raw_predictions, dtype=np.float32)
            if probabilities.shape != (1, len(self.labels)):
                raise ValueError(
                    f"Unexpected model prediction shape: {probabilities.shape}"
                )

            percentages = probabilities[0] * 100.0
            predicted_index = int(np.argmax(percentages))
            predicted_class = self.labels[predicted_index]
            result: dict[str, Any] = {
                "predicted_class": predicted_class,
                "confidence": round(float(percentages[predicted_index]), 2),
                "all_probabilities": {
                    label: round(float(percentages[index]), 2)
                    for index, label in enumerate(self.labels)
                },
            }

            # Attach disease knowledge-base information.
            if self.disease_info_manager is not None:
                disease_details = self.disease_info_manager.get_disease_info(
                    predicted_class,
                )
                result["symptoms"] = disease_details["symptoms"]
                result["treatment"] = disease_details["treatment"]
                result["prevention"] = disease_details["prevention"]
            else:
                result["symptoms"] = ["Knowledge base not available."]
                result["treatment"] = ["Knowledge base not available."]
                result["prevention"] = ["Knowledge base not available."]
            prediction_id = save_prediction(
                {
                    "image_path": str(path),
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                },
                self.database_path,
            )
            LOGGER.info(
                "Prediction saved: id=%d, class=%s, confidence=%.2f%%",
                prediction_id,
                result["predicted_class"],
                result["confidence"],
            )
            return result
        except Exception:
            LOGGER.exception("Prediction failed for %s", path)
            raise


def predict_image(
    image_path: str | Path,
    model: tf.keras.Model | None = None,
) -> dict[str, Any]:
    """Compatibility helper for predicting one image."""
    predictor = PlantDiseasePredictor()
    if model is not None:
        predictor.model = model
    return predictor.predict(image_path)
