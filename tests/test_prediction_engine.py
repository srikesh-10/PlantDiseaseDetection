"""Tests for prediction persistence and image preprocessing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from database import get_prediction_history, save_prediction
from src.predictor import load_labels, preprocess_image


def test_preprocess_image_returns_mobilenet_batch(tmp_path: Path) -> None:
    """Convert a valid image into a normalized 224x224 RGB batch."""
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (30, 40), color=(255, 128, 0)).save(image_path)

    batch = preprocess_image(image_path)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert float(batch.min()) >= -1.0
    assert float(batch.max()) <= 1.0


def test_preprocess_image_rejects_invalid_image(tmp_path: Path) -> None:
    """Raise a useful error for a file Pillow cannot decode."""
    image_path = tmp_path / "invalid.jpg"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read image"):
        preprocess_image(image_path)


def test_load_labels_orders_index_mapping(tmp_path: Path) -> None:
    """Read labels according to model output indices."""
    labels_path = tmp_path / "labels.json"
    labels_path.write_text('{"1": "tomato", "0": "potato"}', encoding="utf-8")

    assert load_labels(labels_path) == ["potato", "tomato"]


def test_save_and_read_prediction_history(tmp_path: Path) -> None:
    """Persist prediction history and return newest records first."""
    database_path = tmp_path / "history.sqlite3"
    prediction_id = save_prediction(
        {
            "image_path": "leaf.jpg",
            "predicted_class": "Tomato___healthy",
            "confidence": 98.25,
        },
        database_path,
    )

    history = get_prediction_history(database_path=database_path)

    assert prediction_id == 1
    assert history[0]["predicted_class"] == "Tomato___healthy"
    assert history[0]["confidence"] == 98.25
