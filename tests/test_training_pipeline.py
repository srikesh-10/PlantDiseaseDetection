"""Focused tests for training pipeline helpers."""

import json
from pathlib import Path

import pytest

from src.trainer import _calculate_class_weights, _validate_dataset
from src.utils import save_labels


def test_validate_dataset_requires_all_splits(tmp_path: Path) -> None:
    """Reject dataset roots that do not contain every required split."""
    (tmp_path / "train").mkdir()

    with pytest.raises(FileNotFoundError, match="val"):
        _validate_dataset(tmp_path)


def test_calculate_class_weights_favors_smaller_classes(tmp_path: Path) -> None:
    """Assign a larger weight to a class with fewer training images."""
    class_names = ["large", "small"]
    for class_name, count in (("large", 4), ("small", 1)):
        class_dir = tmp_path / "train" / class_name
        class_dir.mkdir(parents=True)
        for index in range(count):
            (class_dir / f"{index}.jpg").touch()

    weights = _calculate_class_weights(tmp_path, class_names)

    assert weights[1] > weights[0]


def test_save_labels_writes_index_mapping(tmp_path: Path) -> None:
    """Save class labels in model output index order."""
    path = tmp_path / "labels.json"

    save_labels(["potato", "tomato"], path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "0": "potato",
        "1": "tomato",
    }
