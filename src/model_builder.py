"""MobileNetV2 transfer-learning model construction interface."""

from typing import Any

from config import CLASS_NAMES, IMAGE_SIZE


def build_model(
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, 3),
    num_classes: int = len(CLASS_NAMES),
) -> Any:
    """Build and compile the future MobileNetV2 transfer-learning model."""
    raise NotImplementedError("Model construction has not been implemented yet.")

