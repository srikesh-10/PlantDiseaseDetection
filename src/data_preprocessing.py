"""Dataset and image preprocessing interfaces."""

from pathlib import Path
from typing import Any


def prepare_datasets(dataset_dir: Path) -> tuple[Any, Any, Any]:
    """Prepare future training, validation, and test datasets."""
    raise NotImplementedError("Dataset preprocessing has not been implemented yet.")


def preprocess_image(image_path: Path) -> Any:
    """Load and preprocess one image for future model inference."""
    raise NotImplementedError("Image preprocessing has not been implemented yet.")

