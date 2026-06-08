"""Training orchestration interfaces."""

from pathlib import Path
from typing import Any

from config import DATASET_DIR, MODEL_PATH


def train_model(
    dataset_dir: Path = DATASET_DIR,
    model_path: Path = MODEL_PATH,
) -> Any:
    """Run the future end-to-end model training workflow."""
    raise NotImplementedError("Model training has not been implemented yet.")

