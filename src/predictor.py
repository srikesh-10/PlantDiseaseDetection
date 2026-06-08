"""Plant disease prediction interfaces."""

from pathlib import Path
from typing import Any

from config import MODEL_PATH


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the future trained model artifact."""
    raise NotImplementedError("Model loading has not been implemented yet.")


def predict_image(image_path: Path, model: Any | None = None) -> dict[str, Any]:
    """Predict a disease class and confidence score for one leaf image."""
    raise NotImplementedError("Prediction logic has not been implemented yet.")

