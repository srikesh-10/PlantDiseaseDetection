"""Basic tests for scaffold-level configuration."""

from config import BEST_MODEL_PATH, CLASS_NAMES, FINAL_MODEL_PATH, SAVED_MODELS_DIR


def test_exactly_six_supported_classes() -> None:
    """Confirm that the initial project scope remains limited to six classes."""
    assert len(CLASS_NAMES) == 6


def test_models_use_native_keras_format() -> None:
    """Confirm that future model artifacts use the native Keras format."""
    assert BEST_MODEL_PATH == SAVED_MODELS_DIR / "best_model.keras"
    assert FINAL_MODEL_PATH == SAVED_MODELS_DIR / "final_model.keras"
