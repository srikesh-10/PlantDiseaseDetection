"""Command-line entry point for plant disease prediction."""

from pathlib import Path

from src.predictor import predict_image


def main() -> None:
    """Invoke the future prediction workflow."""
    sample_image = Path("path/to/leaf-image.jpg")
    predict_image(sample_image)


if __name__ == "__main__":
    main()

