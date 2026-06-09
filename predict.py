"""Command-line entry point for plant disease prediction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.predictor import PlantDiseasePredictor


def parse_args() -> argparse.Namespace:
    """Parse the image path supplied to the prediction CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Path to a leaf image.")
    return parser.parse_args()


def main() -> int:
    """Predict one image and print the class and confidence."""
    args = parse_args()
    try:
        result = PlantDiseasePredictor().predict(args.image_path)
    except Exception as exc:
        logging.getLogger("prediction").error("Unable to predict image: %s", exc)
        print(f"Prediction failed: {exc}")
        return 1

    print(f"Disease: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
