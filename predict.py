"""Command-line entry point for plant disease prediction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.disease_info import DiseaseInfoManager
from src.predictor import PlantDiseasePredictor

SEPARATOR = "=" * 50


def parse_args() -> argparse.Namespace:
    """Parse the image path supplied to the prediction CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Path to a leaf image.")
    return parser.parse_args()


def _format_list(items: list[str]) -> str:
    """Format a list of strings as bulleted lines."""
    return "\n".join(f"  - {item}" for item in items)


def main() -> int:
    """Predict one image and print the result with disease details."""
    args = parse_args()
    try:
        result = PlantDiseasePredictor().predict(args.image_path)
    except Exception as exc:
        logging.getLogger("prediction").error("Unable to predict image: %s", exc)
        print(f"Prediction failed: {exc}")
        return 1

    display_name = DiseaseInfoManager.format_display_name(
        result["predicted_class"],
    )

    print()
    print(SEPARATOR)
    print("Plant Disease Detection Result")
    print(SEPARATOR)
    print()
    print(f"Disease:\n  {display_name}")
    print()
    print(f"Confidence:\n  {result['confidence']:.2f}%")
    print()
    print("Symptoms:")
    print(_format_list(result.get("symptoms", ["N/A"])))
    print()
    print("Treatment:")
    print(_format_list(result.get("treatment", ["N/A"])))
    print()
    print("Prevention:")
    print(_format_list(result.get("prevention", ["N/A"])))
    print()
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
