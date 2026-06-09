"""Command-line entry point for model training."""

import logging

from src.trainer import train_model


def main() -> int:
    """Run the end-to-end training workflow."""
    try:
        train_model()
    except Exception as exc:
        logging.getLogger("training").error("Unable to complete training: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
