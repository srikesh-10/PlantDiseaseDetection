"""Command-line entry point for model training."""

from src.trainer import train_model


def main() -> None:
    """Invoke the future training workflow."""
    train_model()


if __name__ == "__main__":
    main()

