"""Prepare selected PlantVillage images for model training.

Run this module directly from the project root:

    python src/data_preprocessing.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

# Direct execution places ``src`` on sys.path, so add the project root before
# importing shared configuration.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    IMAGE_SIZE,
    LOGS_DIR,
    RANDOM_SEED,
    TEST_DATASET_DIR,
    TRAIN_DATASET_DIR,
    VAL_DATASET_DIR,
)

LOGGER = logging.getLogger("dataset_preprocessor")

SELECTED_CLASSES: Final[tuple[str, ...]] = (
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
)

# The downloaded PlantVillage dataset uses single underscores for Tomato class
# directories. Generated split directories use the canonical names above.
SOURCE_CLASS_ALIASES: Final[dict[str, str]] = {
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___healthy": "Tomato_healthy",
}

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


@dataclass(frozen=True)
class SplitPaths:
    """Output directories for generated dataset splits."""

    train: Path
    val: Path
    test: Path

    def as_dict(self) -> dict[str, Path]:
        """Return split names mapped to their output directories."""
        return {"train": self.train, "val": self.val, "test": self.test}


class DatasetPreprocessor:
    """Validate, split, resize, and save selected PlantVillage classes."""

    def __init__(
        self,
        source_dir: Path,
        split_paths: SplitPaths,
        report_path: Path,
        image_size: tuple[int, int] = IMAGE_SIZE,
        seed: int = RANDOM_SEED,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        """Configure a deterministic dataset preparation run."""
        self.source_dir = source_dir.resolve()
        self.split_paths = split_paths
        self.report_path = report_path
        self.image_size = image_size
        self.seed = seed
        self.ratios = {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        }
        self.errors: list[dict[str, str]] = []

    def validate_paths(self) -> None:
        """Validate the source dataset and all selected class directories."""
        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"Raw PlantVillage directory does not exist: {self.source_dir}"
            )

        ratio_total = sum(self.ratios.values())
        if abs(ratio_total - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must total 1.0, received {ratio_total}.")

        if len(self.image_size) != 2 or any(dimension <= 0 for dimension in self.image_size):
            raise ValueError(f"Image size must contain two positive values: {self.image_size}")

        missing = [
            str(self.source_dir / SOURCE_CLASS_ALIASES[class_name])
            for class_name in SELECTED_CLASSES
            if not (self.source_dir / SOURCE_CLASS_ALIASES[class_name]).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "Required source class directories are missing:\n" + "\n".join(missing)
            )

    def prepare(self) -> dict[str, Any]:
        """Generate the train, validation, and test datasets and return statistics."""
        self.validate_paths()
        self._prepare_output_directories()

        report: dict[str, Any] = {
            "source_directory": str(self.source_dir),
            "image_size": list(self.image_size),
            "split_ratios": self.ratios,
            "seed": self.seed,
            "selected_classes": list(SELECTED_CLASSES),
            "classes": {},
            "totals": {"total": 0, "train": 0, "val": 0, "test": 0},
            "errors": self.errors,
        }

        LOGGER.info("Preparing %d selected classes from %s", len(SELECTED_CLASSES), self.source_dir)
        for class_index, class_name in enumerate(SELECTED_CLASSES):
            class_stats = self._process_class(class_name, class_index)
            report["classes"][class_name] = class_stats
            for key in report["totals"]:
                report["totals"][key] += class_stats[key]

        report["error_count"] = len(self.errors)
        self._write_report(report)
        LOGGER.info(
            "Dataset preparation complete: %d images processed, %d errors",
            report["totals"]["total"],
            report["error_count"],
        )
        return report

    def _prepare_output_directories(self) -> None:
        """Recreate selected split class directories without touching raw data."""
        for split_dir in self.split_paths.as_dict().values():
            split_dir.mkdir(parents=True, exist_ok=True)
            for child in split_dir.iterdir():
                if child.name == ".gitkeep":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            for class_name in SELECTED_CLASSES:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    def _process_class(self, class_name: str, class_index: int) -> dict[str, int]:
        """Validate, split, resize, and save all readable images for one class."""
        source_class = self.source_dir / SOURCE_CLASS_ALIASES[class_name]
        candidates = sorted(
            path
            for path in source_class.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        valid_images = [path for path in candidates if self._is_readable_image(path)]
        splits = self._split_images(valid_images, class_index)

        LOGGER.info(
            "%s: %d candidates, %d readable images",
            class_name,
            len(candidates),
            len(valid_images),
        )

        saved_counts = {"train": 0, "val": 0, "test": 0}
        for split_name, images in splits.items():
            output_dir = self.split_paths.as_dict()[split_name] / class_name
            for image_number, source_path in enumerate(images, start=1):
                if self._resize_and_save(source_path, output_dir / source_path.name):
                    saved_counts[split_name] += 1
                if image_number % 250 == 0:
                    LOGGER.info(
                        "%s/%s: processed %d of %d images",
                        split_name,
                        class_name,
                        image_number,
                        len(images),
                    )

        total_saved = sum(saved_counts.values())
        LOGGER.info("%s: saved %d images", class_name, total_saved)
        return {
            "total": total_saved,
            "train": saved_counts["train"],
            "val": saved_counts["val"],
            "test": saved_counts["test"],
            "unreadable": len(candidates) - len(valid_images),
        }

    def _is_readable_image(self, image_path: Path) -> bool:
        """Return whether Pillow can decode an image, recording failures."""
        try:
            with Image.open(image_path) as image:
                image.verify()
            return True
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            self._record_error(image_path, f"Unreadable image: {exc}")
            return False

    def _split_images(
        self, images: Sequence[Path], class_index: int
    ) -> dict[str, list[Path]]:
        """Shuffle and split images deterministically for one class."""
        shuffled = list(images)
        random.Random(self.seed + class_index).shuffle(shuffled)

        train_end = int(len(shuffled) * self.ratios["train"])
        val_end = train_end + int(len(shuffled) * self.ratios["val"])
        return {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    def _resize_and_save(self, source_path: Path, destination_path: Path) -> bool:
        """Decode an image, convert it to RGB, resize it, and save it."""
        try:
            with Image.open(source_path) as image:
                rgb_image = ImageOps.exif_transpose(image).convert("RGB")
                resized_image = rgb_image.resize(self.image_size, Image.Resampling.LANCZOS)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                resized_image.save(destination_path)
            return True
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            self._record_error(source_path, f"Processing failed: {exc}")
            return False

    def _record_error(self, image_path: Path, message: str) -> None:
        """Record and log an image-level processing error."""
        self.errors.append({"path": str(image_path), "error": message})
        LOGGER.error("%s: %s", image_path, message)

    def _write_report(self, report: dict[str, Any]) -> None:
        """Write dataset statistics and encountered errors as JSON."""
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("Dataset report written to %s", self.report_path)


def configure_logging(log_level: str = "INFO") -> Path:
    """Configure console and file logging and return the log file path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "dataset_preprocessing.log"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def prepare_datasets(
    dataset_dir: Path = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage",
) -> dict[str, Any]:
    """Prepare the selected classes using the project's default output paths."""
    preprocessor = DatasetPreprocessor(
        source_dir=dataset_dir,
        split_paths=SplitPaths(
            train=TRAIN_DATASET_DIR,
            val=VAL_DATASET_DIR,
            test=TEST_DATASET_DIR,
        ),
        report_path=PROJECT_ROOT / "docs" / "dataset_report.json",
    )
    return preprocessor.prepare()


def preprocess_image(image_path: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    """Load one image as an RGB Pillow image resized for future inference."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").resize(
                image_size,
                Image.Resampling.LANCZOS,
            )
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unable to preprocess image {image_path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "raw" / "PlantVillage",
        help="Path to the raw PlantVillage class directories.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0])
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    """Run dataset preparation from the command line."""
    args = parse_args()
    log_path = configure_logging(args.log_level)
    preprocessor = DatasetPreprocessor(
        source_dir=args.source_dir,
        split_paths=SplitPaths(
            train=TRAIN_DATASET_DIR,
            val=VAL_DATASET_DIR,
            test=TEST_DATASET_DIR,
        ),
        report_path=PROJECT_ROOT / "docs" / "dataset_report.json",
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
    )

    try:
        report = preprocessor.prepare()
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as exc:
        LOGGER.exception("Dataset preparation failed: %s", exc)
        return 1

    LOGGER.info("Log written to %s", log_path)
    LOGGER.info("Final statistics: %s", report["totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
