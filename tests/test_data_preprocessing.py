"""Tests for deterministic dataset preprocessing helpers."""

from pathlib import Path

from src.data_preprocessing import DatasetPreprocessor, SplitPaths


def test_split_ratios_assign_every_image(tmp_path: Path) -> None:
    """Ensure the 70/15/15 split assigns every input exactly once."""
    preprocessor = DatasetPreprocessor(
        source_dir=tmp_path,
        split_paths=SplitPaths(tmp_path / "train", tmp_path / "val", tmp_path / "test"),
        report_path=tmp_path / "report.json",
    )
    images = [Path(f"image-{index}.jpg") for index in range(100)]

    splits = preprocessor._split_images(images, class_index=0)

    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15
    assert set().union(*map(set, splits.values())) == set(images)
