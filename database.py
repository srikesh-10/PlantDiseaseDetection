"""SQLite persistence helpers for prediction history."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from config import DATABASE_PATH


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    """Create the prediction-history database and schema when needed."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_prediction(
    record: dict[str, Any],
    database_path: Path = DATABASE_PATH,
) -> int:
    """Persist one prediction record and return its database identifier."""
    required_fields = ("image_path", "predicted_class", "confidence")
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"Prediction record is missing fields: {', '.join(missing)}")

    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (image_path, predicted_class, confidence)
            VALUES (?, ?, ?)
            """,
            (
                str(record["image_path"]),
                str(record["predicted_class"]),
                float(record["confidence"]),
            ),
        )
        prediction_id = cursor.lastrowid
    if prediction_id is None:
        raise RuntimeError("SQLite did not return an identifier for the prediction.")
    return int(prediction_id)


def get_prediction_history(
    limit: int = 100,
    database_path: Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return recent prediction records, newest first."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, image_path, predicted_class, confidence, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
