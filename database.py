"""SQLite persistence boundary for prediction history.

Schema creation and CRUD behavior are intentionally deferred until prediction
history requirements are finalized.
"""

from pathlib import Path
from typing import Any

from config import DATABASE_PATH


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    """Create the future prediction-history database and schema."""
    raise NotImplementedError("Database initialization has not been implemented yet.")


def save_prediction(record: dict[str, Any]) -> int:
    """Persist one prediction record and return its future database identifier."""
    raise NotImplementedError("Prediction persistence has not been implemented yet.")


def get_prediction_history(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent prediction records, newest first."""
    raise NotImplementedError("Prediction history has not been implemented yet.")

