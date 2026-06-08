"""Shared utility interfaces for the application."""

from pathlib import Path
from typing import Any

from config import DISEASE_INFO_PATH


def load_disease_knowledgebase(path: Path = DISEASE_INFO_PATH) -> dict[str, Any]:
    """Load disease information from the future JSON knowledge-base service."""
    raise NotImplementedError("Knowledge-base loading has not been implemented yet.")


def ensure_project_directories() -> None:
    """Create runtime directories required by future application workflows."""
    raise NotImplementedError("Directory initialization has not been implemented yet.")

