"""Disease knowledge base manager for PlantDiseaseDetection.

Loads disease information from a JSON knowledge base and maps model output
labels to their corresponding entries.  Designed for modular integration
with both CLI and future Streamlit workflows.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from config import DISEASE_INFO_PATH

LOGGER = logging.getLogger("prediction")


class DiseaseInfoManager:
    """Load and query the disease knowledge base.

    Attributes:
        knowledgebase_path: Path to the ``disease_info.json`` file.
        knowledgebase: Parsed knowledge base dictionary keyed by canonical
            disease name (e.g. ``Potato_Early_Blight``).
    """

    # Fields expected in every knowledge-base entry.
    _EXPECTED_FIELDS: tuple[str, ...] = ("symptoms", "treatment", "prevention")

    def __init__(self, knowledgebase_path: Path = DISEASE_INFO_PATH) -> None:
        """Initialize the manager and load the knowledge base.

        Args:
            knowledgebase_path: Filesystem path to the JSON knowledge base.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid JSON or unexpected
                structure.
        """
        self.knowledgebase_path = knowledgebase_path
        self.knowledgebase: dict[str, dict[str, list[str]]] = {}
        self.load_knowledgebase()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_knowledgebase(self) -> dict[str, dict[str, list[str]]]:
        """Load (or reload) the disease knowledge base from disk.

        Returns:
            The parsed knowledge base dictionary.

        Raises:
            FileNotFoundError: If ``knowledgebase_path`` does not exist.
            ValueError: If the file is not valid JSON or has an unexpected
                top-level structure.
        """
        path = self.knowledgebase_path
        if not path.is_file():
            LOGGER.error("Knowledge base file not found: %s", path)
            raise FileNotFoundError(
                f"Disease knowledge base does not exist: {path}"
            )

        try:
            raw_text = path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            LOGGER.error("Invalid JSON in knowledge base %s: %s", path, exc)
            raise ValueError(
                f"Knowledge base contains invalid JSON: {path}"
            ) from exc

        if not isinstance(data, dict):
            LOGGER.error(
                "Knowledge base must be a JSON object, got %s",
                type(data).__name__,
            )
            raise ValueError(
                f"Knowledge base must be a JSON object, got "
                f"{type(data).__name__}: {path}"
            )

        self.knowledgebase = data
        LOGGER.info(
            "Loaded disease knowledge base with %d entries from %s",
            len(data),
            path,
        )
        return self.knowledgebase

    def get_disease_info(self, class_name: str) -> dict[str, Any]:
        """Return knowledge-base info for a model class label.

        The method automatically converts model labels such as
        ``Potato___Early_blight`` into the knowledge-base key format
        ``Potato_Early_Blight``.

        Args:
            class_name: Model output class name (e.g.
                ``Potato___Early_blight``).

        Returns:
            A dictionary with ``symptoms``, ``treatment``, and
            ``prevention`` lists.  If the entry is missing from the
            knowledge base, each list will contain a single fallback
            message.
        """
        kb_key = self._convert_label(class_name)
        LOGGER.info(
            "Looking up disease info: model_label=%s -> kb_key=%s",
            class_name,
            kb_key,
        )

        entry = self.knowledgebase.get(kb_key)
        if entry is None:
            LOGGER.warning(
                "No knowledge-base entry found for '%s' (key: '%s')",
                class_name,
                kb_key,
            )
            return {
                "symptoms": ["Information not available for this class."],
                "treatment": ["Information not available for this class."],
                "prevention": ["Information not available for this class."],
            }

        # Ensure every expected field is present even if the JSON entry is
        # incomplete.
        result: dict[str, Any] = {}
        for field in self._EXPECTED_FIELDS:
            value = entry.get(field)
            if isinstance(value, list) and value:
                result[field] = value
            else:
                LOGGER.warning(
                    "Missing or empty field '%s' for entry '%s'",
                    field,
                    kb_key,
                )
                result[field] = [f"No {field} information available."]

        LOGGER.info(
            "Disease info retrieved for '%s': %d symptom(s), "
            "%d treatment(s), %d prevention(s)",
            kb_key,
            len(result["symptoms"]),
            len(result["treatment"]),
            len(result["prevention"]),
        )
        return result

    # ------------------------------------------------------------------
    # Label conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_label(model_label: str) -> str:
        """Convert a model class label to a knowledge-base key.

        Conversion rules applied in order:

        1. Replace triple underscores (``___``) with a single underscore.
        2. Convert the whole string to *Title Case* while preserving
           underscores as word separators.

        Examples::

            Potato___Early_blight  ->  Potato_Early_Blight
            Tomato___healthy       ->  Tomato_Healthy

        Args:
            model_label: Raw class label produced by the model.

        Returns:
            The corresponding knowledge-base key.
        """
        # Step 1: Replace triple underscores with a single one.
        normalized = model_label.replace("___", "_")

        # Step 2: Title-case each segment separated by underscores.
        parts = normalized.split("_")
        titled_parts = [part.capitalize() for part in parts if part]

        return "_".join(titled_parts)

    @staticmethod
    def format_display_name(model_label: str) -> str:
        """Return a human-readable disease name from a model label.

        Examples::

            Potato___Early_blight  ->  Potato Early Blight
            Tomato___healthy       ->  Tomato Healthy

        Args:
            model_label: Raw class label produced by the model.

        Returns:
            A human-friendly display string.
        """
        normalized = model_label.replace("___", " ").replace("_", " ")
        return " ".join(word.capitalize() for word in normalized.split())
