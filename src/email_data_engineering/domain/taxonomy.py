"""Label taxonomy for the Email Data Engineering pipeline.

Loads the canonical label definitions and source-to-canonical mapping from
``configs/label_taxonomy.yaml`` and provides validation and lookup utilities
used by the Normalize stage (US-1.2).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml


# ── Enums ─────────────────────────────────────────────────────────────────────

class CanonicalLabel(str, Enum):
    """All approved canonical labels for the project taxonomy."""

    RESPOND = "respond"
    FYI = "fyi"
    ARCHIVE = "archive"
    UNSUBSCRIBE = "unsubscribe"
    SPAM = "spam"
    SECURITY = "security"
    BILLING = "billing"
    SUPPORT = "support"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Exceptions ────────────────────────────────────────────────────────────────

class TaxonomyViolationError(ValueError):
    """Raised when a label is not in the approved taxonomy."""

    def __init__(self, raw_label: str, allowed: set[str]) -> None:
        self.raw_label = raw_label
        self.allowed = allowed
        super().__init__(
            f"Label '{raw_label}' is not in the approved taxonomy. "
            f"Allowed values: {sorted(allowed)}"
        )


# ── Taxonomy ──────────────────────────────────────────────────────────────────

class LabelTaxonomy:
    """Loads label definitions and provides mapping / validation services.

    Parameters
    ----------
    taxonomy_path:
        Path to ``label_taxonomy.yaml``.  Defaults to the bundled config.
    """

    def __init__(self, taxonomy_path: Path | str | None = None) -> None:
        if taxonomy_path is None:
            # Resolve relative to the project root (3 levels up from this file)
            taxonomy_path = (
                Path(__file__).parent.parent.parent.parent
                / "configs"
                / "label_taxonomy.yaml"
            )
        self._path = Path(taxonomy_path)
        self._data = self._load(self._path)
        self._canonical: dict[str, dict[str, Any]] = self._data["canonical_labels"]
        self._mapping: dict[str, str] = {
            k.lower().strip(): v
            for k, v in self._data["label_mapping"].items()
        }

    # ── Loading ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if "canonical_labels" not in data or "label_mapping" not in data:
            raise ValueError(
                "Taxonomy YAML must contain 'canonical_labels' and 'label_mapping' keys."
            )
        return data

    # ── Validation ────────────────────────────────────────────────────────────

    def is_canonical(self, label: str) -> bool:
        """Return True if *label* is already a canonical label."""
        return label.lower().strip() in self._canonical

    def is_mappable(self, raw_label: str) -> bool:
        """Return True if *raw_label* can be mapped to a canonical label."""
        key = raw_label.lower().strip()
        return key in self._mapping or key in self._canonical

    # ── Normalization ─────────────────────────────────────────────────────────

    def normalize(self, raw_label: str) -> str:
        """Map *raw_label* to its canonical label.

        Parameters
        ----------
        raw_label:
            The label value from the source dataset.

        Returns
        -------
        str
            Canonical label string (one of :class:`CanonicalLabel` values).

        Raises
        ------
        TaxonomyViolationError
            When *raw_label* has no mapping and is not itself canonical.
        """
        key = raw_label.lower().strip()

        # Already canonical — return as-is
        if key in self._canonical:
            return key

        # Has an explicit mapping
        if key in self._mapping:
            return self._mapping[key]

        raise TaxonomyViolationError(raw_label=raw_label, allowed=set(self._canonical))

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def get_priority(self, canonical_label: str) -> str:
        """Return the priority string for a canonical label."""
        label_info = self._canonical.get(canonical_label.lower().strip())
        if label_info is None:
            raise TaxonomyViolationError(
                raw_label=canonical_label, allowed=set(self._canonical)
            )
        return label_info["priority"]

    def get_response_required(self, canonical_label: str) -> bool:
        """Return whether a reply is required for a canonical label."""
        label_info = self._canonical.get(canonical_label.lower().strip())
        if label_info is None:
            raise TaxonomyViolationError(
                raw_label=canonical_label, allowed=set(self._canonical)
            )
        return bool(label_info["response_required"])

    def get_description(self, canonical_label: str) -> str:
        label_info = self._canonical.get(canonical_label.lower().strip())
        if label_info is None:
            return ""
        return label_info.get("description", "")

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def canonical_labels(self) -> set[str]:
        """Return the set of all canonical label strings."""
        return set(self._canonical.keys())

    @property
    def source_labels(self) -> set[str]:
        """Return all source labels that have an explicit mapping."""
        return set(self._mapping.keys())

    def mapping_table(self) -> dict[str, str]:
        """Return the full source → canonical mapping dict."""
        return dict(self._mapping)

    def label_info(self, canonical_label: str) -> dict[str, Any]:
        """Return the full metadata dict for a canonical label."""
        info = self._canonical.get(canonical_label.lower().strip())
        if info is None:
            raise TaxonomyViolationError(
                raw_label=canonical_label, allowed=set(self._canonical)
            )
        return dict(info)
