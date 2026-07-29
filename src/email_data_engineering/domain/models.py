"""Domain models for the Email Data Engineering pipeline.

All models are immutable Pydantic v2 dataclasses so that pipeline stages
can pass them between boundaries without accidental mutation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Core record ───────────────────────────────────────────────────────────────

class EmailRecord(BaseModel):
    """A single email record in the data engineering pipeline.

    This is the canonical in-memory representation used by every pipeline stage.
    Records are written to JSONL files at export time.
    """

    model_config = {"frozen": True}

    # Identity
    email_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID).",
    )
    thread_id: str = Field(
        default="",
        description="Thread / conversation group identifier.",
    )

    # Content
    sender_domain: str = Field(
        default="",
        description="Sender domain extracted from the From address.",
    )
    subject: str = Field(description="Email subject line.")
    body_text: str = Field(description="Cleaned email body text.")

    # Labels
    label: str = Field(description="Canonical label from the project taxonomy.")
    original_label: str = Field(
        description="Raw label from the source dataset before normalization."
    )

    # Classification metadata
    priority: str = Field(
        description="Priority derived from the canonical label: high | medium | low."
    )
    response_required: bool = Field(
        description="Whether this email requires a reply, derived from the taxonomy."
    )

    # Provenance
    source: str = Field(
        default="",
        description="Origin file path or dataset name.",
    )
    split: str = Field(
        default="",
        description="Dataset split assignment: train | val | test.",
    )
    version: str = Field(
        default="",
        description="Dataset version string (e.g. '0.1.0').",
    )

    # Augmentation flags
    is_augmented: bool = Field(
        default=False,
        description="True when this record was synthetically derived.",
    )
    augmentation_type: str | None = Field(
        default=None,
        description="Augmentation strategy used, or None for original records.",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this record was created in the pipeline.",
    )

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        if v not in allowed:
            raise ValueError(f"priority must be one of {allowed}, got '{v}'")
        return v

    @field_validator("split")
    @classmethod
    def _validate_split(cls, v: str) -> str:
        allowed = {"train", "val", "test", ""}
        if v not in allowed:
            raise ValueError(f"split must be one of {allowed}, got '{v}'")
        return v

    def with_updates(self, **kwargs: Any) -> "EmailRecord":
        """Return a new EmailRecord with the given fields overridden."""
        return self.model_copy(update=kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSONL export."""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        return data


# ── Dataset versioning ────────────────────────────────────────────────────────

class SplitManifest(BaseModel):
    """Record counts per split for a dataset version."""

    model_config = {"frozen": True}

    train: int = Field(ge=0)
    val: int = Field(ge=0)
    test: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    def as_dict(self) -> dict[str, int]:
        return {"train": self.train, "val": self.val, "test": self.test, "total": self.total}


class DatasetVersion(BaseModel):
    """Metadata for a single versioned dataset build.

    Written alongside the JSONL split files as ``version.json``.
    """

    model_config = {"frozen": True}

    # Identification
    version: str = Field(description="Semantic version string, e.g. '0.1.0'.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the version build.",
    )

    # Source traceability
    source_path: str = Field(description="Path to the raw source file.")
    source_hash: str = Field(
        default="",
        description="SHA-256 hex digest of the raw source file.",
    )

    # Counts
    split_counts: SplitManifest = Field(description="Record counts per split.")
    label_distribution: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-split label counts, e.g. {'train': {'spam': 120, ...}}.",
    )

    # Processing metadata
    preprocessing_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of pipeline stages applied.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed used for augmentation and splitting.",
    )

    # Config snapshot
    config_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Full YAML config snapshot at build time.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON export."""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["split_counts"] = self.split_counts.as_dict()
        return data


# ── Stage result objects ───────────────────────────────────────────────────────

class ImportResult(BaseModel):
    """Result produced by the Import stage (US-1.1)."""

    model_config = {"frozen": True}

    total_rows_read: int = Field(ge=0)
    valid_records: int = Field(ge=0)
    rejected_records: int = Field(ge=0)
    source_path: str
    rejection_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="Reason → count mapping for rejected records.",
    )
    records: list[EmailRecord] = Field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.total_rows_read == 0:
            return 0.0
        return self.valid_records / self.total_rows_read


class NormalizationResult(BaseModel):
    """Result produced by the Normalize stage (US-1.2)."""

    model_config = {"frozen": True}

    total_records: int = Field(ge=0)
    normalized_records: int = Field(ge=0)
    rejected_unknown_label: int = Field(ge=0)
    mapping_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="original_label → {canonical_label: count}.",
    )
    unmapped_labels: list[str] = Field(
        default_factory=list,
        description="Source labels that had no mapping and were rejected.",
    )
    records: list[EmailRecord] = Field(default_factory=list)


class CleaningResult(BaseModel):
    """Result produced by the Clean stage (US-1.3)."""

    model_config = {"frozen": True}

    input_records: int = Field(ge=0)
    output_records: int = Field(ge=0)
    exact_duplicates_removed: int = Field(ge=0)
    near_duplicates_removed: int = Field(ge=0)
    pii_redactions: int = Field(ge=0)
    bodies_clipped: int = Field(ge=0)
    html_stripped: int = Field(ge=0)
    signatures_stripped: int = Field(ge=0)
    records: list[EmailRecord] = Field(default_factory=list)

    @property
    def records_removed(self) -> int:
        return self.input_records - self.output_records


class AugmentationResult(BaseModel):
    """Result produced by the Augment stage (US-1.4)."""

    model_config = {"frozen": True}

    original_records: int = Field(ge=0)
    augmented_records: int = Field(ge=0)
    random_seed: int
    strategy_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Augmentation strategy → number of records produced.",
    )
    records: list[EmailRecord] = Field(default_factory=list)

    @property
    def total_records(self) -> int:
        return self.original_records + self.augmented_records


class SplitResult(BaseModel):
    """Result produced by the Split stage."""

    model_config = {"frozen": True}

    total_records: int = Field(ge=0)
    split_counts: SplitManifest
    label_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)
    random_seed: int
    stratified: bool
    train_records: list[EmailRecord] = Field(default_factory=list)
    val_records: list[EmailRecord] = Field(default_factory=list)
    test_records: list[EmailRecord] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Aggregated result of the full 8-stage pipeline run."""

    model_config = {"frozen": True}

    success: bool
    version: str
    import_result: ImportResult | None = None
    normalization_result: NormalizationResult | None = None
    cleaning_result: CleaningResult | None = None
    augmentation_result: AugmentationResult | None = None
    split_result: SplitResult | None = None
    dataset_version: DatasetVersion | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
