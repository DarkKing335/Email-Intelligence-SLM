"""Email Data Engineering — Domain Layer."""

from email_data_engineering.domain.models import (
    EmailRecord,
    DatasetVersion,
    SplitManifest,
    ImportResult,
    NormalizationResult,
    CleaningResult,
    AugmentationResult,
    SplitResult,
    PipelineResult,
)
from email_data_engineering.domain.taxonomy import (
    LabelTaxonomy,
    CanonicalLabel,
    Priority,
    TaxonomyViolationError,
)

__all__ = [
    "EmailRecord",
    "DatasetVersion",
    "SplitManifest",
    "ImportResult",
    "NormalizationResult",
    "CleaningResult",
    "AugmentationResult",
    "SplitResult",
    "PipelineResult",
    "LabelTaxonomy",
    "CanonicalLabel",
    "Priority",
    "TaxonomyViolationError",
]
