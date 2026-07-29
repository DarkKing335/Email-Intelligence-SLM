"""Normalize stage — US-1.2.

Maps raw source labels to canonical project taxonomy labels using
:class:`LabelTaxonomy`.  Preserves the original label on every record
and emits a :class:`NormalizationResult` with full mapping statistics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from email_data_engineering.domain.models import EmailRecord, NormalizationResult
from email_data_engineering.domain.taxonomy import LabelTaxonomy, TaxonomyViolationError


class LabelNormalizer:
    """Normalize stage: map source labels to canonical taxonomy labels.

    Implements US-1.2 acceptance criteria:
    - Labels follow a fixed taxonomy.
    - Each mapped record stores the original label and normalized label.
    - Records with labels outside the taxonomy are rejected (or defaulted).
    - Mapping stats are written to an auditable report.

    Parameters
    ----------
    taxonomy:
        :class:`LabelTaxonomy` instance.  If not provided, loaded from
        the default ``configs/label_taxonomy.yaml``.
    unknown_label_strategy:
        ``"reject"`` → drop records with unknown labels.
        ``"default"`` → assign *default_label* to unknown records.
    default_label:
        Used when *unknown_label_strategy* is ``"default"``.
    report_dir:
        Directory where ``normalization_report.json`` is written.
    """

    def __init__(
        self,
        taxonomy: LabelTaxonomy | None = None,
        unknown_label_strategy: str = "reject",
        default_label: str = "fyi",
        report_dir: str | Path = "reports",
    ) -> None:
        self._taxonomy = taxonomy or LabelTaxonomy()
        if unknown_label_strategy not in ("reject", "default"):
            raise ValueError(
                f"unknown_label_strategy must be 'reject' or 'default', "
                f"got '{unknown_label_strategy}'"
            )
        self._strategy = unknown_label_strategy
        self._default_label = default_label
        self._report_dir = Path(report_dir)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, records: list[EmailRecord]) -> NormalizationResult:
        """Normalize labels for all *records*.

        Returns
        -------
        NormalizationResult
            Normalized records and mapping statistics.
        """
        logger.info(f"[Normalize] Normalizing labels for {len(records)} records")

        normalized: list[EmailRecord] = []
        rejected_unknown = 0
        unmapped_labels: list[str] = []
        # mapping_counts: original_label -> {canonical_label: count}
        mapping_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for record in records:
            raw_label = record.original_label or record.label

            try:
                canonical = self._taxonomy.normalize(raw_label)
                priority = self._taxonomy.get_priority(canonical)
                response_required = self._taxonomy.get_response_required(canonical)

                updated = record.with_updates(
                    label=canonical,
                    original_label=raw_label,
                    priority=priority,
                    response_required=response_required,
                )
                normalized.append(updated)
                mapping_counts[raw_label][canonical] += 1

            except TaxonomyViolationError:
                if self._strategy == "default":
                    canonical = self._default_label
                    priority = self._taxonomy.get_priority(canonical)
                    response_required = self._taxonomy.get_response_required(canonical)
                    updated = record.with_updates(
                        label=canonical,
                        original_label=raw_label,
                        priority=priority,
                        response_required=response_required,
                    )
                    normalized.append(updated)
                    mapping_counts[raw_label][canonical] += 1
                    logger.warning(
                        f"Unknown label '{raw_label}' defaulted to '{canonical}'"
                    )
                else:
                    rejected_unknown += 1
                    if raw_label not in unmapped_labels:
                        unmapped_labels.append(raw_label)
                    logger.warning(
                        f"Rejecting record with unknown label: '{raw_label}'"
                    )

        result = NormalizationResult(
            total_records=len(records),
            normalized_records=len(normalized),
            rejected_unknown_label=rejected_unknown,
            mapping_counts={k: dict(v) for k, v in mapping_counts.items()},
            unmapped_labels=unmapped_labels,
            records=normalized,
        )

        self._write_report(result)

        logger.info(
            f"[Normalize] Done — {result.normalized_records} normalized, "
            f"{result.rejected_unknown_label} rejected (unknown label), "
            f"{len(result.unmapped_labels)} unmapped label types"
        )
        return result

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _write_report(self, result: NormalizationResult) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "stage": "normalize",
            "total_records": result.total_records,
            "normalized_records": result.normalized_records,
            "rejected_unknown_label": result.rejected_unknown_label,
            "unmapped_labels": result.unmapped_labels,
            "mapping_counts": result.mapping_counts,
            "canonical_label_distribution": self._label_distribution(result.records),
        }
        path = self._report_dir / "normalization_report.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.debug(f"Normalization report written: {path}")

    @staticmethod
    def _label_distribution(records: list[EmailRecord]) -> dict[str, int]:
        dist: dict[str, int] = defaultdict(int)
        for r in records:
            dist[r.label] += 1
        return dict(dist)
