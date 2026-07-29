"""Split stage.

Splits records into stratified train, validation, and test datasets.
Controlled by a random seed for reproducible splits.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from sklearn.model_selection import train_test_split

from email_data_engineering.domain.models import EmailRecord, SplitManifest, SplitResult


class DataSplitter:
    """Split stage: split dataset into train, validation, and test splits.

    Maintains class stratification and deterministic splitting via random seed.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        stratify: bool = True,
        report_dir: str | Path = "reports",
    ) -> None:
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._seed = random_seed
        self._stratify = stratify
        self._report_dir = Path(report_dir)

        # Validate ratios sum to 1.0 roughly
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got: {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
            )

    def run(self, records: list[EmailRecord]) -> SplitResult:
        logger.info(
            f"[Split] Splitting {len(records)} records with seed={self._seed}, "
            f"ratios={self._train_ratio}/{self._val_ratio}/{self._test_ratio}"
        )

        if not records:
            empty_manifest = SplitManifest(train=0, val=0, test=0)
            return SplitResult(
                total_records=0,
                split_counts=empty_manifest,
                label_distribution={},
                random_seed=self._seed,
                stratified=self._stratify,
                train_records=[],
                val_records=[],
                test_records=[],
            )

        # Handle very small datasets where splitting is tricky
        if len(records) < 3:
            logger.warning("[Split] Dataset is too small (<3 records). Placing all in train split.")
            train_recs = [r.with_updates(split="train") for r in records]
            counts = SplitManifest(train=len(train_recs), val=0, test=0)
            result = SplitResult(
                total_records=len(records),
                split_counts=counts,
                label_distribution=self._compute_label_dist(train_recs, [], []),
                random_seed=self._seed,
                stratified=False,
                train_records=train_recs,
                val_records=[],
                test_records=[],
            )
            self._write_report(result)
            return result

        # Labels for stratification
        labels = [r.label for r in records]

        # In case some labels only have 1 instance, stratification will fail in sklearn.
        # We check label counts. If any label has count < 2 and stratify is True, we disable stratification or log it.
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1

        use_stratify = self._stratify
        if use_stratify:
            min_instances = min(label_counts.values())
            # For 3-way split, we need at least 3 instances to stratify properly if we split twice.
            # But let's check if we can. If the smallest class has less than 2 items, we must disable stratify.
            if min_instances < 2:
                logger.warning(
                    f"[Split] Smallest class has {min_instances} instance(s). "
                    "Disabling stratification for this split to avoid sklearn errors."
                )
                use_stratify = False

        # Split 1: Split test out
        # test_ratio is the size of the test split relative to 1.0
        test_size = self._test_ratio
        
        try:
            if use_stratify:
                temp_records, test_records, _, _ = train_test_split(
                    records,
                    labels,
                    test_size=test_size,
                    random_state=self._seed,
                    stratify=labels,
                )
            else:
                temp_records, test_records = train_test_split(
                    records,
                    test_size=test_size,
                    random_state=self._seed,
                )

            # Split 2: Split train and val out of the temp_records
            # The remaining val size is relative to train + val ratio
            val_relative_size = self._val_ratio / (self._train_ratio + self._val_ratio)
            temp_labels = [r.label for r in temp_records]

            # Re-check stratification eligibility for split 2
            temp_label_counts = defaultdict(int)
            for tl in temp_labels:
                temp_label_counts[tl] += 1
            
            use_stratify_2 = use_stratify
            if use_stratify_2 and min(temp_label_counts.values()) < 2:
                use_stratify_2 = False

            if use_stratify_2:
                train_records, val_records, _, _ = train_test_split(
                    temp_records,
                    temp_labels,
                    test_size=val_relative_size,
                    random_state=self._seed,
                    stratify=temp_labels,
                )
            else:
                train_records, val_records = train_test_split(
                    temp_records,
                    test_size=val_relative_size,
                    random_state=self._seed,
                )
        except Exception as exc:
            logger.warning(f"[Split] Stratification split failed: {exc}. Falling back to random splitting.")
            # Fallback to pure random split if stratification still fails for some edge cases
            temp_records, test_records = train_test_split(
                records,
                test_size=test_size,
                random_state=self._seed,
            )
            val_relative_size = self._val_ratio / (self._train_ratio + self._val_ratio)
            train_records, val_records = train_test_split(
                temp_records,
                test_size=val_relative_size,
                random_state=self._seed,
            )
            use_stratify = False

        # Tag splits
        train_tagged = [r.with_updates(split="train") for r in train_records]
        val_tagged = [r.with_updates(split="val") for r in val_records]
        test_tagged = [r.with_updates(split="test") for r in test_records]

        counts = SplitManifest(
            train=len(train_tagged),
            val=len(val_tagged),
            test=len(test_tagged),
        )

        label_dist = self._compute_label_dist(train_tagged, val_tagged, test_tagged)

        result = SplitResult(
            total_records=len(records),
            split_counts=counts,
            label_distribution=label_dist,
            random_seed=self._seed,
            stratified=use_stratify,
            train_records=train_tagged,
            val_records=val_tagged,
            test_records=test_tagged,
        )

        self._write_report(result)
        logger.info(
            f"[Split] Split complete. Train: {len(train_tagged)}, Val: {len(val_tagged)}, Test: {len(test_tagged)}"
        )
        return result

    @staticmethod
    def _compute_label_dist(
        train: list[EmailRecord], val: list[EmailRecord], test: list[EmailRecord]
    ) -> dict[str, dict[str, int]]:
        dist: dict[str, dict[str, int]] = {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int),
        }
        for r in train:
            dist["train"][r.label] += 1
        for r in val:
            dist["val"][r.label] += 1
        for r in test:
            dist["test"][r.label] += 1
        return {split: dict(vals) for split, vals in dist.items()}

    def _write_report(self, result: SplitResult) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "stage": "split",
            "total_records": result.total_records,
            "split_counts": result.split_counts.as_dict(),
            "random_seed": result.random_seed,
            "stratified": result.stratified,
            "label_distribution": result.label_distribution,
        }
        path = self._report_dir / "split_report.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.debug(f"Split report written: {path}")
