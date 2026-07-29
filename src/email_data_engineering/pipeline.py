"""Master Data Engineering Pipeline Orchestrator.

Orchestrates all 8 stages of the dataset lifecycle:
Import -> Validate -> Normalize -> Clean -> Augment -> Split -> Version -> Export.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from email_data_engineering.domain.models import PipelineResult
from email_data_engineering.domain.taxonomy import LabelTaxonomy
from email_data_engineering.application.importer import DataImporter
from email_data_engineering.application.normalizer import LabelNormalizer
from email_data_engineering.application.cleaner import DataCleaner
from email_data_engineering.application.augmentor import DataAugmentor
from email_data_engineering.application.splitter import DataSplitter
from email_data_engineering.application.versioner import DataVersioner
from email_data_engineering.application.exporter import DataExporter
from email_data_engineering.logging_config import configure_logging


class DataEngineeringPipeline:
    """The master pipeline runner for Epic 1.

    Orchestrates all 8 stages of the data engineering process.
    Configured via a central YAML configuration.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            # Default config path
            config_path = (
                Path(__file__).parent.parent.parent / "configs" / "data_engineering.yaml"
            )
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        
        # Configure logging using pipeline config settings
        logging_cfg = self.config.get("logging", {})
        configure_logging(
            level=logging_cfg.get("level", "INFO"),
            log_file=logging_cfg.get("log_file", "logs/data_engineering.log"),
            rotation=logging_cfg.get("rotation", "50 MB"),
            retention=logging_cfg.get("retention", "7 days"),
            json_format=logging_cfg.get("json_logs", True),
        )

    def run(
        self,
        source_override: str | Path | None = None,
        version_override: str | None = None,
        seed_override: int | None = None,
    ) -> PipelineResult:
        """Run the end-to-end data pipeline.

        Parameters
        ----------
        source_override:
            Override the raw source path specified in the config.
        version_override:
            Override the target version (e.g. force 'v1.0.0').
        seed_override:
            Override the random seed used for augmentation/splitting.

        Returns
        -------
        PipelineResult
            The outcome containing metadata from every completed stage.
        """
        start_time = time.perf_counter()
        logger.info("=== Starting Data Engineering Pipeline ===")

        # Resolve configuration values & overrides
        source_path = Path(
            source_override
            or self.config.get("source", {}).get("primary_csv")
            or "data/raw/archive/emails.csv"
        )
        random_seed = (
            seed_override
            or self.config.get("splitting", {}).get("random_seed")
            or 42
        )

        logger.info(f"Source file: {source_path}")
        logger.info(f"Random seed: {random_seed}")

        # Track steps applied
        steps_applied = []

        try:
            # ── 1. & 2. Import & Validate ─────────────────────────────────────
            import_cfg = self.config.get("source", {})
            validation_cfg = self.config.get("validation", {})
            dirs_cfg = self.config.get("directories", {})
            
            importer = DataImporter(
                column_mapping=import_cfg.get("column_mapping"),
                chunk_size=import_cfg.get("chunk_size", 10_000),
                encoding=import_cfg.get("encoding", "utf-8"),
                required_fields=validation_cfg.get("required_fields"),
                max_body_chars=validation_cfg.get("max_body_chars", 8000),
                min_body_chars=validation_cfg.get("min_body_chars", 10),
                max_subject_chars=validation_cfg.get("max_subject_chars", 500),
                report_dir=dirs_cfg.get("reports", "reports"),
            )
            import_result = importer.run(source_path)
            steps_applied.extend(["import", "validate"])

            if not import_result.records:
                raise ValueError("No valid records were imported. Pipeline aborted.")

            # ── 3. Normalize ──────────────────────────────────────────────────
            norm_cfg = self.config.get("normalization", {})
            taxonomy_path = norm_cfg.get("taxonomy_file")
            
            taxonomy = LabelTaxonomy(taxonomy_path=taxonomy_path)
            normalizer = LabelNormalizer(
                taxonomy=taxonomy,
                unknown_label_strategy=norm_cfg.get("unknown_label_strategy", "reject"),
                default_label=norm_cfg.get("default_label", "fyi"),
                report_dir=dirs_cfg.get("reports", "reports"),
            )
            normalization_result = normalizer.run(import_result.records)
            steps_applied.append("normalize")

            if not normalization_result.records:
                raise ValueError("No records survived label normalization. Pipeline aborted.")

            # ── 4. Clean ──────────────────────────────────────────────────────
            clean_cfg = self.config.get("cleaning", {})
            dedup_cfg = clean_cfg.get("deduplication", {})
            text_cfg = clean_cfg.get("text_normalization", {})
            pii_cfg = clean_cfg.get("pii_redaction", {})
            length_cfg = clean_cfg.get("length", {})

            cleaner = DataCleaner(
                remove_exact=dedup_cfg.get("remove_exact", True),
                remove_near_duplicates=dedup_cfg.get("remove_near_duplicates", True),
                near_duplicate_threshold=dedup_cfg.get("near_duplicate_threshold", 0.95),
                collapse_whitespace=text_cfg.get("collapse_whitespace", True),
                normalize_unicode_punctuation=text_cfg.get("normalize_unicode_punctuation", True),
                strip_html=text_cfg.get("strip_html", True),
                strip_signatures=text_cfg.get("strip_signatures", True),
                strip_quoted_reply=text_cfg.get("strip_quoted_reply", True),
                pii_enabled=pii_cfg.get("enabled", True),
                replacement_token=pii_cfg.get("replacement_token", "[REDACTED]"),
                pii_patterns=pii_cfg.get("patterns"),
                clip_long_bodies=length_cfg.get("clip_long_bodies", True),
                max_body_chars=validation_cfg.get("max_body_chars", 8000),
                report_dir=dirs_cfg.get("reports", "reports"),
            )
            cleaning_result = cleaner.run(normalization_result.records)
            steps_applied.append("clean")

            if not cleaning_result.records:
                raise ValueError("No records survived cleaning. Pipeline aborted.")

            # ── 5. Augment ────────────────────────────────────────────────────
            aug_cfg = self.config.get("augmentation", {})
            strat_cfg = aug_cfg.get("strategies", {})
            syn_cfg = strat_cfg.get("synonym_substitution", {})
            fmt_cfg = strat_cfg.get("formatting_variation", {})
            case_cfg = strat_cfg.get("case_variation", {})
            ws_cfg = strat_cfg.get("whitespace_variation", {})

            augmentor = DataAugmentor(
                random_seed=random_seed,
                augmentation_ratio=aug_cfg.get("augmentation_ratio", 0.5),
                synonym_enabled=syn_cfg.get("enabled", True),
                substitution_rate=syn_cfg.get("substitution_rate", 0.15),
                formatting_enabled=fmt_cfg.get("enabled", True),
                vary_greeting=fmt_cfg.get("vary_greeting", True),
                vary_closing=fmt_cfg.get("vary_closing", True),
                case_enabled=case_cfg.get("enabled", True),
                subject_cases=case_cfg.get("subject_cases"),
                whitespace_enabled=ws_cfg.get("enabled", True),
                vary_paragraph_spacing=ws_cfg.get("vary_paragraph_spacing", True),
                skip_labels=aug_cfg.get("skip_labels"),
                report_dir=dirs_cfg.get("reports", "reports"),
            )
            augmentation_result = augmentor.run(cleaning_result.records)
            steps_applied.append("augment")

            # ── 6. Split ──────────────────────────────────────────────────────
            split_cfg = self.config.get("splitting", {})
            splitter = DataSplitter(
                train_ratio=split_cfg.get("train_ratio", 0.8),
                val_ratio=split_cfg.get("val_ratio", 0.1),
                test_ratio=split_cfg.get("test_ratio", 0.1),
                random_seed=random_seed,
                stratify=split_cfg.get("stratify", True),
                report_dir=dirs_cfg.get("reports", "reports"),
            )
            split_result = splitter.run(augmentation_result.records)
            steps_applied.append("split")

            # ── 7. Version ────────────────────────────────────────────────────
            version_cfg = self.config.get("versioning", {})
            versioner = DataVersioner(
                storage_dir=dirs_cfg.get("processed", "data/processed"),
                initial_version=version_cfg.get("initial_version", "0.1.0"),
                auto_increment=version_cfg.get("auto_increment", "patch"),
                capture_source_hash=version_cfg.get("capture_source_hash", True),
                capture_config_snapshot=version_cfg.get("capture_config_snapshot", True),
                report_dir=dirs_cfg.get("reports", "reports"),
            )
            
            # Determine semantic version
            target_version = versioner.determine_version(version_override)
            steps_applied.append("version")
            steps_applied.append("export")

            # Generate DatasetVersion metadata object
            version_metadata = versioner.build_metadata(
                version=target_version,
                source_path=source_path,
                train_count=len(split_result.train_records),
                val_count=len(split_result.val_records),
                test_count=len(split_result.test_records),
                label_dist=split_result.label_distribution,
                preprocessing_steps=steps_applied,
                random_seed=random_seed,
                config_snapshot=self.config,
            )

            # ── 8. Export ─────────────────────────────────────────────────────
            exporter = DataExporter(storage_dir=dirs_cfg.get("processed", "data/processed"))
            export_paths = exporter.run(
                version=target_version,
                train_records=split_result.train_records,
                val_records=split_result.val_records,
                test_records=split_result.test_records,
                metadata=version_metadata,
            )

            duration = time.perf_counter() - start_time
            logger.info(f"=== Pipeline Succeeded (v{target_version.lstrip('v')}) in {duration:.2f}s ===")
            logger.info(f"Artifacts exported to:")
            for name, path in export_paths.items():
                logger.info(f"  - {name}: {path}")

            return PipelineResult(
                success=True,
                version=target_version,
                import_result=import_result,
                normalization_result=normalization_result,
                cleaning_result=cleaning_result,
                augmentation_result=augmentation_result,
                split_result=split_result,
                dataset_version=version_metadata,
                duration_seconds=duration,
            )

        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.error(f"=== Pipeline Failed in {duration:.2f}s ===")
            logger.exception("Pipeline execution failed")
            return PipelineResult(
                success=False,
                version="",
                error_message=str(exc),
                duration_seconds=duration,
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh) or {}
                return data
            except yaml.YAMLError as exc:
                raise ValueError(f"Error parsing pipeline YAML config at {path}: {exc}")
