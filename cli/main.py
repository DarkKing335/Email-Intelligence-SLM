"""CLI Entrypoint for the Email Data Engineering Pipeline.

Provides interactive and scriptable commands to run the pipeline,
inspect dataset versions, and compare builds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from email_data_engineering.pipeline import DataEngineeringPipeline
from email_data_engineering.infrastructure.storage import DatasetStorage
from email_data_engineering.domain.taxonomy import LabelTaxonomy
from email_data_engineering.application.importer import DataImporter
from email_data_engineering.application.normalizer import LabelNormalizer
from email_data_engineering.application.cleaner import DataCleaner
from email_data_engineering.application.augmentor import DataAugmentor
from email_data_engineering.application.splitter import DataSplitter


console = Console()


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the custom YAML configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config: Path | None) -> None:
    """Email Intelligence SLM — Dataset Engineering Module (Epic 1)."""
    ctx.ensure_object(dict)
    # Default config resolve
    if config is None:
        config = Path("configs/data_engineering.yaml")
    ctx.obj["CONFIG_PATH"] = config
    
    # Pre-load config to fetch directories etc
    if config.exists():
        with config.open("r", encoding="utf-8") as f:
            ctx.obj["CONFIG"] = yaml.safe_load(f) or {}
    else:
        ctx.obj["CONFIG"] = {}


# ── Pipeline command ──────────────────────────────────────────────────────────

@cli.command(name="pipeline")
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Override the raw source dataset path.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    help="Force a specific target version string (e.g. '0.1.0').",
)
@click.option(
    "--seed",
    type=int,
    help="Override the random seed for augmentation and splits.",
)
@click.pass_context
def run_pipeline(
    ctx: click.Context,
    source: Path | None,
    version: str | None,
    seed: int | None,
) -> None:
    """Run the entire 8-stage pipeline end-to-end."""
    config_path = ctx.obj["CONFIG_PATH"]
    
    console.print(f"[bold blue]Starting Data Engineering Pipeline using config: {config_path}[/]")
    
    pipeline = DataEngineeringPipeline(config_path=config_path)
    result = pipeline.run(
        source_override=source,
        version_override=version,
        seed_override=seed,
    )

    if result.success:
        console.print(f"[bold green][OK] Pipeline Succeeded! Version: {result.version}[/]")
        # Print summary table of splits
        if result.split_result:
            counts = result.split_result.split_counts
            table = Table(title="Record Counts per Split")
            table.add_column("Split", style="cyan")
            table.add_column("Record Count", style="magenta")
            table.add_row("Train", str(counts.train))
            table.add_row("Validation", str(counts.val))
            table.add_row("Test", str(counts.test))
            table.add_row("Total", str(counts.total))
            console.print(table)
    else:
        console.print(f"[bold red]✘ Pipeline Failed: {result.error_message}[/]")
        raise click.Abort()


# ── Step commands for debugging / individual runs ────────────────────────────

@cli.command(name="import")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Raw dataset file path (.csv or .jsonl)",
)
@click.pass_context
def import_cmd(ctx: click.Context, source: Path) -> None:
    """US-1.1: Import and validate raw dataset."""
    cfg = ctx.obj["CONFIG"]
    import_cfg = cfg.get("source", {})
    validation_cfg = cfg.get("validation", {})
    dirs_cfg = cfg.get("directories", {})

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
    result = importer.run(source)
    console.print(f"[green]✔ Ingested {result.total_rows_read} rows from {source.name}[/]")
    console.print(f"  Valid: {result.valid_records}, Rejected: {result.rejected_records}")
    console.print("  Reports written to reports/import_report.json and reports/rejection_report.json")


@cli.command(name="normalize")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Raw CSV/JSONL path to load, import, then normalize.",
)
@click.pass_context
def normalize_cmd(ctx: click.Context, source: Path) -> None:
    """US-1.2: Normalize labels to the canon taxonomy."""
    cfg = ctx.obj["CONFIG"]
    import_cfg = cfg.get("source", {})
    validation_cfg = cfg.get("validation", {})
    dirs_cfg = cfg.get("directories", {})
    norm_cfg = cfg.get("normalization", {})

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
    import_res = importer.run(source)

    taxonomy = LabelTaxonomy(taxonomy_path=norm_cfg.get("taxonomy_file"))
    normalizer = LabelNormalizer(
        taxonomy=taxonomy,
        unknown_label_strategy=norm_cfg.get("unknown_label_strategy", "reject"),
        default_label=norm_cfg.get("default_label", "fyi"),
        report_dir=dirs_cfg.get("reports", "reports"),
    )
    result = normalizer.run(import_res.records)
    console.print(f"[green]✔ Label normalization complete. Mapped: {result.normalized_records}, Rejected: {result.rejected_unknown_label}[/]")
    console.print("  Report written to reports/normalization_report.json")


@cli.command(name="clean")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Raw CSV/JSONL path to run through Import -> Normalize -> Clean.",
)
@click.pass_context
def clean_cmd(ctx: click.Context, source: Path) -> None:
    """US-1.3: Clean, deduplicate, and redact dataset."""
    cfg = ctx.obj["CONFIG"]
    import_cfg = cfg.get("source", {})
    validation_cfg = cfg.get("validation", {})
    dirs_cfg = cfg.get("directories", {})
    norm_cfg = cfg.get("normalization", {})
    clean_cfg = cfg.get("cleaning", {})
    dedup_cfg = clean_cfg.get("deduplication", {})
    text_cfg = clean_cfg.get("text_normalization", {})
    pii_cfg = clean_cfg.get("pii_redaction", {})
    length_cfg = clean_cfg.get("length", {})

    # Import -> Normalize -> Clean
    import_res = DataImporter(
        column_mapping=import_cfg.get("column_mapping"),
        chunk_size=import_cfg.get("chunk_size", 10_000),
        encoding=import_cfg.get("encoding", "utf-8"),
        required_fields=validation_cfg.get("required_fields"),
        max_body_chars=validation_cfg.get("max_body_chars", 8000),
        min_body_chars=validation_cfg.get("min_body_chars", 10),
        max_subject_chars=validation_cfg.get("max_subject_chars", 500),
        report_dir=dirs_cfg.get("reports", "reports"),
    ).run(source)

    norm_res = LabelNormalizer(
        taxonomy=LabelTaxonomy(taxonomy_path=norm_cfg.get("taxonomy_file")),
        unknown_label_strategy=norm_cfg.get("unknown_label_strategy", "reject"),
        default_label=norm_cfg.get("default_label", "fyi"),
        report_dir=dirs_cfg.get("reports", "reports"),
    ).run(import_res.records)

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
    result = cleaner.run(norm_res.records)
    console.print(f"[green]✔ Cleaning stage finished. Output records: {result.output_records}[/]")
    console.print(f"  Exact dups removed: {result.exact_duplicates_removed}, Near dups removed: {result.near_duplicates_removed}")
    console.print("  Report written to reports/cleaning_report.json")


@cli.command(name="augment")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Raw CSV/JSONL path to run through Clean -> Augment.",
)
@click.option("--seed", type=int, default=42, help="Seed for augmentation")
@click.pass_context
def augment_cmd(ctx: click.Context, source: Path, seed: int) -> None:
    """US-1.4: Augment dataset using rule-based strategies."""
    cfg = ctx.obj["CONFIG"]
    dirs_cfg = cfg.get("directories", {})
    aug_cfg = cfg.get("augmentation", {})
    strat_cfg = aug_cfg.get("strategies", {})
    syn_cfg = strat_cfg.get("synonym_substitution", {})
    fmt_cfg = strat_cfg.get("formatting_variation", {})
    case_cfg = strat_cfg.get("case_variation", {})
    ws_cfg = strat_cfg.get("whitespace_variation", {})

    # Mock Clean chain for speed
    import_res = DataImporter(
        column_mapping=cfg.get("source", {}).get("column_mapping"),
        report_dir=dirs_cfg.get("reports", "reports"),
    ).run(source)
    norm_res = LabelNormalizer(
        taxonomy=LabelTaxonomy(taxonomy_path=cfg.get("normalization", {}).get("taxonomy_file")),
        report_dir=dirs_cfg.get("reports", "reports"),
    ).run(import_res.records)
    clean_res = DataCleaner(
        report_dir=dirs_cfg.get("reports", "reports"),
    ).run(norm_res.records)

    augmentor = DataAugmentor(
        random_seed=seed,
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
    result = augmentor.run(clean_res.records)
    console.print(f"[green]✔ Augmentation finished. Original count: {result.original_records}, Augmented count: {result.augmented_records}[/]")
    console.print("  Report written to reports/augmentation_report.json")


# ── Metadata and exploration commands ─────────────────────────────────────────

@cli.command(name="list-versions")
@click.pass_context
def list_versions(ctx: click.Context) -> None:
    """US-1.5: List all available versioned datasets in processed directory."""
    cfg = ctx.obj["CONFIG"]
    processed_dir = cfg.get("directories", {}).get("processed", "data/processed")
    
    storage = DatasetStorage(processed_dir)
    versions = storage.list_versions()

    if not versions:
        console.print("[yellow]No processed dataset versions found.[/]")
        return

    table = Table(title="Available Dataset Versions")
    table.add_column("Version", style="cyan")
    table.add_column("Train Count", style="magenta")
    table.add_column("Val Count", style="magenta")
    table.add_column("Test Count", style="magenta")
    table.add_column("Created At", style="green")

    for v in versions:
        try:
            manifest = storage.read_version_manifest(v)
            counts = manifest.get("split_counts", {})
            created = manifest.get("created_at", "N/A")
            # Parse timestamp for clean display
            if created != "N/A":
                created = created.split("T")[0]
            table.add_row(
                v,
                str(counts.get("train", 0)),
                str(counts.get("val", 0)),
                str(counts.get("test", 0)),
                created,
            )
        except Exception:
            table.add_row(v, "Error", "Error", "Error", "N/A")

    console.print(table)


@cli.command(name="show-version")
@click.argument("version", type=str)
@click.pass_context
def show_version(ctx: click.Context, version: str) -> None:
    """US-1.5: Show full metadata manifest for a specific version."""
    cfg = ctx.obj["CONFIG"]
    processed_dir = cfg.get("directories", {}).get("processed", "data/processed")
    
    storage = DatasetStorage(processed_dir)
    try:
        manifest = storage.read_version_manifest(version)
        console.print_json(data=manifest)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Version {version} manifest not found.")
        raise click.Abort()


@cli.command(name="compare-versions")
@click.argument("version_a", type=str)
@click.argument("version_b", type=str)
@click.pass_context
def compare_versions(ctx: click.Context, version_a: str, version_b: str) -> None:
    """US-1.5: Diff two dataset versions."""
    cfg = ctx.obj["CONFIG"]
    processed_dir = cfg.get("directories", {}).get("processed", "data/processed")
    
    storage = DatasetStorage(processed_dir)
    try:
        diff = storage.compare_versions(version_a, version_b)
        console.print_json(data=diff)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise click.Abort()


@cli.command(name="stats")
@click.option(
    "--version",
    "-v",
    type=str,
    help="Version to inspect. If not specified, inspects the latest version.",
)
@click.pass_context
def stats(ctx: click.Context, version: str | None) -> None:
    """Display label distributions and stats for a dataset version."""
    cfg = ctx.obj["CONFIG"]
    processed_dir = cfg.get("directories", {}).get("processed", "data/processed")
    
    storage = DatasetStorage(processed_dir)
    if version is None:
        version = storage.get_latest_version()
        if not version:
            console.print("[yellow]No processed dataset versions found.[/]")
            return
    
    try:
        manifest = storage.read_version_manifest(version)
        dist = manifest.get("label_distribution", {})
        
        table = Table(title=f"Label Distribution for Dataset Version {version}")
        table.add_column("Canonical Label", style="cyan")
        table.add_column("Train Count", style="magenta")
        table.add_column("Val Count", style="magenta")
        table.add_column("Test Count", style="magenta")
        table.add_column("Total", style="bold green")

        # Get unique label set across splits
        all_labels = set()
        for split_dist in dist.values():
            all_labels.update(split_dist.keys())

        for label in sorted(all_labels):
            tr = dist.get("train", {}).get(label, 0)
            va = dist.get("val", {}).get(label, 0)
            te = dist.get("test", {}).get(label, 0)
            table.add_row(label, str(tr), str(va), str(te), str(tr + va + te))
            
        console.print(table)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Version {version} manifest not found.")
        raise click.Abort()


if __name__ == "__main__":
    cli(obj={})
