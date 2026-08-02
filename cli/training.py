"""CLI for Epic 2: SLM Training Pipeline.

Commands:
    email-train prepare   — Convert Epic 1 JSONL to ChatML SFT format
    email-train run       — Run LoRA fine-tuning job
    email-train evaluate  — Evaluate a trained adapter on test split
    email-train export    — Register adapter to MLOps model registry
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger


@click.group()
@click.version_option(version="0.1.0", prog_name="email-train")
def cli() -> None:
    """AI Email Intelligence SLM — Training Pipeline (Epic 2)."""


# ── prepare ────────────────────────────────────────────────────────────────────

@cli.command("prepare")
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
    help="Directory containing Epic 1 versioned dataset output.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/training"),
    show_default=True,
    help="Directory to write ChatML SFT-format JSONL files.",
)
@click.option(
    "--version",
    "-v",
    default="v0.1.0",
    show_default=True,
    help="Dataset version to load (e.g. v0.1.0).",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Cap training split at N samples (useful for quick tests).",
)
@click.option("--seed", type=int, default=42, show_default=True)
def prepare(
    input_dir: Path,
    output_dir: Path,
    version: str,
    max_samples: int | None,
    seed: int,
) -> None:
    """Convert Epic 1 JSONL dataset to ChatML instruction format for SFT training."""
    from email_training import DataPrepConfig
    from email_training.data_prep import prepare_all

    cfg = DataPrepConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        version=version,
        max_samples=max_samples,
        seed=seed,
    )

    click.echo(f"Preparing training data from {input_dir / version} → {output_dir}")
    counts = prepare_all(cfg)

    for split, count in counts.items():
        click.echo(f"  ✔ {split}: {count} samples → {output_dir / split}.jsonl")
    click.echo(f"\nTotal: {sum(counts.values())} samples ready for training.")


# ── run ────────────────────────────────────────────────────────────────────────

@cli.command("run")
@click.option(
    "--train-data",
    type=click.Path(path_type=Path),
    default=Path("data/training/train.jsonl"),
    show_default=True,
)
@click.option(
    "--val-data",
    type=click.Path(path_type=Path),
    default=Path("data/training/val.jsonl"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("models/checkpoints"),
    show_default=True,
)
@click.option("--adapter-name", default="email-intelligence-adapter", show_default=True)
@click.option("--base-model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", show_default=True)
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--lora-r", type=int, default=16, show_default=True)
@click.option("--lora-alpha", type=int, default=16, show_default=True)
@click.option("--learning-rate", type=float, default=2e-4, show_default=True)
@click.option("--batch-size", type=int, default=2, show_default=True)
@click.option("--grad-accum", type=int, default=4, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--tracking-backend",
    type=click.Choice(["local", "mlflow"]),
    default="local",
    show_default=True,
)
def run(
    train_data: Path,
    val_data: Path,
    output_dir: Path,
    adapter_name: str,
    base_model: str,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    seed: int,
    tracking_backend: str,
) -> None:
    """Run LoRA fine-tuning on the prepared training data.

    Requires GPU and: pip install unsloth trl datasets
    """
    from email_training import TrainingConfig
    from email_training.trainer import run_training

    cfg = TrainingConfig(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        output_dir=output_dir,
        adapter_name=adapter_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        seed=seed,
        train_data=train_data,
        val_data=val_data,
        tracking_backend=tracking_backend,
    )

    click.echo(f"Starting training: {base_model} → LoRA r={lora_r}, alpha={lora_alpha}")
    click.echo(f"Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}×{grad_accum}")

    try:
        adapter_path = run_training(cfg)
        click.echo(f"\n✔ Training complete! Adapter saved to: {adapter_path}")
        click.echo("Run 'email-train evaluate' to assess the adapter quality.")
        click.echo("Run 'email-train export' to register it to the model registry.")
    except ImportError as exc:
        click.echo(f"\n✗ Missing GPU dependencies: {exc}", err=True)
        raise SystemExit(1) from exc


# ── evaluate ───────────────────────────────────────────────────────────────────

@cli.command("evaluate")
@click.option(
    "--adapter",
    "adapter_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to trained adapter directory.",
)
@click.option(
    "--test-data",
    type=click.Path(path_type=Path),
    default=Path("data/training/test.jsonl"),
    show_default=True,
)
@click.option("--max-new-tokens", type=int, default=512, show_default=True)
@click.option("--temperature", type=float, default=0.1, show_default=True)
@click.option(
    "--output-report",
    type=click.Path(path_type=Path),
    default=Path("reports/evaluation_report.json"),
    show_default=True,
)
def evaluate(
    adapter_path: Path,
    test_data: Path,
    max_new_tokens: int,
    temperature: float,
    output_report: Path,
) -> None:
    """Evaluate a trained adapter: JSON validity, classification & priority accuracy."""
    from email_training import EvaluatorConfig
    from email_training.evaluator import evaluate as run_eval

    cfg = EvaluatorConfig(
        adapter_path=adapter_path,
        test_data=test_data,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_report=output_report,
    )

    results = run_eval(cfg)

    click.echo("\n── Evaluation Results ──────────────────────────────")
    click.echo(f"  Total samples:         {results['total_samples']}")
    click.echo(f"  JSON validity:         {results['json_validity_pct']}%")
    click.echo(f"  Classification acc:    {results['classification_accuracy_pct']}%")
    click.echo(f"  Priority accuracy:     {results['priority_accuracy_pct']}%")
    click.echo(f"\n  Report saved to: {output_report}")


# ── export ─────────────────────────────────────────────────────────────────────

@cli.command("export")
@click.argument("adapter_path", type=click.Path(path_type=Path))
@click.option(
    "--name",
    "model_name",
    default="email-intelligence-adapter",
    show_default=True,
    help="Registry name for this model.",
)
@click.option("--dataset-version", default=None, help="Dataset version used for training.")
@click.option("--run-id", default=None, help="Experiment tracking run ID for lineage.")
@click.option(
    "--promote",
    is_flag=True,
    default=False,
    help="Automatically promote to staging after registration.",
)
def export(
    adapter_path: Path,
    model_name: str,
    dataset_version: str | None,
    run_id: str | None,
    promote: bool,
) -> None:
    """Register a trained adapter to the MLOps model registry (Epic 6)."""
    from email_training.registry_export import export_to_registry

    click.echo(f"Registering adapter: {adapter_path} → registry name: {model_name}")

    version = export_to_registry(
        adapter_path=adapter_path,
        model_name=model_name,
        dataset_version=dataset_version,
        source_run_id=run_id,
        promote_to_staging=promote,
    )

    click.echo(f"✔ Registered as version {version}")
    if promote:
        click.echo(f"✔ Promoted to STAGING")
    click.echo("\nUse 'email-mlops list-versions {model_name}' to inspect registry.")
