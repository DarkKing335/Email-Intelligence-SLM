"""Registry export: register a trained adapter to the MLOps Model Registry (Epic 6).

After training, call this to register the adapter so the deployment pipeline
(email-mlops deploy) can discover and ship it.

Usage:
    email-train export --adapter models/checkpoints/email-intelligence-adapter \\
                       --name email-intelligence-adapter \\
                       --dataset-version v0.1.0 \\
                       --run-id <experiment-run-id>
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


def export_to_registry(
    adapter_path: Path,
    model_name: str,
    dataset_version: str | None = None,
    source_run_id: str | None = None,
    base_model: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    promote_to_staging: bool = False,
) -> int:
    """Register a trained LoRA adapter to the MLOps model registry.

    Args:
        adapter_path: Directory containing the saved PEFT adapter.
        model_name: Registry name for this model (e.g. "email-intelligence-adapter").
        dataset_version: Dataset version used for training (e.g. "v0.1.0").
        source_run_id: Experiment tracking run ID for lineage.
        base_model: Base model identifier.
        promote_to_staging: If True, automatically transition to staging stage.

    Returns:
        Registered version number.

    Raises:
        FileNotFoundError: If adapter_path does not exist.
    """
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    # Load training summary if available
    metrics: dict[str, float] = {}
    summary_path = adapter_path / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "training_loss" in summary:
            metrics["train_loss"] = summary["training_loss"]
        if "global_step" in summary:
            metrics["global_step"] = float(summary["global_step"])

    from email_mlops import create_registry, RegistryStage

    registry = create_registry()

    version = registry.register_model(
        name=model_name,
        adapter_path=str(adapter_path),
        source_run_id=source_run_id,
        base_model=base_model,
        dataset_version=dataset_version,
        metrics=metrics,
        description=f"LoRA adapter fine-tuned on email triage dataset {dataset_version or ''}",
        tags={
            "task": "email-classification",
            "framework": "unsloth+peft",
            "base_model": base_model,
        },
    )

    logger.success(
        f"Registered '{model_name}' version {version} to model registry"
        + (f" [dataset: {dataset_version}]" if dataset_version else "")
    )

    if promote_to_staging:
        registry.transition_stage(model_name, version, RegistryStage.STAGING)
        logger.info(f"Promoted version {version} to STAGING")

    return version
