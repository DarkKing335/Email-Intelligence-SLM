"""Email Training Package — Epic 2: SLM Fine-Tuning Pipeline.

This package implements the fine-tuning pipeline for the AI Email Intelligence SLM,
using Unsloth + PEFT + TRL to train a LoRA adapter on top of Qwen2.5-7B-Instruct.

Pipeline:
    data_prep.py    → Convert Epic 1 JSONL output to ChatML SFT format
    trainer.py      → Unsloth/TRL SFTTrainer for QLoRA fine-tuning
    evaluator.py    → Post-training evaluation (classification accuracy, JSON validity)
    registry_export.py → Register trained adapter to MLOps model registry (Epic 6)

CLI:
    email-train prepare  — Prepare training data from Epic 1 output
    email-train run      — Run fine-tuning job
    email-train evaluate — Evaluate a trained adapter
    email-train export   — Register adapter to model registry
"""

from __future__ import annotations

__all__ = ["DataPrepConfig", "TrainingConfig", "EvaluatorConfig"]

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataPrepConfig:
    """Configuration for data preparation (Epic 1 JSONL → SFT format)."""

    input_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/training")
    version: str = "v0.1.0"
    split: str = "train"  # "train" | "val" | "test"
    max_samples: int | None = None
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for the LoRA fine-tuning job."""

    # Model
    base_model: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_length: int = 2048

    # LoRA hyperparameters (matching the existing adapter/adapter_config.json)
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training hyperparameters
    output_dir: Path = Path("models/checkpoints")
    adapter_name: str = "email-intelligence-adapter"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42

    # Data
    train_data: Path = Path("data/training/train.jsonl")
    val_data: Path = Path("data/training/val.jsonl")

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    mlflow_experiment: str = "slm-finetuning"
    tracking_backend: str = "local"  # "local" | "mlflow"


@dataclass
class EvaluatorConfig:
    """Configuration for post-training evaluation."""

    adapter_path: Path = Path("models/checkpoints")
    test_data: Path = Path("data/training/test.jsonl")
    max_new_tokens: int = 512
    temperature: float = 0.1
    output_report: Path = Path("reports/evaluation_report.json")
