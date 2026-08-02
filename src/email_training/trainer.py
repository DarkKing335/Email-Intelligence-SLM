"""Epic 2: SLM Fine-Tuning Script — Unsloth + TRL SFTTrainer.

Trains a LoRA adapter on top of Qwen2.5-7B-Instruct-bnb-4bit using the
ChatML instruction data prepared by data_prep.py.

Requires GPU and the following extra dependencies:
    pip install unsloth trl datasets

Usage (via CLI):
    email-train run --config configs/training.yaml

Usage (Python):
    from email_training.trainer import run_training
    from email_training import TrainingConfig
    run_training(TrainingConfig())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from email_training import TrainingConfig

if TYPE_CHECKING:
    pass


# ── Prompt formatting ──────────────────────────────────────────────────────────

def _format_chatml(example: dict) -> dict:
    """Format a ChatML conversation dict into a single training text string."""
    conversations = example.get("conversations", [])
    parts: list[str] = []
    for turn in conversations:
        role = turn["role"]
        content = turn["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("")  # trailing newline after last turn
    return {"text": "\n".join(parts)}


# ── Core training function ─────────────────────────────────────────────────────

def run_training(cfg: TrainingConfig) -> Path:
    """Run the full LoRA fine-tuning pipeline.

    Args:
        cfg: TrainingConfig with model, LoRA, and training hyperparameters.

    Returns:
        Path to the saved adapter directory.

    Raises:
        ImportError: If unsloth, trl, or datasets are not installed.
        FileNotFoundError: If training data files do not exist.
    """
    # ── Import GPU dependencies ────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Training requires GPU dependencies. Install with:\n"
            "  pip install unsloth trl datasets\n"
            "These are not included in the default project dependencies."
        ) from exc

    if not cfg.train_data.exists():
        raise FileNotFoundError(
            f"Training data not found: {cfg.train_data}\n"
            "Run 'email-train prepare' first."
        )

    logger.info(f"Loading base model: {cfg.base_model}")
    logger.info(f"LoRA config: r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}")

    # ── Load base model with Unsloth (4-bit) ──────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,  # auto-detect
    )

    # ── Attach LoRA adapter ────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # ── Load and format dataset ────────────────────────────────────────────────
    data_files: dict[str, str] = {"train": str(cfg.train_data)}
    if cfg.val_data.exists():
        data_files["validation"] = str(cfg.val_data)

    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(_format_chatml, remove_columns=dataset["train"].column_names)

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")

    logger.info(
        f"Dataset loaded — train: {len(train_dataset)} samples"
        + (f", val: {len(eval_dataset)} samples" if eval_dataset else "")
    )

    # ── Configure experiment tracking ─────────────────────────────────────────
    _start_tracking(cfg)

    # ── SFTTrainer configuration ───────────────────────────────────────────────
    output_dir = cfg.output_dir / cfg.adapter_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=bool(eval_dataset),
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=False,
        fp16=False,
        bf16=True,
        report_to="none",  # disable external loggers; use email_mlops tracker
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("Starting training…")
    train_result = trainer.train()

    logger.info(
        f"Training complete — loss: {train_result.training_loss:.4f}, "
        f"steps: {train_result.global_step}"
    )

    # ── Save adapter ───────────────────────────────────────────────────────────
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Write training summary
    summary = {
        "adapter_name": cfg.adapter_name,
        "base_model": cfg.base_model,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "num_train_epochs": cfg.num_train_epochs,
        "train_samples": len(train_dataset),
        "training_loss": train_result.training_loss,
        "global_step": train_result.global_step,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _end_tracking(train_result)

    logger.success(f"Adapter saved to: {output_dir}")
    return output_dir


# ── Experiment tracking helpers ────────────────────────────────────────────────

_tracker = None


def _start_tracking(cfg: TrainingConfig) -> None:
    """Start an MLOps experiment run (non-blocking — fails gracefully)."""
    global _tracker  # noqa: PLW0603
    try:
        from email_mlops import create_tracker

        _tracker = create_tracker(backend_override=cfg.tracking_backend)
        _tracker.start_run(
            run_name=f"{cfg.adapter_name}-r{cfg.lora_r}",
            experiment_name=cfg.mlflow_experiment,
            tags={"base_model": cfg.base_model, "framework": "unsloth+peft"},
        )
        _tracker.log_parameters({
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "learning_rate": cfg.learning_rate,
            "num_epochs": cfg.num_train_epochs,
            "batch_size": cfg.per_device_train_batch_size,
            "grad_accum_steps": cfg.gradient_accumulation_steps,
        })
        logger.info(f"Experiment tracking started ({cfg.tracking_backend})")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not start experiment tracking: {exc}")
        _tracker = None


def _end_tracking(train_result) -> None:
    """Log final metrics and end the experiment run."""
    if _tracker is None:
        return
    try:
        _tracker.log_metric("final_train_loss", train_result.training_loss)
        _tracker.log_metric("global_step", train_result.global_step)
        _tracker.end_run(status="FINISHED")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not finalize experiment tracking: {exc}")
