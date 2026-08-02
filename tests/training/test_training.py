"""Unit tests for Epic 2: SLM Training Pipeline (data_prep, configs, CLI)."""

import json
from pathlib import Path
from click.testing import CliRunner

from email_training import DataPrepConfig, EvaluatorConfig, TrainingConfig
from email_training.data_prep import (
    _build_target_json,
    _label_to_classification,
    _priority_from_label,
    _record_to_chatml,
    prepare_split,
)
from cli.training import cli


def test_label_to_classification_mapping():
    assert _label_to_classification("security") == "action_required"
    assert _label_to_classification("unsubscribe") == "spam"
    assert _label_to_classification("fyi") == "notification"
    assert _label_to_classification("meeting") == "respond"
    assert _label_to_classification("social") == "social"
    assert _label_to_classification("unknown") == "notification"


def test_priority_from_label():
    assert _priority_from_label("security") == "high"
    assert _priority_from_label("action") == "high"
    assert _priority_from_label("unsubscribe") == "low"
    assert _priority_from_label("meeting") == "medium"


def test_record_to_chatml():
    record = {
        "email_id": "test-123",
        "subject": "System Alert: Server Down",
        "body_text": "The main server crashed at 14:00.",
        "label": "security",
        "priority": "high",
        "sender_domain": "corp.com",
        "split": "train",
        "version": "v0.1.0",
    }
    chatml = _record_to_chatml(record)

    assert "conversations" in chatml
    assert len(chatml["conversations"]) == 3
    assert chatml["conversations"][0]["role"] == "system"
    assert chatml["conversations"][1]["role"] == "user"
    assert "System Alert: Server Down" in chatml["conversations"][1]["content"]
    assert chatml["conversations"][2]["role"] == "assistant"

    target_json = json.loads(chatml["conversations"][2]["content"])
    assert target_json["classification"] == "action_required"
    assert target_json["priority"] == "high"


def test_prepare_split(tmp_path: Path):
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"

    records = [
        {
            "email_id": f"email-{i}",
            "subject": f"Subject {i}",
            "body_text": f"Body text {i}",
            "label": "fyi",
            "split": "train",
        }
        for i in range(5)
    ]
    with input_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    count = prepare_split(input_file, output_file)
    assert count == 5
    assert output_file.exists()

    with output_file.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 5
    assert lines[0]["conversations"][0]["role"] == "system"


def test_training_configs_defaults():
    prep_cfg = DataPrepConfig()
    assert prep_cfg.version == "v0.1.0"

    train_cfg = TrainingConfig()
    assert train_cfg.lora_r == 16
    assert train_cfg.lora_alpha == 16
    assert train_cfg.base_model == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

    eval_cfg = EvaluatorConfig()
    assert eval_cfg.temperature == 0.1


def test_email_train_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "AI Email Intelligence SLM" in result.output
    assert "prepare" in result.output
    assert "run" in result.output
    assert "evaluate" in result.output
    assert "export" in result.output
