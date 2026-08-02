"""Post-training evaluator — measure JSON validity, classification accuracy, priority accuracy.

Usage:
    email-train evaluate --adapter models/checkpoints/email-intelligence-adapter
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from email_training import EvaluatorConfig


def _parse_json_output(text: str) -> dict | None:
    """Attempt to parse model output as JSON. Returns None on failure."""
    text = text.strip()
    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _run_inference(model, tokenizer, email_text: str, max_new_tokens: int, temperature: float) -> str:
    """Run inference and return raw text output."""
    from email_training.data_prep import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(email_text=email_text)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def evaluate(cfg: EvaluatorConfig) -> dict:
    """Evaluate a trained adapter on the test split.

    Args:
        cfg: EvaluatorConfig specifying adapter path, test data, and generation params.

    Returns:
        Evaluation results dict with accuracy metrics.

    Raises:
        ImportError: If unsloth is not installed.
        FileNotFoundError: If adapter or test data is missing.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires GPU dependencies. Install with:\n"
            "  pip install unsloth"
        ) from exc

    if not cfg.adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {cfg.adapter_path}")
    if not cfg.test_data.exists():
        raise FileNotFoundError(f"Test data not found: {cfg.test_data}")

    logger.info(f"Loading adapter from: {cfg.adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(cfg.adapter_path),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # ── Load test records ──────────────────────────────────────────────────────
    test_records: list[dict] = []
    with cfg.test_data.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))

    logger.info(f"Evaluating on {len(test_records)} test samples…")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    total = len(test_records)
    json_valid = 0
    classification_correct = 0
    priority_correct = 0
    errors: list[str] = []

    for i, record in enumerate(test_records):
        conversations = record.get("conversations", [])
        # Extract ground truth from assistant turn
        ground_truth_text = next(
            (t["content"] for t in conversations if t["role"] == "assistant"), None
        )
        # Extract email text from user turn
        user_content = next(
            (t["content"] for t in conversations if t["role"] == "user"), ""
        )
        # Parse email body from user prompt
        email_lines = user_content.split("Email to analyze:")
        email_text = email_lines[-1].strip().replace("JSON Output:", "").strip()

        if not ground_truth_text:
            errors.append(f"sample_{i}: missing ground truth")
            continue

        ground_truth = _parse_json_output(ground_truth_text)
        if not ground_truth:
            errors.append(f"sample_{i}: invalid ground truth JSON")
            continue

        # Run inference
        raw_output = _run_inference(
            model, tokenizer, email_text, cfg.max_new_tokens, cfg.temperature
        )
        parsed = _parse_json_output(raw_output)

        if parsed is not None:
            json_valid += 1
            if parsed.get("classification") == ground_truth.get("classification"):
                classification_correct += 1
            if parsed.get("priority") == ground_truth.get("priority"):
                priority_correct += 1
        else:
            errors.append(f"sample_{i}: invalid JSON output: {raw_output[:100]}")

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{total}")

    # ── Compile results ────────────────────────────────────────────────────────
    valid_pct = 100 * json_valid / total if total else 0
    clf_acc = 100 * classification_correct / json_valid if json_valid else 0
    pri_acc = 100 * priority_correct / json_valid if json_valid else 0

    results = {
        "total_samples": total,
        "json_valid": json_valid,
        "json_validity_pct": round(valid_pct, 2),
        "classification_accuracy_pct": round(clf_acc, 2),
        "priority_accuracy_pct": round(pri_acc, 2),
        "errors": errors[:10],  # first 10 errors only
        "adapter_path": str(cfg.adapter_path),
    }

    cfg.output_report.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_report.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.success(
        f"Evaluation complete — "
        f"JSON valid: {valid_pct:.1f}%, "
        f"Classification: {clf_acc:.1f}%, "
        f"Priority: {pri_acc:.1f}%"
    )
    return results
