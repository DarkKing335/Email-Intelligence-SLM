"""Data preparation: convert Epic 1 JSONL output into ChatML SFT training format.

Input:  data/processed/<version>/train.jsonl  (EmailRecord JSONL from Epic 1)
Output: data/training/train.jsonl             (ChatML instruction pairs for SFTTrainer)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from loguru import logger

from email_training import DataPrepConfig

# ── System & user prompt templates ────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "You must output ONLY a valid JSON object."
)

USER_PROMPT_TEMPLATE = """Analyze the following email and extract its metadata into a strict JSON format with exactly these keys:
- "classification": You MUST choose exactly ONE label based on these strict rules:
    * "action_required": Use ONLY for system alerts, build/pipeline failures, or tasks assigned by a professor.
    * "respond": Use ONLY for personal communication, meeting requests, or collaborations that require a human reply.
    * "notification": Use for informational updates (e.g., successful training jobs, patch notes) needing no immediate action.
    * "social": Use ONLY for casual check-ins, like gym/workout schedules or diet plans.
    * "spam": Use ONLY for unsolicited promotions, sales, or discounts.

- "priority": Choose strictly based on these rules:
    * "high": Critical system failures, pipeline breaks, or urgent professor requests.
    * "medium": Normal meeting requests, training job completions, or standard notifications.
    * "low": Social chats, gym plans, game newsletters, or spam.

- "summary": (a short summary of the email)

- "entities": A strict JSON object. Only use the following keys if entities are present:
    * "people": (array of strings)
    * "date": (string, format YYYY-MM-DD or as written in text)
    * "time": (string)
    * "project_or_product": (array of strings)
    * "technology": (array of strings)
    * "misc_items": (array of strings)

- "recommended_action": (short string; "Ignore the offer." for spam; empty string "" for notifications/alerts).

- "draft": (a short reply if needed; MUST be empty string "" if classification is "notification", "spam", or "action_required" for system builds).

Email to analyze:
{email_text}

JSON Output:"""


def _label_to_classification(label: str) -> str:
    """Map Epic 1 canonical taxonomy labels to SLM classification labels."""
    mapping = {
        "security": "action_required",
        "unsubscribe": "spam",
        "fyi": "notification",
        "action": "action_required",
        "meeting": "respond",
        "promotion": "spam",
        "social": "social",
        "newsletter": "notification",
    }
    return mapping.get(label.lower(), "notification")


def _priority_from_label(label: str) -> str:
    """Infer priority from label for training examples."""
    high_labels = {"security", "action"}
    low_labels = {"unsubscribe", "social", "promotion", "newsletter"}
    if label.lower() in high_labels:
        return "high"
    if label.lower() in low_labels:
        return "low"
    return "medium"


def _build_target_json(record: dict) -> str:
    """Build expected JSON output from an Epic 1 EmailRecord."""
    label = record.get("label", "notification")
    classification = _label_to_classification(label)
    priority = record.get("priority") or _priority_from_label(label)

    subject = record.get("subject", "")
    body = record.get("body_text", "")
    summary = (body[:120].strip() + "...") if len(body) > 120 else body.strip()

    # Build entities from available fields
    entities: dict = {}
    sender_domain = record.get("sender_domain", "")
    if sender_domain:
        entities["technology"] = [sender_domain]

    target = {
        "classification": classification,
        "priority": priority,
        "summary": summary or subject,
        "entities": entities,
        "recommended_action": "Ignore the offer." if classification == "spam" else "",
        "draft": "" if classification in ("notification", "spam", "action_required") else
                 f"Thank you for your email regarding '{subject}'. I will get back to you shortly.",
    }
    return json.dumps(target, ensure_ascii=False)


def _record_to_chatml(record: dict) -> dict:
    """Convert a single EmailRecord to a ChatML conversation dict."""
    subject = record.get("subject", "")
    body = record.get("body_text", "")
    email_text = f"Subject: {subject}\n\n{body}".strip()

    user_prompt = USER_PROMPT_TEMPLATE.format(email_text=email_text)
    assistant_output = _build_target_json(record)

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ],
        "_meta": {
            "email_id": record.get("email_id", ""),
            "label": record.get("label", ""),
            "split": record.get("split", ""),
            "version": record.get("version", ""),
            "is_augmented": record.get("is_augmented", False),
        },
    }


def _iter_records(jsonl_path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare_split(
    input_path: Path,
    output_path: Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> int:
    """Convert a single JSONL split file to ChatML format.

    Args:
        input_path: Source JSONL (Epic 1 output).
        output_path: Destination JSONL (ChatML SFT format).
        max_samples: Optional limit on number of samples.
        seed: Random seed for shuffling when max_samples is set.

    Returns:
        Number of records written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(_iter_records(input_path))
    if max_samples and max_samples < len(records):
        random.seed(seed)
        records = random.sample(records, max_samples)

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            try:
                chatml = _record_to_chatml(record)
                f.write(json.dumps(chatml, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Skipping record {record.get('email_id', '?')}: {exc}")

    logger.info(f"Wrote {written} records to {output_path}")
    return written


def prepare_all(cfg: DataPrepConfig) -> dict[str, int]:
    """Prepare all splits (train, val, test) from an Epic 1 versioned dataset.

    Args:
        cfg: DataPrepConfig with input/output dirs and version.

    Returns:
        Dict mapping split name → number of records written.
    """
    version_dir = cfg.input_dir / cfg.version
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Dataset version directory not found: {version_dir}\n"
            f"Run 'email-data pipeline' first to generate the dataset."
        )

    splits = ["train", "val", "test"]
    counts: dict[str, int] = {}

    for split in splits:
        input_path = version_dir / f"{split}.jsonl"
        if not input_path.exists():
            logger.warning(f"Split file not found, skipping: {input_path}")
            continue
        output_path = cfg.output_dir / f"{split}.jsonl"
        counts[split] = prepare_split(
            input_path=input_path,
            output_path=output_path,
            max_samples=cfg.max_samples if split == "train" else None,
            seed=cfg.seed,
        )

    total = sum(counts.values())
    logger.info(
        f"Data preparation complete — {total} total samples "
        f"({', '.join(f'{k}: {v}' for k, v in counts.items())})"
    )
    return counts
