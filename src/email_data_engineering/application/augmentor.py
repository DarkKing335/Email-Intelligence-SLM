"""Augment stage — US-1.4.

Derives synthetic variants from existing records using synonym substitution,
greeting/closing formatting shifts, case changes, and spacing variations.
Controlled by a random seed for deterministic generation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from loguru import logger

from email_data_engineering.domain.models import EmailRecord, AugmentationResult


class DataAugmentor:
    """Augment stage: perform deterministic, rule-based data augmentation.

    Implements US-1.4 acceptance criteria:
    - Augmented records are clearly marked (`is_augmented=True`, `augmentation_type`).
    - Repeatable with a configurable seed.
    - Original records are preserved; augmented records are appended.
    - Write report to reports directory.
    """

    # Local, simple synonym mapping for email-related domains
    DEFAULT_SYNONYMS: dict[str, list[str]] = {
        "immediately": ["right away", "instantly", "without delay", "promptly"],
        "urgent": ["critical", "high-priority", "pressing"],
        "important": ["significant", "essential", "key"],
        "verification": ["auth", "security check", "confirmation"],
        "invoices": ["bills", "statements", "payment records"],
        "receipt": ["confirmation", "proof of payment"],
        "support": ["assistance", "helpdesk", "service"],
        "update": ["notification", "announcement", "news"],
        "meeting": ["session", "discussion", "gathering"],
        "customer": ["client", "user", "patron"],
        "request": ["inquiry", "appeal", "solicitation"],
        "review": ["inspect", "examine", "check over"],
    }

    # Greetings / Closings for formatting variations
    GREETINGS = [
        "Hi,", "Hello,", "Dear User,", "Hey,", "Good day,", "Hi team,", "Greetings,"
    ]
    CLOSINGS = [
        "Best regards,", "Kind regards,", "Sincerely,", "Thanks,", "Regards,", "Warmly,"
    ]

    def __init__(
        self,
        random_seed: int = 42,
        augmentation_ratio: float = 0.5,
        synonym_enabled: bool = True,
        substitution_rate: float = 0.15,
        formatting_enabled: bool = True,
        vary_greeting: bool = True,
        vary_closing: bool = True,
        case_enabled: bool = True,
        subject_cases: list[str] | None = None,
        whitespace_enabled: bool = True,
        vary_paragraph_spacing: bool = True,
        skip_labels: list[str] | None = None,
        report_dir: str | Path = "reports",
    ) -> None:
        self._seed = random_seed
        self._ratio = augmentation_ratio
        self._synonym_enabled = synonym_enabled
        self._sub_rate = substitution_rate
        self._formatting_enabled = formatting_enabled
        self._vary_greeting = vary_greeting
        self._vary_closing = vary_closing
        self._case_enabled = case_enabled
        self._subject_cases = subject_cases or ["title_case", "sentence_case"]
        self._whitespace_enabled = whitespace_enabled
        self._vary_spacing = vary_paragraph_spacing
        self._skip_labels = skip_labels or []
        self._report_dir = Path(report_dir)

    def run(self, records: list[EmailRecord]) -> AugmentationResult:
        logger.info(
            f"[Augment] Starting augmentation with seed={self._seed}, ratio={self._ratio}"
        )
        
        # Seed python's random for deterministic runs
        rng = random.Random(self._seed)

        original_count = len(records)
        strategy_counts = {
            "synonym_substitution": 0,
            "formatting_variation": 0,
            "case_variation": 0,
            "whitespace_variation": 0,
        }

        # Filter out records belonging to skipped labels
        eligible_records = [
            r for r in records if r.label not in self._skip_labels
        ]

        if not eligible_records:
            logger.warning("[Augment] No records eligible for augmentation.")
            return AugmentationResult(
                original_records=original_count,
                augmented_records=0,
                random_seed=self._seed,
                strategy_counts=strategy_counts,
                records=list(records),
            )

        # Determine how many augmented records to generate
        num_to_generate = int(original_count * self._ratio)
        logger.info(f"[Augment] Generating {num_to_generate} augmented records.")

        augmented_list: list[EmailRecord] = []
        
        # Sample base records to augment (with replacement if ratio > 1.0)
        base_samples = rng.choices(eligible_records, k=num_to_generate)

        strategies = []
        if self._synonym_enabled:
            strategies.append("synonym_substitution")
        if self._formatting_enabled:
            strategies.append("formatting_variation")
        if self._case_enabled:
            strategies.append("case_variation")
        if self._whitespace_enabled:
            strategies.append("whitespace_variation")

        if not strategies:
            logger.warning("[Augment] No augmentation strategies enabled.")
            return AugmentationResult(
                original_records=original_count,
                augmented_records=0,
                random_seed=self._seed,
                strategy_counts=strategy_counts,
                records=list(records),
            )

        for base_rec in base_samples:
            # Choose a strategy deterministically using our rng
            strategy = rng.choice(strategies)
            
            aug_subject = base_rec.subject
            aug_body = base_rec.body_text

            if strategy == "synonym_substitution":
                aug_body = self._apply_synonyms(aug_body, rng)
                strategy_counts["synonym_substitution"] += 1

            elif strategy == "formatting_variation":
                aug_body = self._apply_formatting(aug_body, rng)
                strategy_counts["formatting_variation"] += 1

            elif strategy == "case_variation":
                aug_subject = self._apply_cases(aug_subject, rng)
                strategy_counts["case_variation"] += 1

            elif strategy == "whitespace_variation":
                aug_body = self._apply_whitespace(aug_body, rng)
                strategy_counts["whitespace_variation"] += 1

            # Build the augmented record. Keep original IDs but change the email_id.
            # Retain thread_id to keep them grouped with their original if needed.
            aug_rec = base_rec.with_updates(
                email_id=f"aug-{base_rec.email_id[4:] if base_rec.email_id.startswith('aug-') else base_rec.email_id}",
                subject=aug_subject,
                body_text=aug_body,
                is_augmented=True,
                augmentation_type=strategy,
            )
            augmented_list.append(aug_rec)

        # Merge original and augmented lists
        combined_records = list(records) + augmented_list

        result = AugmentationResult(
            original_records=original_count,
            augmented_records=len(augmented_list),
            random_seed=self._seed,
            strategy_counts=strategy_counts,
            records=combined_records,
        )

        self._write_report(result)
        logger.info(
            f"[Augment] Done. Original: {original_count}, Augmented: {len(augmented_list)}, "
            f"Total output: {len(combined_records)}"
        )
        return result

    # ── Augmentation operations ───────────────────────────────────────────────

    def _apply_synonyms(self, body: str, rng: random.Random) -> str:
        words = body.split()
        for idx, word in enumerate(words):
            # Strip punctuation for matching
            clean_word = word.lower().strip(".,!?;:()[]\"'")
            if clean_word in self.DEFAULT_SYNONYMS:
                if rng.random() <= self._sub_rate:
                    syn = rng.choice(self.DEFAULT_SYNONYMS[clean_word])
                    # Preserve case roughly
                    if word.istitle():
                        syn = syn.title()
                    elif word.isupper():
                        syn = syn.upper()
                    # Re-attach punctuation
                    start_punct = ""
                    end_punct = ""
                    for char in word:
                        if char in "([\"'":
                            start_punct += char
                        else:
                            break
                    for char in reversed(word):
                        if char in ".,!?;:)]\"'":
                            end_punct = char + end_punct
                        else:
                            break
                    words[idx] = f"{start_punct}{syn}{end_punct}"
        return " ".join(words)

    def _apply_formatting(self, body: str, rng: random.Random) -> str:
        # Check if body starts with a greeting or ends with a closing.
        # We can prepend a random greeting and append a closing.
        lines = body.split("\n")
        
        if self._vary_greeting:
            greeting = rng.choice(self.GREETINGS)
            # Prepend greeting if one doesn't exist
            if lines and not any(lines[0].startswith(g) for g in ["Hi", "Hello", "Dear", "Hey"]):
                lines.insert(0, greeting)
                lines.insert(1, "")  # blank line
                
        if self._vary_closing:
            closing = rng.choice(self.CLOSINGS)
            # Append closing if one doesn't exist
            if lines and not any(lines[-1].startswith(c) for c in ["Best", "Kind", "Sincerely", "Thanks", "Regards"]):
                lines.append("")
                lines.append(closing)
                lines.append("Team Assistant")

        return "\n".join(lines)

    def _apply_cases(self, subject: str, rng: random.Random) -> str:
        case = rng.choice(self._subject_cases)
        if case == "title_case":
            return subject.title()
        elif case == "sentence_case":
            return subject.capitalize()
        elif case == "uppercase":
            return subject.upper()
        return subject

    def _apply_whitespace(self, body: str, rng: random.Random) -> str:
        if self._vary_spacing:
            # Let's double line breaks or reduce them
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            if rng.choice([True, False]):
                # Add more paragraph spacing
                return "\n\n\n".join(paragraphs)
            else:
                # Tight single-spacing spacing
                return "\n".join(paragraphs)
        return body

    def _write_report(self, result: AugmentationResult) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "stage": "augment",
            "original_records": result.original_records,
            "augmented_records": result.augmented_records,
            "total_records": result.total_records,
            "random_seed": result.random_seed,
            "strategy_counts": result.strategy_counts,
        }
        path = self._report_dir / "augmentation_report.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.debug(f"Augmentation report written: {path}")
