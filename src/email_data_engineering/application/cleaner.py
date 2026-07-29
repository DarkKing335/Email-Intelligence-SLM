"""Clean stage — US-1.3.

Deduplicates, redacts PII, normalizes text, strips signature blocks,
and clips body length. Emits a :class:`CleaningResult`.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from email_data_engineering.domain.models import EmailRecord, CleaningResult


class DataCleaner:
    """Clean stage: clean, deduplicate, and redact email records.

    Implements US-1.3 acceptance criteria:
    - Deduplicate exact and near-duplicates.
    - Text normalization (whitespace, unicode).
    - Signature stripping.
    - PII Redaction.
    - Length clipping.
    """

    def __init__(
        self,
        remove_exact: bool = True,
        remove_near_duplicates: bool = True,
        near_duplicate_threshold: float = 0.95,
        collapse_whitespace: bool = True,
        normalize_unicode_punctuation: bool = True,
        strip_html: bool = True,
        strip_signatures: bool = True,
        strip_quoted_reply: bool = True,
        pii_enabled: bool = True,
        replacement_token: str = "[REDACTED]",
        pii_patterns: dict[str, bool] | None = None,
        clip_long_bodies: bool = True,
        max_body_chars: int = 8_000,
        report_dir: str | Path = "reports",
    ) -> None:
        self._remove_exact = remove_exact
        self._remove_near = remove_near_duplicates
        self._near_threshold = near_duplicate_threshold
        self._collapse_whitespace = collapse_whitespace
        self._normalize_unicode = normalize_unicode_punctuation
        self._strip_html = strip_html
        self._strip_signatures = strip_signatures
        self._strip_quoted_reply = strip_quoted_reply
        self._pii_enabled = pii_enabled
        self._replacement_token = replacement_token
        self._pii_patterns = pii_patterns or {
            "email_address": True,
            "phone_number": True,
            "credit_card": True,
            "ssn": True,
            "ip_address": True,
        }
        self._clip_long_bodies = clip_long_bodies
        self._max_body_chars = max_body_chars
        self._report_dir = Path(report_dir)

        # Precompiled regex patterns
        self._regex_html = re.compile(r"<[^>]*>")
        
        # Simple signature identifiers
        self._regex_sig = re.compile(
            r"(?i)(?:^--\s*$|^\s*best regards|^\s*kind regards|^\s*sincerely|^\s*thanks\s*,\s*$|^\s*regards\s*,\s*$).*",
            re.MULTILINE | re.DOTALL
        )
        
        # Common quoted chain identifier
        self._regex_quote = re.compile(
            r"(?m)(?:^\s*On\s+.*\s+wrote:\s*$|^\s*-+\s*Original Message\s*-+\s*$|^\s*From:\s+.*@.*$).*",
            re.DOTALL
        )

        # PII Regexes
        self._regex_email = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        # Matches typical phone patterns like (123) 456-7890, 123-456-7890, +1 123 456 7890
        self._regex_phone = re.compile(r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}")
        # Matches visa, mastercard, etc (13-16 digits optionally spaced)
        self._regex_cc = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
        # SSN
        self._regex_ssn = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        # IPv4
        self._regex_ip = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

    def run(self, records: list[EmailRecord]) -> CleaningResult:
        logger.info(f"[Clean] Cleaning {len(records)} records")
        
        input_count = len(records)
        exact_dups = 0
        near_dups = 0
        pii_count = 0
        clip_count = 0
        html_stripped_count = 0
        sigs_stripped_count = 0

        # Exact deduplication mapping hash -> record
        seen_hashes: set[str] = set()
        unique_records: list[EmailRecord] = []

        # 1. Clean individual records and exact dedup
        for record in records:
            cleaned_body = record.body_text
            cleaned_subject = record.subject

            # Strip HTML
            if self._strip_html and self._regex_html.search(cleaned_body):
                cleaned_body = self._regex_html.sub(" ", cleaned_body)
                html_stripped_count += 1

            # Strip quoted reply chains
            if self._strip_quoted_reply:
                match = self._regex_quote.search(cleaned_body)
                if match:
                    cleaned_body = cleaned_body[:match.start()]

            # Strip signatures
            if self._strip_signatures:
                match = self._regex_sig.search(cleaned_body)
                if match:
                    cleaned_body = cleaned_body[:match.start()]
                    sigs_stripped_count += 1

            # Normalize text
            if self._normalize_unicode:
                cleaned_body = (
                    cleaned_body.replace("“", '"')
                    .replace("”", '"')
                    .replace("‘", "'")
                    .replace("’", "'")
                    .replace("–", "-")
                    .replace("—", "-")
                    .replace("…", "...")
                )
                cleaned_subject = (
                    cleaned_subject.replace("“", '"')
                    .replace("”", '"')
                    .replace("‘", "'")
                    .replace("’", "'")
                    .replace("–", "-")
                    .replace("—", "-")
                    .replace("…", "...")
                )

            if self._collapse_whitespace:
                cleaned_body = re.sub(r"\s+", " ", cleaned_body).strip()
                cleaned_subject = re.sub(r"\s+", " ", cleaned_subject).strip()

            # Redact PII
            redactions_in_record = 0
            if self._pii_enabled:
                if self._pii_patterns.get("email_address"):
                    cleaned_body, count = self._regex_email.subn(self._replacement_token, cleaned_body)
                    redactions_in_record += count
                if self._pii_patterns.get("phone_number"):
                    cleaned_body, count = self._regex_phone.subn(self._replacement_token, cleaned_body)
                    redactions_in_record += count
                if self._pii_patterns.get("credit_card"):
                    cleaned_body, count = self._regex_cc.subn(self._replacement_token, cleaned_body)
                    redactions_in_record += count
                if self._pii_patterns.get("ssn"):
                    cleaned_body, count = self._regex_ssn.subn(self._replacement_token, cleaned_body)
                    redactions_in_record += count
                if self._pii_patterns.get("ip_address"):
                    cleaned_body, count = self._regex_ip.subn(self._replacement_token, cleaned_body)
                    redactions_in_record += count
                if redactions_in_record > 0:
                    pii_count += 1

            # Clip long bodies
            if self._clip_long_bodies and len(cleaned_body) > self._max_body_chars:
                cleaned_body = cleaned_body[:self._max_body_chars].strip()
                clip_count += 1

            # Hash for exact dedup
            content_hash = sha256(f"{cleaned_subject}|||{cleaned_body}".encode("utf-8")).hexdigest()

            if self._remove_exact and content_hash in seen_hashes:
                exact_dups += 1
                continue

            seen_hashes.add(content_hash)
            
            # Create the intermediate record
            updated_record = record.with_updates(
                subject=cleaned_subject,
                body_text=cleaned_body,
            )
            unique_records.append(updated_record)

        # 2. Near deduplication (cosine similarity on vectorised texts)
        final_records: list[EmailRecord] = []
        if self._remove_near and len(unique_records) > 1:
            # Combine subject and body for text vectorisation
            texts = [f"{r.subject} {r.body_text}" for r in unique_records]
            
            try:
                vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Identify duplicates
                indices_to_remove = set()
                n = len(unique_records)
                for i in range(n):
                    if i in indices_to_remove:
                        continue
                    for j in range(i + 1, n):
                        if j in indices_to_remove:
                            continue
                        if similarity_matrix[i, j] >= self._near_threshold:
                            indices_to_remove.add(j)
                            near_dups += 1

                for i, record in enumerate(unique_records):
                    if i not in indices_to_remove:
                        final_records.append(record)
            except Exception as exc:
                logger.warning(f"TF-IDF Vectorisation failed during near-deduplication: {exc}. Falling back to clean set without near-dedup.")
                final_records = unique_records
        else:
            final_records = unique_records

        result = CleaningResult(
            input_records=input_count,
            output_records=len(final_records),
            exact_duplicates_removed=exact_dups,
            near_duplicates_removed=near_dups,
            pii_redactions=pii_count,
            bodies_clipped=clip_count,
            html_stripped=html_stripped_count,
            signatures_stripped=sigs_stripped_count,
            records=final_records,
        )

        self._write_report(result)
        logger.info(
            f"[Clean] Completed: {result.output_records} records remaining. "
            f"Exact dups removed: {exact_dups}, Near dups removed: {near_dups}, "
            f"PII occurrences redacted: {pii_count}, Bodies clipped: {clip_count}"
        )
        return result

    def _write_report(self, result: CleaningResult) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "stage": "clean",
            "input_records": result.input_records,
            "output_records": result.output_records,
            "exact_duplicates_removed": result.exact_duplicates_removed,
            "near_duplicates_removed": result.near_duplicates_removed,
            "pii_redactions": result.pii_redactions,
            "bodies_clipped": result.bodies_clipped,
            "html_stripped": result.html_stripped,
            "signatures_stripped": result.signatures_stripped,
        }
        path = self._report_dir / "cleaning_report.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.debug(f"Cleaning report written: {path}")
