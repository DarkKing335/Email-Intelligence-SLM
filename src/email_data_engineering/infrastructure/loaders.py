"""Dataset loaders for the Email Data Engineering pipeline.

Provides memory-efficient iterators over CSV and JSONL source files.
Only local file sources are supported; no network or cloud loaders.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from loguru import logger


# ── Type alias ────────────────────────────────────────────────────────────────

RawRecord = dict[str, Any]


# ── Base class ────────────────────────────────────────────────────────────────

class BaseEmailLoader:
    """Abstract base for all email loaders."""

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

    def iter_records(self) -> Generator[RawRecord, None, None]:
        """Yield raw record dicts one at a time."""
        raise NotImplementedError

    def count_records(self) -> int:
        """Return the total number of records (may require a full pass)."""
        return sum(1 for _ in self.iter_records())


# ── CSV Loader ────────────────────────────────────────────────────────────────

class CSVEmailLoader(BaseEmailLoader):
    """Load email records from a local CSV file using chunked pandas reads.

    Handles large files (1+ GB) without loading the entire file into memory.

    Parameters
    ----------
    source_path:
        Path to the CSV file.
    column_mapping:
        Maps source CSV column names to internal field names.
        Keys are internal names; values are CSV column names.
        ``None`` values mean the field is auto-generated.
    chunk_size:
        Number of rows per pandas chunk.
    encoding:
        File encoding (default ``utf-8``).
    """

    # Default column mapping that matches the raw archive emails.csv schema.
    # Override via the ``column_mapping`` parameter or YAML config.
    DEFAULT_COLUMN_MAPPING: dict[str, str | None] = {
        "email_id": None,       # auto-generated
        "thread_id": None,      # auto-generated
        "sender": "sender",
        "subject": "subject",
        "body": "body",
        "label": "category",
    }

    def __init__(
        self,
        source_path: str | Path,
        column_mapping: dict[str, str | None] | None = None,
        chunk_size: int = 10_000,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(source_path)
        self._column_mapping = column_mapping or self.DEFAULT_COLUMN_MAPPING
        self._chunk_size = chunk_size
        self._encoding = encoding

    def iter_records(self) -> Generator[RawRecord, None, None]:
        """Yield one normalised raw record dict per CSV row."""
        logger.debug(
            "Opening CSV source",
            source=str(self.source_path),
            chunk_size=self._chunk_size,
        )
        chunk_index = 0
        total_rows = 0

        try:
            for chunk in pd.read_csv(
                self.source_path,
                chunksize=self._chunk_size,
                encoding=self._encoding,
                low_memory=False,
                on_bad_lines="warn",
            ):
                chunk_index += 1
                chunk_rows = len(chunk)
                total_rows += chunk_rows
                logger.debug(
                    f"Processing CSV chunk {chunk_index} ({chunk_rows} rows, "
                    f"{total_rows} total so far)"
                )

                for _, row in chunk.iterrows():
                    yield self._map_row(row)

        except Exception as exc:
            logger.error(f"Failed to read CSV: {exc}")
            raise

        logger.debug(f"CSV load complete: {total_rows} rows across {chunk_index} chunks.")

    def _map_row(self, row: pd.Series) -> RawRecord:
        """Map a pandas Series row to a raw record dict."""
        record: RawRecord = {}

        for internal_name, csv_col in self._column_mapping.items():
            if csv_col is None:
                # Auto-generate missing identity fields
                if internal_name in ("email_id", "thread_id"):
                    record[internal_name] = str(uuid.uuid4())
                else:
                    record[internal_name] = ""
            else:
                value = row.get(csv_col, "")
                record[internal_name] = "" if pd.isna(value) else str(value).strip()

        # Derive sender_domain from sender address
        sender = record.get("sender", "")
        record["sender_domain"] = _extract_domain(sender)

        return record

    def count_records(self) -> int:
        """Count total rows without full iteration (uses pd.read_csv count)."""
        total = 0
        for chunk in pd.read_csv(
            self.source_path,
            chunksize=self._chunk_size,
            encoding=self._encoding,
            usecols=[0],  # read only first column for speed
            low_memory=False,
            on_bad_lines="skip",
        ):
            total += len(chunk)
        return total


# ── JSONL Loader ──────────────────────────────────────────────────────────────

class JSONLEmailLoader(BaseEmailLoader):
    """Load email records from a local JSONL file.

    Used to reload previously exported pipeline artifacts for further processing.

    Parameters
    ----------
    source_path:
        Path to the ``.jsonl`` file.
    """

    def iter_records(self) -> Generator[RawRecord, None, None]:
        """Yield one record dict per non-empty line."""
        logger.debug("Opening JSONL source", source=str(self.source_path))
        line_number = 0
        valid = 0
        errors = 0

        with self.source_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line_number += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    valid += 1
                    yield record
                except json.JSONDecodeError as exc:
                    errors += 1
                    logger.warning(f"Skipping malformed JSON on line {line_number}: {exc}")

        logger.debug(
            f"JSONL load complete: {valid} valid records, {errors} errors "
            f"({line_number} lines total)."
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _extract_domain(sender: str) -> str:
    """Extract the domain portion from an email address string.

    Examples
    --------
    >>> _extract_domain("Alice <alice@example.com>")
    'example.com'
    >>> _extract_domain("no-reply@mail.service.io")
    'mail.service.io'
    >>> _extract_domain("not-an-email")
    ''
    """
    if not sender:
        return ""
    # Strip display name wrapper: "Name <addr>"
    if "<" in sender and ">" in sender:
        start = sender.index("<") + 1
        end = sender.index(">")
        sender = sender[start:end].strip()
    if "@" in sender:
        return sender.split("@", 1)[-1].lower().strip()
    return ""
