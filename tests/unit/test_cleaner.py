"""Unit tests for the DataCleaner stage (US-1.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from email_data_engineering.domain.models import EmailRecord
from email_data_engineering.application.cleaner import DataCleaner


def test_dedup_removes_exact_duplicates(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Same subject",
            body_text="Same body text that is long enough.",
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        ),
        EmailRecord(
            email_id="2",
            subject="Same subject",
            body_text="Same body text that is long enough.",
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        ),
        EmailRecord(
            email_id="3",
            subject="Different subject",
            body_text="Same body text that is long enough.",
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        ),
    ]

    cleaner = DataCleaner(
        remove_exact=True,
        remove_near_duplicates=False,
        report_dir=tmp_dir / "reports",
    )
    result = cleaner.run(records)

    assert result.input_records == 3
    assert result.output_records == 2
    assert result.exact_duplicates_removed == 1


def test_dedup_removes_near_duplicates(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Meeting tomorrow",
            body_text="Hi team, don't forget our project status sync tomorrow morning at 10 AM.",
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        ),
        # Slightly altered whitespace and greeting
        EmailRecord(
            email_id="2",
            subject="Meeting Tomorrow",
            body_text="Hello team, do not forget our project status sync tomorrow morning at 10 AM.",
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        ),
        # Completely different
        EmailRecord(
            email_id="3",
            subject="Reset password OTP",
            body_text="Your verification code is: 104958. This reset token expires in 15 mins.",
            label="security",
            original_label="verify_code",
            priority="high",
            response_required=True,
        ),
    ]

    cleaner = DataCleaner(
        remove_exact=False,
        remove_near_duplicates=True,
        near_duplicate_threshold=0.78,
        report_dir=tmp_dir / "reports",
    )
    result = cleaner.run(records)

    assert result.input_records == 3
    assert result.output_records == 2
    assert result.near_duplicates_removed == 1


def test_pii_redaction(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Reset Details",
            body_text="My email is developer@company.com and phone is +1-888-555-1234. Card: 4111-2222-3333-4444. SSN: 111-22-3333. Server: 192.168.1.1",
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        )
    ]

    cleaner = DataCleaner(
        pii_enabled=True,
        replacement_token="[REDACTED]",
        report_dir=tmp_dir / "reports",
    )
    result = cleaner.run(records)

    body = result.records[0].body_text
    assert "developer@company.com" not in body
    assert "+1-888-555-1234" not in body
    assert "4111-2222-3333-4444" not in body
    assert "111-22-3333" not in body
    assert "192.168.1.1" not in body
    assert "[REDACTED]" in body
    assert result.pii_redactions == 1


def test_length_clipping(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Long email",
            body_text="A" * 1500,
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        )
    ]

    cleaner = DataCleaner(
        clip_long_bodies=True,
        max_body_chars=500,
        report_dir=tmp_dir / "reports",
    )
    result = cleaner.run(records)

    assert len(result.records[0].body_text) == 500
    assert result.bodies_clipped == 1


def test_signature_and_reply_chain_stripping(tmp_dir: Path) -> None:
    body = (
        "This is the actual important content of the email.\n"
        "\n"
        "Best regards,\n"
        "Alice Smith\n"
        "Alice Smith Consulting LLC\n"
        "\n"
        "----- Original Message -----\n"
        "From: bob@company.com\n"
        "To: alice@company.com\n"
        "Subject: Project Status\n"
        "\n"
        "Hi Alice, how is the project going?"
    )
    records = [
        EmailRecord(
            email_id="1",
            subject="Re: Project Status",
            body_text=body,
            label="fyi",
            original_label="updates",
            priority="medium",
            response_required=False,
        )
    ]

    cleaner = DataCleaner(
        strip_signatures=True,
        strip_quoted_reply=True,
        report_dir=tmp_dir / "reports",
    )
    result = cleaner.run(records)

    cleaned_body = result.records[0].body_text
    assert "This is the actual important content of the email." in cleaned_body
    assert "Best regards" not in cleaned_body
    assert "Alice Smith Consulting LLC" not in cleaned_body
    assert "Original Message" not in cleaned_body
    assert "Hi Alice" not in cleaned_body
