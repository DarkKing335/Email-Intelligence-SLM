"""Unit tests for the LabelTaxonomy domain model (US-1.2)."""

from __future__ import annotations

import pytest

from email_data_engineering.domain.taxonomy import LabelTaxonomy, TaxonomyViolationError


def test_all_canonical_labels_valid() -> None:
    taxonomy = LabelTaxonomy()
    
    # Check that canonical labels exist
    assert "respond" in taxonomy.canonical_labels
    assert "fyi" in taxonomy.canonical_labels
    assert "archive" in taxonomy.canonical_labels
    assert "unsubscribe" in taxonomy.canonical_labels
    assert "spam" in taxonomy.canonical_labels
    assert "security" in taxonomy.canonical_labels
    assert "billing" in taxonomy.canonical_labels
    assert "support" in taxonomy.canonical_labels

    assert len(taxonomy.canonical_labels) == 8


def test_label_mappings_correct() -> None:
    taxonomy = LabelTaxonomy()

    # Raw mapped labels should resolve correctly
    assert taxonomy.normalize("verify_code") == "security"
    assert taxonomy.normalize("promotions") == "unsubscribe"
    assert taxonomy.normalize("updates") == "fyi"
    assert taxonomy.normalize("forum") == "fyi"
    assert taxonomy.normalize("social_media") == "fyi"
    assert taxonomy.normalize("spam") == "spam"

    # Edge cases - casing and whitespaces should be normalized automatically
    assert taxonomy.normalize(" VERIFY_CODE ") == "security"
    assert taxonomy.normalize("promotions") == "unsubscribe"
    assert taxonomy.normalize("Primary") == "respond"


def test_unknown_label_raises_error() -> None:
    taxonomy = LabelTaxonomy()

    with pytest.raises(TaxonomyViolationError):
        taxonomy.normalize("completely_unknown_raw_label")


def test_priority_and_response_lookups() -> None:
    taxonomy = LabelTaxonomy()

    # Check priorities
    assert taxonomy.get_priority("respond") == "high"
    assert taxonomy.get_priority("fyi") == "medium"
    assert taxonomy.get_priority("archive") == "low"
    assert taxonomy.get_priority("security") == "high"

    # Check response requirements
    assert taxonomy.get_response_required("respond") is True
    assert taxonomy.get_response_required("fyi") is False
    assert taxonomy.get_response_required("archive") is False
    assert taxonomy.get_response_required("security") is True
