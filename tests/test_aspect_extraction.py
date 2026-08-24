"""
Tests for aspect_extraction.py.

Tests that need spaCy's word vectors (`en_core_web_md`/`lg`) are marked
and will be skipped automatically if spaCy or the model isn't installed,
so `pytest` still runs cleanly in environments without the full NLP
stack (e.g. CI without model downloads).
"""

import pytest

from src import aspect_extraction as ax


def _spacy_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_md")
        return True
    except Exception:
        return False


def test_default_aspects_are_restaurant_domain():
    """Regression test: the domain-mismatch bug (hotel aspects like
    'room'/'location') found in the original notebook must not reappear."""
    hotel_terms = {"room", "location", "bedroom", "bathroom"}
    assert not (hotel_terms & set(ax.DEFAULT_ASPECTS))
    assert "food" in ax.DEFAULT_ASPECTS
    assert "service" in ax.DEFAULT_ASPECTS


def test_synonym_lookup_covers_all_default_aspects():
    for aspect in ax.DEFAULT_ASPECTS:
        assert aspect in ax.ASPECT_SYNONYMS, f"missing synonym seed list for '{aspect}'"
        assert aspect in ax.ASPECT_SYNONYMS[aspect], f"aspect '{aspect}' should be its own synonym"


@pytest.mark.skipif(not _spacy_available(), reason="spaCy / en_core_web_md not installed")
def test_group_into_aspects_matches_synonym_exactly():
    import spacy

    nlp = spacy.load("en_core_web_md")
    grouped = ax.group_into_aspects(["staff", "waiter"], nlp)
    assert "service" in grouped
    assert {"staff", "waiter"} <= grouped["service"].surface_forms


@pytest.mark.skipif(not _spacy_available(), reason="spaCy / en_core_web_md not installed")
def test_extract_aspects_end_to_end_food_and_service():
    result = ax.extract_aspects(
        "The pasta was amazing but the waiter took forever.",
        aspects=["food", "service"],
    )
    assert "food" in result
    assert "service" in result
