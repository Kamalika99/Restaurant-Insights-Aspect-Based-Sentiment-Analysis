from src import preprocessing as pp


def test_clean_for_transformer_preserves_punctuation_and_negation():
    text = "The food wasn't good, but the service was great!"
    cleaned = pp.clean_for_transformer(text)
    # Punctuation/negation words must survive light cleaning — the whole point
    # of keeping this path separate from the classical-ML cleaner.
    assert "n't" in cleaned or "wasn't" in cleaned
    assert "!" in cleaned


def test_clean_for_transformer_strips_urls_and_html():
    text = "Great place <b>visit</b> https://example.com/review !"
    cleaned = pp.clean_for_transformer(text)
    assert "http" not in cleaned
    assert "<b>" not in cleaned


def test_split_sentences_basic():
    text = "The food was great. The service was slow."
    sentences = pp.split_sentences(text)
    assert len(sentences) == 2
    assert "food" in sentences[0].lower()
    assert "service" in sentences[1].lower()


def test_legacy_aggressive_clean_strips_negation_markers():
    """Documents WHY legacy_aggressive_clean is not used for transformer input:
    it destroys the apostrophe in negations."""
    text = "The service wasn't terrible."
    cleaned = pp.legacy_aggressive_clean(text)
    assert "wasn't" not in cleaned
    assert "wasn t" in cleaned or "wasnt" in cleaned.replace(" ", "")
