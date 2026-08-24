"""
Text preprocessing for Restaurant Insight ABSA.

Two preprocessing paths are intentionally kept separate, because they serve
different consumers:

1. `clean_for_transformer` — LIGHT cleaning only (strip URLs/HTML/mentions,
   normalize whitespace). Punctuation, casing, and stopwords are preserved
   because transformer models (DeBERTa/BERT) rely on that information and
   aggressive cleaning measurably hurts them. Use this before feeding text
   to the DeBERTa-ABSA model or to sentence splitting / aspect extraction.

2. `clean_for_classical_ml` — heavier cleaning (lowercase, strip
   non-alphabetic characters, remove stopwords, stem) matching the
   preprocessing that produced the classical bag-of-words baseline in
   notebooks/1. Food_reviews.ipynb. Use this only for the classical
   TF-IDF / CountVectorizer + LogisticRegression baseline, never for the
   transformer pipeline.

Per the project audit, the original ABSA notebook used aggressive
punctuation stripping (`re.sub('[^A-z0-9]', ' ', text)`) even for text
that was about to be sent to a transformer model. That is preserved here
only inside `legacy_aggressive_clean` for reference / comparison, and is
NOT used by the default pipeline.
"""

from __future__ import annotations

import re
import string

_URL_RE = re.compile(r"(https?://\S+)|(www\.\S+)")
_HTML_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def clean_for_transformer(text: str) -> str:
    """Light cleaning that preserves punctuation and casing for transformer input."""
    text = str(text)
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _NON_ASCII_RE.sub(" ", text)  # drop emojis / non-ascii noise only
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def legacy_aggressive_clean(text: str) -> str:
    """
    Reproduction of the original notebook's `preprocess()` function
    (strips ALL punctuation). Kept only for baseline comparison —
    do not feed this into a transformer model; it removes negation
    markers and contrast punctuation that the model relies on.
    """
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = re.sub(r"\S*@\S*\s?", " ", text)
    text = _NON_ASCII_RE.sub(" ", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_for_classical_ml(text: str, stopwords: set[str] | None = None, stem: bool = True) -> str:
    """
    Reproduces the notebook 1 `text_preprocessing` function:
    lowercase -> strip non-alpha -> remove stopwords -> Porter-stem -> rejoin.

    `stopwords` and stemming require NLTK (`nltk.corpus.stopwords`,
    `nltk.stem.porter.PorterStemmer`). Pass the stopword set explicitly so
    this module has no hard NLTK import dependency at module load time.
    """
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    tokens = text.lower().split()
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if stem:
        from nltk.stem.porter import PorterStemmer

        ps = PorterStemmer()
        tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter used for sentence-level aspect context
    (see src/absa_pipeline.py). Falls back to a regex splitter if NLTK's
    punkt tokenizer isn't available, so the rest of the pipeline degrades
    gracefully rather than hard-failing.
    """
    try:
        import nltk

        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        # Fallback: split on sentence-ending punctuation.
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]
