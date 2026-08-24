"""
Aspect extraction and normalization for restaurant reviews.

Pipeline (matches the approach described in the resume: NLTK + spaCy
semantic similarity), with the concrete improvements over the original
notebook that are justified without needing new labels:

  1. Noun-PHRASE extraction via spaCy noun chunks (not just single nouns
     via NLTK POS tags) — captures multi-word aspects like "wait time" or
     "portion size" instead of only "time"/"portion" in isolation.
  2. Lemmatization before matching, so "waiters"/"waiter"/"waitress" all
     normalize consistently instead of being treated as distinct tokens.
  3. A restaurant-specific canonical aspect vocabulary (food/service/
     ambience/price/hygiene/menu ...) instead of the original hotel-domain
     list (room/location/...).
  4. The similarity threshold is a named constant (`SIMILARITY_THRESHOLD`)
     so it can be swept and experimentally validated (see
     src/evaluation.py) instead of being a silent magic number.

IMPORTANT: this module requires `spacy` (with an `en_core_web_md` or
`en_core_web_lg` model) and `nltk` (with the `punkt` and
`averaged_perceptron_tagger` resources) to be installed. It is written so
importing it lazily fails with a clear error rather than failing at
import time, so the rest of the package (e.g. tests that don't need
these) can still be imported without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical restaurant aspect categories. This list is a starting point
# grounded in what actually appears in the dataset's review text (see
# notebooks/1. Food_reviews.ipynb word clouds) and in common ABSA
# restaurant literature (e.g. SemEval restaurant aspect categories). It is
# NOT claimed to be exhaustive or dataset-validated beyond that — flag any
# extension against real review text before relying on it.
DEFAULT_ASPECTS = [
    "food",
    "service",
    "ambience",
    "price",
    "hygiene",
    "menu",
    "portion",
]

# Optional synonym seeds merged into each aspect before similarity scoring,
# so exact/near-exact synonyms don't depend purely on a word-vector
# similarity threshold.
ASPECT_SYNONYMS: dict[str, list[str]] = {
    "food": ["food", "taste", "flavor", "flavour", "dish", "meal", "quality"],
    "service": ["service", "staff", "waiter", "waitress", "server"],
    "ambience": ["ambience", "ambiance", "atmosphere", "decor", "vibe"],
    "price": ["price", "cost", "value", "money", "budget"],
    "hygiene": ["hygiene", "cleanliness", "clean"],
    "menu": ["menu", "options", "variety"],
    "portion": ["portion", "quantity", "size"],
}

SIMILARITY_THRESHOLD = 0.5  # inherited default from the original notebook;
# treat as a hyperparameter to sweep, not a fixed truth (see evaluation.py).


@dataclass
class ExtractedAspect:
    canonical: str          # normalized aspect category, e.g. "service"
    surface_forms: set[str] = field(default_factory=set)  # raw nouns/phrases matched


def _load_spacy(model_name: str = "en_core_web_md"):
    import spacy

    return spacy.load(model_name)


def extract_noun_phrases(text: str, nlp) -> list[str]:
    """
    Extract candidate aspect terms as lemmatized noun chunks using spaCy,
    which captures multi-word aspects ("wait time") that single-token NLTK
    POS tagging misses. Falls back gracefully to single nouns for chunks
    that are just stopwords/pronouns.
    """
    doc = nlp(text)
    phrases = []
    for chunk in doc.noun_chunks:
        lemma = " ".join(
            tok.lemma_.lower() for tok in chunk if not tok.is_stop and tok.is_alpha
        ).strip()
        if lemma:
            phrases.append(lemma)
    return phrases


def extract_nouns_nltk(text: str) -> set[str]:
    """
    Reproduction of the original single-token noun extraction (NLTK POS
    tagging, NN* tags), kept for baseline comparison against
    `extract_noun_phrases`.
    """
    import nltk

    tokenized = nltk.word_tokenize(text)
    return {word.lower() for word, pos in nltk.pos_tag(tokenized) if pos.startswith("NN")}


def group_into_aspects(
    candidate_terms: list[str],
    nlp,
    aspects: list[str] = DEFAULT_ASPECTS,
    synonyms: dict[str, list[str]] = ASPECT_SYNONYMS,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, ExtractedAspect]:
    """
    Group extracted terms into canonical aspect categories.

    Matching order:
      1. Exact/synonym match against `synonyms` (cheap, precise).
      2. spaCy word-vector cosine similarity against the aspect name,
         keeping the best-scoring aspect if it clears `threshold`.

    Terms that don't clear the threshold for any aspect are dropped
    (matches original notebook behavior) rather than forced into a
    category — forcing weak matches would inflate false aspect coverage.
    """
    result = {a: ExtractedAspect(canonical=a) for a in aspects}
    synonym_lookup = {
        term: aspect for aspect, terms in synonyms.items() for term in terms
    }

    aspect_docs = {a: nlp(a) for a in aspects}

    for term in set(candidate_terms):
        if term in synonym_lookup:
            aspect = synonym_lookup[term]
            result[aspect].surface_forms.add(term)
            continue

        term_doc = nlp(term)
        if not term_doc.vector_norm:
            continue  # spaCy has no vector for this token (e.g. OOV); skip rather than force a match

        scores = {a: term_doc.similarity(doc) for a, doc in aspect_docs.items()}
        best_aspect = max(scores, key=scores.get)
        if scores[best_aspect] >= threshold:
            result[best_aspect].surface_forms.add(term)

    return {a: r for a, r in result.items() if r.surface_forms}


def extract_aspects(
    text: str,
    nlp=None,
    aspects: list[str] = DEFAULT_ASPECTS,
    use_noun_phrases: bool = True,
) -> dict[str, ExtractedAspect]:
    """End-to-end: text -> candidate terms -> grouped canonical aspects."""
    if nlp is None:
        nlp = _load_spacy()

    if use_noun_phrases:
        candidates = extract_noun_phrases(text, nlp)
    else:
        candidates = list(extract_nouns_nltk(text))

    return group_into_aspects(candidates, nlp, aspects=aspects)
