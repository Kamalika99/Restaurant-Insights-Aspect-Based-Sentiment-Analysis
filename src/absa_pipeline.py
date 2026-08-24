"""
End-to-end Aspect-Based Sentiment Analysis pipeline for restaurant reviews.

    review text
        -> light cleaning (preprocessing.clean_for_transformer)
        -> sentence splitting (preprocessing.split_sentences)
        -> per-sentence aspect extraction (aspect_extraction.extract_aspects)
        -> relevant-context selection per aspect (this module)
        -> DeBERTa-ABSA sentiment per (context, aspect) pair (sentiment.predict_batch)
        -> aggregation across duplicate/repeated aspect mentions

Two context strategies are supported and should be compared empirically
rather than assumed (see README "Baseline vs improved" and
src/evaluation.py `compare_context_strategies`):

  - "full_review": every aspect is scored against the ENTIRE review text.
    This is the baseline / naive approach — it risks sentiment
    contamination, e.g. "food was amazing, but service was terrible"
    could leak the negative service sentiment into the food score.

  - "sentence": each aspect is scored only against the sentence(s) that
    actually mention it (or a synonym/surface form of it). This is the
    candidate improvement described in the project brief.

Negation and contrast ("not good", "wasn't terrible", "although X, Y")
are NOT hand-coded here with brittle keyword rules. Instead, the design
relies on giving the pretrained ABSA model the smallest correct context
(the sentence/clause that actually contains the negation), since the
model was trained on exactly this kind of construction. Keyword-based
negation flipping was considered and rejected because it's easy to get
wrong ("never disappointed" is positive despite containing "never") and
duplicates what the model already handles from training data — this is
a design decision, not a validated experimental result, and should be
noted as such in any writeup.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import aspect_extraction as ax
from . import preprocessing as pp
from . import sentiment as sm


@dataclass
class AspectResult:
    aspect: str
    label: str
    confidence: float
    evidence: list[str]  # sentence(s) that produced this result


def _sentences_mentioning(term_variants: set[str], sentences: list[str]) -> list[str]:
    matched = []
    for sent in sentences:
        low = sent.lower()
        if any(term in low for term in term_variants):
            matched.append(sent)
    return matched


def analyze_review(
    review_text: str,
    aspects: list[str] = ax.DEFAULT_ASPECTS,
    context_mode: str = "sentence",
    nlp=None,
) -> list[AspectResult]:
    """
    Run the full pipeline on a single review and return one AspectResult
    per aspect that was actually detected in the review (aspects not
    mentioned are simply absent from the output, not defaulted to
    "Neutral" — a Neutral label should only be predicted by the model,
    never invented).
    """
    if nlp is None:
        nlp = ax._load_spacy()

    cleaned = pp.clean_for_transformer(review_text)
    sentences = pp.split_sentences(cleaned)
    if not sentences:
        return []

    detected = ax.extract_aspects(cleaned, nlp=nlp, aspects=aspects)
    if not detected:
        return []

    pairs: list[tuple[str, str]] = []
    pair_meta: list[tuple[str, list[str]]] = []  # (aspect, evidence sentences) per pair

    for aspect_name, extracted in detected.items():
        if context_mode == "full_review":
            context = cleaned
            evidence = sentences
        else:  # "sentence"
            evidence = _sentences_mentioning(extracted.surface_forms, sentences)
            if not evidence:
                # Aspect matched at the review level (e.g. via similarity on
                # a phrase spaCy split across sentence boundaries) but no
                # single sentence contains the surface form verbatim.
                # Fall back to the full review rather than silently dropping
                # a detected aspect.
                evidence = sentences
            context = " ".join(evidence)

        pairs.append((context, aspect_name))
        pair_meta.append((aspect_name, evidence))

    predictions = sm.predict_batch(pairs)

    # Aggregate: with sentence-mode context, each aspect appears once per
    # aspect already (context = all its evidence sentences joined). Kept
    # as a simple list; if an aspect were split into multiple independent
    # predictions in a future version (e.g. per-mention scoring), this is
    # the place to add majority-vote / confidence-weighted aggregation.
    results = []
    for pred, (aspect_name, evidence) in zip(predictions, pair_meta):
        results.append(
            AspectResult(
                aspect=aspect_name,
                label=pred.label,
                confidence=pred.confidence,
                evidence=evidence,
            )
        )
    return results


def format_results(results: list[AspectResult]) -> str:
    """Human-readable CLI-style output, e.g.:
        Food      -> Positive (0.97)
        Service   -> Negative (0.93)
        Ambience  -> Positive (0.88)
    """
    if not results:
        return "No aspects detected in this review."
    width = max(len(r.aspect) for r in results)
    lines = [
        f"{r.aspect.capitalize().ljust(width)} -> {r.label} ({r.confidence})"
        for r in results
    ]
    return "\n".join(lines)
