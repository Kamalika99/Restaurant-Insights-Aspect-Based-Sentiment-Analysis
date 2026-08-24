"""
Evaluation utilities.

Two distinct kinds of evaluation live here, and they must NOT be
conflated (see README "Model/results claims"):

1. Rating-based sentiment classification — has ground-truth labels
   (derived from the star rating), so real accuracy/F1/precision/recall/
   confusion-matrix metrics are legitimate. See `evaluate_rating_classifier`.

2. Aspect-level sentiment (ABSA) — this dataset has NO ground-truth
   aspect-level annotations. `run_qualitative_absa_eval` therefore does
   NOT compute or report any F1/accuracy number for ABSA. It only:
     - runs the pipeline on a fixed set of example reviews,
     - prints the extracted aspects + predicted sentiment for manual
       inspection,
     - optionally loads a MANUALLY curated CSV of (review, aspect,
       true_label) rows, if the user has created one, and reports
       agreement against that — clearly labeled as a small, manually
       curated evaluation set, never as a benchmark number.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import rating_classifier as rc


def evaluate_rating_classifier(csv_path: str) -> dict:
    """Thin wrapper around rating_classifier.train() that just returns the report."""
    result = rc.train(csv_path)
    return {
        k: v for k, v in result.items() if k != "pipeline"
    }


def compare_context_strategies(reviews: list[str], aspects=None) -> pd.DataFrame:
    """
    Run the ABSA pipeline twice per review — once with `context_mode="full_review"`
    and once with `context_mode="sentence"` — and return a side-by-side
    DataFrame so the two strategies can be manually compared on the same
    inputs, per the project brief's instruction to measure rather than
    assume the sentence-level context is better.

    This produces a comparison table, not an automatic winner: without
    ground-truth aspect labels there is no automatic metric to declare
    one strategy superior. A human should read the `evidence` column and
    judge which context produced a more sensible sentiment call,
    especially on reviews with contrastive aspects
    ("food was great but service was slow").
    """
    from . import absa_pipeline as pipe
    from . import aspect_extraction as ax

    aspects = aspects or ax.DEFAULT_ASPECTS
    nlp = ax._load_spacy()

    rows = []
    for review in reviews:
        full = pipe.analyze_review(review, aspects=aspects, context_mode="full_review", nlp=nlp)
        sent = pipe.analyze_review(review, aspects=aspects, context_mode="sentence", nlp=nlp)
        full_map = {r.aspect: r for r in full}
        sent_map = {r.aspect: r for r in sent}
        for aspect_name in set(full_map) | set(sent_map):
            rows.append(
                {
                    "review": review,
                    "aspect": aspect_name,
                    "full_review_label": full_map[aspect_name].label if aspect_name in full_map else None,
                    "full_review_conf": full_map[aspect_name].confidence if aspect_name in full_map else None,
                    "sentence_label": sent_map[aspect_name].label if aspect_name in sent_map else None,
                    "sentence_conf": sent_map[aspect_name].confidence if aspect_name in sent_map else None,
                    "agree": (
                        full_map[aspect_name].label == sent_map[aspect_name].label
                        if aspect_name in full_map and aspect_name in sent_map
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_qualitative_absa_eval(reviews: list[str], aspects=None) -> pd.DataFrame:
    """Run the ABSA pipeline on example reviews and return a table for manual inspection."""
    from . import absa_pipeline as pipe
    from . import aspect_extraction as ax

    aspects = aspects or ax.DEFAULT_ASPECTS
    nlp = ax._load_spacy()

    rows = []
    for review in reviews:
        results = pipe.analyze_review(review, aspects=aspects, nlp=nlp)
        for r in results:
            rows.append(
                {
                    "review": review,
                    "aspect": r.aspect,
                    "predicted_label": r.label,
                    "confidence": r.confidence,
                    "evidence": " | ".join(r.evidence),
                }
            )
    return pd.DataFrame(rows)


def score_against_manual_labels(labeled_csv_path: str) -> dict:
    """
    Optional: if you create a small manually-labeled CSV with columns
    [review, aspect, true_label], run the pipeline on it and report
    accuracy against YOUR manual labels. This is explicitly a small,
    manually curated evaluation set, not a benchmark dataset, and must
    be described as such in any writeup (see project audit rules).
    """
    from . import absa_pipeline as pipe
    from . import aspect_extraction as ax

    df = pd.read_csv(labeled_csv_path)
    required = {"review", "aspect", "true_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Labeled CSV must contain columns: {required}")

    nlp = ax._load_spacy()
    preds = []
    for review in df["review"].unique():
        results = {r.aspect: r.label for r in pipe.analyze_review(review, nlp=nlp)}
        for _, row in df[df["review"] == review].iterrows():
            preds.append(results.get(row["aspect"], "NotDetected"))

    y_true = df["true_label"].tolist()
    return {
        "n_manually_labeled_examples": len(df),
        "accuracy": accuracy_score(y_true, preds),
        "classification_report": classification_report(y_true, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "note": "Evaluated against a small, manually curated label set — not a public benchmark.",
    }


def measure_inference_time(reviews: list[str], n_runs: int = 1) -> dict:
    """Measure actual (not assumed) end-to-end inference time for the ABSA pipeline."""
    from . import absa_pipeline as pipe
    from . import aspect_extraction as ax

    nlp = ax._load_spacy()
    # Warm up (excludes one-time model loading from the timed measurement)
    if reviews:
        pipe.analyze_review(reviews[0], nlp=nlp)

    start = time.perf_counter()
    for _ in range(n_runs):
        for review in reviews:
            pipe.analyze_review(review, nlp=nlp)
    elapsed = time.perf_counter() - start

    total_calls = len(reviews) * n_runs
    return {
        "total_reviews_processed": total_calls,
        "total_seconds": round(elapsed, 3),
        "avg_seconds_per_review": round(elapsed / total_calls, 4) if total_calls else None,
    }
