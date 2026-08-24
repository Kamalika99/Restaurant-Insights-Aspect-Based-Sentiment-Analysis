"""
Rating-based sentiment classification baseline.

This reproduces and FIXES a reproducibility bug found in the original
project: `notebooks/1. Food_reviews.ipynb` saved a fitted
`LogisticRegression` to `models/LogisticRegression.joblib` but never saved
the `CountVectorizer` it depended on, so the saved model could not
correctly score new text on its own. This module fits vectorizer + model
together as a single sklearn `Pipeline` and saves/loads them as one unit.

This module is intentionally scoped to RATING-based sentiment (mapping
1-2 -> Negative, 3 -> Neutral, 4-5 -> Positive, matching
notebooks/2. sentiment_analysis.ipynb's `label_encode`), which is a
review-level classification task. It is NOT aspect-level sentiment and
must not be described as such — see README "Model/results claims".

Leakage prevention:
  - Deduplicate reviews BEFORE splitting.
  - Split BEFORE fitting the TF-IDF vectorizer (fit only on train).
  - Stratified split on the label to preserve class balance.
  - Fixed random seed for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None

RANDOM_SEED = 42


def label_encode(rating: float) -> int:
    if rating in (1, 2):
        return 0  # Negative
    if rating == 3:
        return 1  # Neutral
    if rating in (4, 5):
        return 2  # Positive
    raise ValueError(f"Unexpected rating value: {rating}")


LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV and apply only the well-justified, verifiable
    cleaning steps from the original audit:
      - drop the malformed stray column if present
      - drop the single invalid 'Like' rating row
      - coerce Rating to float, round half-stars to nearest integer
      - drop missing Review/Rating rows
      - deduplicate on Review text (before any split)
    """
    df = pd.read_csv(csv_path)
    df = df[["Review", "Rating"]].copy()
    df = df[df["Rating"] != "Like"]
    df = df.dropna(subset=["Review", "Rating"])
    df["Rating"] = df["Rating"].astype(float).round()
    df = df.drop_duplicates(subset="Review", keep="first")
    df["label"] = df["Rating"].apply(label_encode)
    return df


def train(csv_path: str, test_size: float = 0.2) -> dict:
    """
    Train the TF-IDF + LogisticRegression pipeline and return a dict with
    the fitted pipeline plus a full evaluation report on a held-out test
    split (accuracy, macro F1, weighted F1, per-class precision/recall/F1,
    and confusion matrix) — filling the gap in the original notebook,
    which only ever printed accuracy.
    """
    df = load_and_clean(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Review"],
        df["label"],
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    # Explicit duplicate-leakage check between splits (belt-and-braces,
    # since dedup already happened above on the full set).
    overlap = set(X_train) & set(X_test)
    assert not overlap, f"Data leakage: {len(overlap)} reviews appear in both splits"

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    all_labels = sorted(LABEL_NAMES)  # [0, 1, 2] — always report all 3 classes,
    # even if one is absent from a particular test split (e.g. small/synthetic data).
    report = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=[LABEL_NAMES[i] for i in all_labels],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "pipeline": pipeline,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)
