# Restaurant Insight — Aspect-Based Sentiment Analysis

Analyzes restaurant reviews at a granular, per-aspect level (food, service, ambience, price, hygiene, menu, portion) instead of a single overall rating — extracting the aspects actually mentioned in a review and predicting the sentiment associated with each one.

Built and verified in two parts: a **zero-shot ABSA pipeline** (spaCy aspect extraction → pretrained DeBERTa-v3 ABSA model) and a **fine-tuned DeBERTa-v3 review-level sentiment classifier**, benchmarked against a TF-IDF + Logistic Regression baseline.

## Results

**Review-level rating sentiment** (Negative / Neutral / Positive, stratified 80/20 split, deduplicated, no train/test leakage):

| Metric | TF-IDF + Logistic Regression | DeBERTa-v3-base (fine-tuned) |
|---|---|---|
| Accuracy | 84.25% | **87.83%** |
| Macro F1 | 66.53% | **77.38%** |
| Weighted F1 | 81.17% | **87.08%** |

**Aspect-based sentiment** — verified end-to-end on a real review, correctly separating contradictory sentiment per aspect:
```
$ python infer.py "The food was delicious but the service was extremely slow and the ambience was beautiful."

Food     -> Positive (0.9954)
Service  -> Negative (0.9946)
Ambience -> Positive (0.9914)
```

## Architecture
```
review text
    │
    ▼
light cleaning (preserve punctuation/negation)  ── src/preprocessing.py
    │
    ▼
sentence splitting                              ── src/preprocessing.py
    │
    ▼
noun-phrase extraction + spaCy similarity        ── src/aspect_extraction.py
grouping into canonical restaurant aspects
    │
    ▼
relevant-context selection per aspect            ── src/absa_pipeline.py
    │
    ▼
DeBERTa-ABSA sentiment prediction                ── src/sentiment.py
(yangheng/deberta-v3-base-absa-v1.1, zero-shot,
 batched, model loaded once)
    │
    ▼
per-aspect (label, confidence) output
```
A separate pipeline (`src/rating_classifier.py`, `src/transformer_rating_classifier.py`) handles **review-level** rating sentiment as its own task, evaluated independently with standard supervised metrics (see Results above).

## Why aspect-based sentiment analysis
A single star rating tells you *whether* a customer was happy, not *why*. "3 stars" could mean great food and terrible service, or the reverse — very different, actionable feedback for a restaurant owner. ABSA answers "what did they think about *X* specifically" for each aspect mentioned, instead of collapsing everything into one label.

## Dataset
`data/Restaurant reviews.csv` — 10,000 scraped restaurant reviews across 100 restaurants. Cleaned (deduplicated, malformed rows dropped, ratings normalized) before any modeling. No aspect-level ground truth exists in this dataset .

## Installation
```bash
git clone <this-repo>
cd restaurant-insight-absa
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"
```

## Usage
Run aspect-based sentiment on a custom review:
```bash
python infer.py "The pasta was amazing but the waiter took forever."
```
Or programmatically:
```python
from src import absa_pipeline as pipe
results = pipe.analyze_review("The food was great, but the service was terrible.")
print(pipe.format_results(results))
```
Reproduce the classical rating-sentiment baseline (fast, CPU-only):
```python
from src import rating_classifier as rc
result = rc.train("data/Restaurant reviews.csv")
print(result["macro_f1"], result["weighted_f1"])
```
Reproduce the DeBERTa-v3 fine-tune (GPU recommended; ~15 min on a free Colab T4):
```python
from src import transformer_rating_classifier as trc
result = trc.train_transformer("data/Restaurant reviews.csv")
print(result["macro_f1"], result["weighted_f1"])
```

## Project structure
```
restaurant-insight-absa/
├── data/
│   └── Restaurant reviews.csv
├── src/
│   ├── preprocessing.py                # sentence splitting, transformer-safe text cleaning
│   ├── aspect_extraction.py            # spaCy noun-phrase extraction + aspect grouping
│   ├── sentiment.py                    # DeBERTa-ABSA wrapper (batched, zero-shot)
│   ├── absa_pipeline.py                # end-to-end ABSA orchestration
│   ├── rating_classifier.py            # TF-IDF + Logistic Regression baseline
│   ├── transformer_rating_classifier.py# DeBERTa-v3 fine-tuning (HF Trainer)
│   └── evaluation.py                   # metrics, context-strategy comparison, qualitative eval
├── models/
│   ├── rating_sentiment_pipeline.joblib    # trained classical baseline (vectorizer + classifier)
│   ├── rating_sentiment_metrics.json
│   └── deberta_rating_sentiment/metrics.json  # fine-tuned model's verified metrics + training log summary
├── tests/
├── infer.py                            # CLI entry point
├── requirements.txt
└── README.md
```
Note: the fine-tuned DeBERTa-v3 model weights (~370MB) are not committed to this repo — regenerate them with `transformer_rating_classifier.train_transformer()` above, or host them separately (e.g. Hugging Face Hub / Git LFS) if needed.

## Design decisions worth knowing about
- **Zero-shot ABSA, not fine-tuned:** `yangheng/deberta-v3-base-absa-v1.1` is used as-is because no aspect-level labels exist for this dataset to fine-tune against — fabricating labels to enable fine-tuning was explicitly avoided.
- **Sentence-level context per aspect, not whole-review:** each aspect is scored against the sentence(s) that actually mention it, to avoid sentiment leaking across unrelated aspects in the same review (e.g. "food was great, but service was terrible").
- **No aspect-level benchmark metric is claimed**, because none exists for this data — ABSA quality is demonstrated qualitatively (see Results above) rather than with an invented number.

