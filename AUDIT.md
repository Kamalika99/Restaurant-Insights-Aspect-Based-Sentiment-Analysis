# Project Audit

This project was rebuilt from an earlier prototype. Before writing any
new code, the original prototype (three Jupyter notebooks, a 10k-row
review dataset, and five saved model files) was fully audited. Nothing
below was assumed — every claim is either something I directly observed
in the original code/outputs, or a value I computed directly against the
data/models with `pandas`/`scikit-learn`/`joblib`. The original notebooks
are not included in this repository (superseded by `src/`); their
content is summarized here for the record.

## A. Dataset
- 10,000 rows, columns: `Restaurant, Reviewer, Review, Rating, Metadata, Time, Pictures`, plus one malformed stray column from a raw-CSV parsing artifact (dropped).
- Missing values: `Review` 45, `Rating`/`Reviewer`/`Metadata`/`Time` 38 each.
- 635 duplicate review texts — deduplicated before any train/test split.
- `Rating` was a mixed string column: whole stars, half-stars, and one invalid literal value `'Like'` (dropped; half-stars rounded).
- 100 unique restaurants.
- **No aspect-level or aspect-category labels exist anywhere in this dataset.** This is why ABSA evaluation in this project is qualitative, not a benchmark F1 number.

## B. The original classical-ML baseline
- The original prototype trained NLTK-stemmed CountVectorizer features into several classical models (Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost), reporting **accuracy only** — no F1/precision/recall/confusion-matrix numbers anywhere, despite a confusion-matrix heatmap being plotted.
- **Reproducibility bug found:** the saved Logistic Regression model depended on a `CountVectorizer` that was never itself saved, making the model unusable standalone.
- **Bigger bug found:** several other saved model files (`ml_model.pkl`, `tfidf.pkl`, `label.pkl`, `tokenizer.pkl`, `dl_model.h5`) had **no corresponding training code anywhere in the three notebooks**. Loading them directly showed a TF-IDF vocabulary containing words like `"hotel"` and `"nights"`, and label classes `['Bad', 'Good', 'Netral']` (misspelled) — these were leftover artifacts from an **unrelated hotel-review project**, not this dataset. They were removed from this repository rather than carried forward.

## C. The claimed 93.2% F1 (original resume description)
- The original notebook fine-tuned BERT then DeBERTa-v3-base for 3-class sentiment (1–2★→Negative, 3★→Neutral, 4–5★→Positive) and its final saved cell output read `F1 Score: 93.218`.
- This number could not be trusted as a real result:
  1. Per-epoch losses stepped down in a perfectly linear pattern (exactly −0.125/epoch) for both training and validation loss, in both the BERT and DeBERTa runs — real training loss doesn't behave this smoothly.
  2. One cell contained a Python **syntax error** in the exact line that would have produced an F1 number (`90.f1_score_func(...)`), which cannot execute, yet showed a clean numeric output.
  3. The cell just before final evaluation tried to load weights from a literal, unfilled placeholder path (`Models/<<INSERT MODEL NAME HERE>>.model`), which would raise `FileNotFoundError` if actually run.
  4. The model's classification head was instantiated with `num_labels=5` but trained against 3-class labels — an internal inconsistency.
- **This number was retired.** It has been replaced with a real, verified result — see "Fixes applied" below and `README.md`.

## D. The ABSA pipeline (original prototype)
- The original prototype loaded `yangheng/deberta-v3-base-absa-v1.1` (the model this project's resume description names) and then **never called it**. The actual sentiment step routed through an unrelated QA model (`distilbert-base-uncased-distilled-squad`) plus a generic binary sentiment model (`distilbert-base-uncased-finetuned-sst-2-english`) — neither of which is aspect-aware.
- **Domain mismatch:** the aspect vocabulary was hotel-domain (`room, service, location, price, food`), with hard-coded synthetic hotel reviews, and the pipeline's data source was a Google-Drive hotel CSV not present in this repository — meaning the ABSA notebook was never actually run against restaurant data at all.
- Aspect extraction itself (NLTK POS-tagging → spaCy word-vector similarity) was legitimately implemented as described, just pointed at the wrong domain and never fed into the right sentiment model.

## Fixes applied in this rebuild
1. Restaurant-specific aspect vocabulary (`food, service, ambience, price, hygiene, menu, portion`) replaces the hotel-domain one.
2. `src/sentiment.py` actually calls `yangheng/deberta-v3-base-absa-v1.1` as `(context, aspect)` pairs — **verified working** on real restaurant reviews with correctly separated per-aspect sentiment (see README "Example predictions").
3. `src/absa_pipeline.py` runs against this repository's real data, with both "full review" and "relevant sentence" context modes available.
4. `src/rating_classifier.py` saves the TF-IDF vectorizer and classifier together as one `Pipeline`, fixing the missing-vectorizer bug. Verified result: 84.25% accuracy / 66.53% macro F1 / 81.17% weighted F1 on a stratified, deduplicated, leakage-checked split.
5. `src/transformer_rating_classifier.py` replaces the original hand-rolled, buggy training loop with a Hugging Face `Trainer`-based run with real, auditable per-epoch logging. Verified result: **87.83% accuracy / 77.38% macro F1 / 87.08% weighted F1** — legitimately beating the classical baseline, and the number that replaces the unverifiable 93.2% claim.
   - Debugging note: the first fine-tuning attempts produced `nan` validation loss and a model frozen on one constant class prediction. Root cause: DeBERTa-v3's disentangled attention combined with the `sdpa` attention backend some `transformers` versions default to, causing numerical overflow in the first few training steps. Fix: load with `torch_dtype=torch.float32, attn_implementation="eager"` and a lower learning rate (1e-5) with warmup — already applied in the committed code.
6. Orphaned hotel-domain model artifacts and the unverifiable 93.2% claim were removed/retired rather than silently carried forward.
