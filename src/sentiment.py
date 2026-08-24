"""
Aspect-level sentiment scoring using the pretrained ABSA model
`yangheng/deberta-v3-base-absa-v1.1`.

This is a *zero-shot* use of a model that was already fine-tuned for
aspect-based sentiment classification (trained on SemEval-style ABSA
data) — this repository does not fine-tune it further, because no
aspect-level labeled data exists for this dataset (see the project
audit / README "Limitations"). Do not claim this model was fine-tuned
on this dataset; it was not.

Compared to the original notebook's approach (loading this model and
then never calling it, instead routing through a QA model + a generic
binary SST-2 sentiment model), this module actually uses it as intended:
the model takes a (context, aspect_term) pair and outputs a
Positive/Negative/Neutral distribution directly.

Engineering notes (see README "Engineering improvements"):
  - The model and tokenizer are loaded ONCE at module/class level, not
    per-call.
  - Inference is batched instead of one (context, aspect) pair at a time.
  - Uses `torch.inference_mode()` instead of no context manager at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"


@dataclass
class AspectSentiment:
    aspect: str
    context: str
    label: str          # "Positive" | "Negative" | "Neutral"
    confidence: float    # softmax probability of the predicted label; never invented


@lru_cache(maxsize=1)
def _load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load once and cache — repeated calls are free after the first."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict_batch(pairs: list[tuple[str, str]], batch_size: int = 16) -> list[AspectSentiment]:
    """
    Predict sentiment for a batch of (context_text, aspect_term) pairs in
    one or more batched forward passes.

    `pairs`: list of (context, aspect) tuples, e.g.
        [("The pasta was amazing but the waiter took forever.", "pasta"),
         ("The pasta was amazing but the waiter took forever.", "waiter")]
    """
    import torch

    tokenizer, model = _load_model_and_tokenizer()
    id2label = model.config.id2label

    results: list[AspectSentiment] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            contexts = [c for c, _ in chunk]
            aspects = [a for _, a in chunk]

            inputs = tokenizer(
                contexts,
                aspects,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)

            for (context, aspect), idx, prob in zip(chunk, top_idx.tolist(), top_prob.tolist()):
                label = id2label[idx]
                results.append(
                    AspectSentiment(
                        aspect=aspect,
                        context=context,
                        label=label.capitalize(),
                        confidence=round(float(prob), 4),
                    )
                )
    return results


def predict_one(context: str, aspect: str) -> AspectSentiment:
    return predict_batch([(context, aspect)])[0]
