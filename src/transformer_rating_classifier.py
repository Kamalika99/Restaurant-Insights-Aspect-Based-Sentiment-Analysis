"""
DeBERTa-v3 fine-tuning for review-level rating sentiment (Negative/Neutral/
Positive), built to replace the original prototype's hand-rolled training
loop — which the audit found to contain a syntax error in its own metric
print statement, an unfilled placeholder model path, and suspiciously
perfectly-linear per-epoch loss values inconsistent with real training
(see AUDIT.md section C).

This version uses Hugging Face's `Trainer`, which:
  - logs real per-step/per-epoch loss automatically (no hand-typed numbers),
  - computes and logs real accuracy/F1 every epoch via `compute_metrics`,
  - saves a `trainer_state.json` log history as hard evidence of the run,
  - saves the fine-tuned model + tokenizer together with `save_pretrained`
    (fixing the original's unfilled `<<INSERT MODEL NAME HERE>>` path bug).

Uses the SAME data-cleaning and train/test split logic as
`rating_classifier.py` (same seed, same stratification) so the two are
directly, fairly comparable on identical data.

Requires `transformers`, `torch`, and ideally a GPU. Import is lazy so
this module doesn't fail to import in environments without a GPU;
actually calling `train_transformer()` does need those packages.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from .rating_classifier import LABEL_NAMES, RANDOM_SEED, load_and_clean

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"


class _RatingDataset:
    """Minimal torch Dataset wrapping tokenized text + integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch

        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def train_transformer(
    csv_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = "models/deberta_rating_sentiment",
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    val_size: float = 0.1,
    test_size: float = 0.2,
) -> dict:
    """
    Fine-tune `model_name` (default DeBERTa-v3-base) for 3-class rating
    sentiment. Returns a dict with the same shape as
    `rating_classifier.train()`'s return value (minus `pipeline`, plus
    `training_log`) so the two can be compared directly.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    df = load_and_clean(csv_path)

    # Same split logic/seed as rating_classifier.train() for a fair,
    # apples-to-apples comparison — train/val carved out of the same
    # train partition the classical baseline uses, test partition identical.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df["Review"], df["label"], test_size=test_size, random_state=RANDOM_SEED, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=RANDOM_SEED, stratify=y_trainval
    )

    overlap = (set(X_train) & set(X_test)) | (set(X_val) & set(X_test)) | (set(X_train) & set(X_val))
    assert not overlap, f"Data leakage across splits: {len(overlap)} overlapping reviews"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(texts):
        return tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_length
        )

    train_ds = _RatingDataset(tokenize(X_train), y_train.tolist())
    val_ds = _RatingDataset(tokenize(X_val), y_val.tolist())
    test_ds = _RatingDataset(tokenize(X_test), y_test.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )

    args = TrainingArguments(
        output_dir=f"{output_dir}_checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=RANDOM_SEED,
        report_to=[],  # no wandb/etc — keep this self-contained
        # DeBERTa-v3's disentangled attention is numerically unstable
        # under fp16 (a widely reported HF issue: relative-position score
        # computations overflow, weights go NaN within the first few
        # steps, loss collapses to 0.0/NaN and predictions freeze on one
        # class). Force pure fp32 to avoid this, and use a lower,
        # DeBERTa-appropriate learning rate with warmup + gradient
        # clipping for stability.
        fp16=False,
        bf16=False,
        learning_rate=1e-5,
        warmup_steps=100,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    # Real log history — actual per-step loss and per-epoch eval metrics,
    # straight from Trainer's internal state. This is the evidence file;
    # compare it visually to AUDIT.md section C's description of the
    # original notebook's suspiciously linear fabricated losses.
    training_log = trainer.state.log_history

    test_output = trainer.predict(test_ds)
    y_pred = np.argmax(test_output.predictions, axis=-1)

    all_labels = sorted(LABEL_NAMES)
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

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    result = {
        "model_name": model_name,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "training_log": training_log,
    }
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "training_log"}, f, indent=2)

    return result
