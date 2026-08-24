#!/usr/bin/env python3
"""
Simple CLI entry point for running the ABSA pipeline on a custom review.

Usage:
    python infer.py "The food was delicious but the service was extremely slow."
    python infer.py --context-mode full_review "The food was great, but service was slow."
    echo "The pasta was amazing but the waiter took forever." | python infer.py
"""

from __future__ import annotations

import argparse
import sys

from src import absa_pipeline as pipe
from src import aspect_extraction as ax


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aspect-based sentiment analysis on a restaurant review.")
    parser.add_argument("review", nargs="?", help="Review text. If omitted, reads from stdin.")
    parser.add_argument(
        "--context-mode",
        choices=["sentence", "full_review"],
        default="sentence",
        help="Whether to score each aspect against its relevant sentence(s) or the full review (default: sentence).",
    )
    args = parser.parse_args()

    review_text = args.review or sys.stdin.read()
    if not review_text.strip():
        parser.error("No review text provided (pass as an argument or pipe via stdin).")

    nlp = ax._load_spacy()
    results = pipe.analyze_review(review_text, context_mode=args.context_mode, nlp=nlp)
    print(pipe.format_results(results))


if __name__ == "__main__":
    main()
