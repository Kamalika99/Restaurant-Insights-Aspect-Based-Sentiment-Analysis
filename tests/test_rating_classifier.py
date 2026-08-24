import pandas as pd
import pytest

from src import rating_classifier as rc


def test_label_encode_mapping():
    assert rc.label_encode(1) == 0
    assert rc.label_encode(2) == 0
    assert rc.label_encode(3) == 1
    assert rc.label_encode(4) == 2
    assert rc.label_encode(5) == 2


def test_label_encode_rejects_invalid_rating():
    with pytest.raises(ValueError):
        rc.label_encode(0)


def test_load_and_clean_drops_like_rows_and_dedupes(tmp_path):
    csv_path = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "Restaurant": ["A", "A", "A", "A"],
            "Review": ["great food", "great food", "bad service", "Like"],
            "Rating": ["5", "5", "1", "Like"],
        }
    ).to_csv(csv_path, index=False)

    df = rc.load_and_clean(str(csv_path))

    # 'Like' row dropped, duplicate "great food" collapsed to one row
    assert len(df) == 2
    assert set(df["Review"]) == {"great food", "bad service"}
    assert set(df["label"]) == {0, 2}  # bad service -> Negative(0), great food -> Positive(2)


def test_train_test_split_has_no_review_overlap(tmp_path):
    """Regression test for the leakage check in rating_classifier.train()."""
    csv_path = tmp_path / "toy.csv"
    reviews = [f"review number {i} about the food" for i in range(60)]
    ratings = ["5" if i % 2 == 0 else "1" for i in range(60)]
    pd.DataFrame({"Restaurant": ["A"] * 60, "Review": reviews, "Rating": ratings}).to_csv(
        csv_path, index=False
    )

    result = rc.train(str(csv_path), test_size=0.25)
    assert result["n_train"] + result["n_test"] == 60
    assert "macro_f1" in result
    assert "confusion_matrix" in result
