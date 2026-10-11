"""
tests/test_data_loader.py
Tests for src/data_loader.py
"""

import csv as csv_module
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_and_clean, REQUIRED_COLUMNS, VALID_SENTIMENTS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_valid_csv(path: Path, rows: int = 5) -> Path:
    """Write a well-formed CSV with valid rows (no commas in text fields)."""
    sentiments = ["positive", "negative", "neutral", "positive", "negative"]
    texts = [
        "Great product very happy excellent quality outstanding",
        "Terrible item broke immediately awful waste of money",
        "Okay product nothing special does basic job adequately",
        "Excellent quality fast shipping very satisfied recommend",
        "Does not work as described very disappointed bad product",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerow(["review_id", "product_category", "rating", "review_text", "sentiment", "date"])
        for i in range(rows):
            s = sentiments[i % len(sentiments)]
            t = texts[i % len(texts)]
            rating = 5 if s == "positive" else (1 if s == "negative" else 3)
            month = (i % 9) + 1
            writer.writerow([i + 1, "Electronics", rating, t, s, f"2023-{month:02d}-01"])
    return path


def _write_raw_csv(path: Path, content: str) -> Path:
    """Write a raw CSV string to a file (for edge-case tests where exact content matters)."""
    path.write_text(content.strip(), encoding="utf-8")
    return path


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLoadAndClean:

    def test_valid_csv_returns_dataframe(self, tmp_path):
        csv = _write_valid_csv(tmp_path / "reviews.csv")
        df, summary = load_and_clean(csv, verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns_present(self, tmp_path):
        csv = _write_valid_csv(tmp_path / "reviews.csv")
        df, _ = load_and_clean(csv, verbose=False)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_invalid_file_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_clean(tmp_path / "does_not_exist.csv", verbose=False)

    def test_missing_required_column_raises(self, tmp_path):
        # Missing 'sentiment' column
        content = "review_id,product_category,rating,review_text,date\n1,Electronics,5,Good,2023-01-01"
        csv = _write_raw_csv(tmp_path / "bad.csv", content)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_and_clean(csv, verbose=False)

    def test_empty_csv_raises(self, tmp_path):
        # Header only, no rows
        content = ",".join(REQUIRED_COLUMNS)
        csv = _write_raw_csv(tmp_path / "empty.csv", content)
        with pytest.raises(ValueError):
            load_and_clean(csv, verbose=False)

    def test_duplicate_rows_removed(self, tmp_path):
        path = tmp_path / "dupes.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(["review_id", "product_category", "rating", "review_text", "sentiment", "date"])
            for _ in range(3):
                writer.writerow([1, "Electronics", 5, "Great product excellent quality", "positive", "2023-01-01"])
        df, summary = load_and_clean(path, drop_duplicates=True, verbose=False)
        assert len(df) == 1
        assert summary["drop_reasons"].get("duplicate_rows", 0) == 2

    def test_blank_review_text_dropped(self, tmp_path):
        path = tmp_path / "blanks.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(["review_id", "product_category", "rating", "review_text", "sentiment", "date"])
            writer.writerow([1, "Electronics", 5, "Great product outstanding quality", "positive", "2023-01-01"])
            writer.writerow([2, "Electronics", 5, "", "positive", "2023-01-02"])
            writer.writerow([3, "Electronics", 5, "   ", "neutral", "2023-01-03"])
        df, summary = load_and_clean(path, verbose=False)
        assert len(df) == 1
        assert summary["drop_reasons"].get("missing_review_text", 0) == 2

    def test_invalid_sentiment_label_dropped(self, tmp_path):
        path = tmp_path / "bad_labels.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(["review_id", "product_category", "rating", "review_text", "sentiment", "date"])
            writer.writerow([1, "Electronics", 5, "Great product", "positive", "2023-01-01"])
            writer.writerow([2, "Electronics", 5, "Average product", "mixed", "2023-01-02"])
            writer.writerow([3, "Electronics", 5, "Bad product", "UNKNOWN", "2023-01-03"])
        df, summary = load_and_clean(path, verbose=False)
        assert len(df) == 1
        assert summary["drop_reasons"].get("invalid_sentiment_label", 0) == 2

    def test_invalid_rating_dropped(self, tmp_path):
        path = tmp_path / "bad_ratings.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(["review_id", "product_category", "rating", "review_text", "sentiment", "date"])
            writer.writerow([1, "Electronics", 5, "Great product", "positive", "2023-01-01"])
            writer.writerow([2, "Electronics", 6, "Okay product", "neutral", "2023-01-02"])
            writer.writerow([3, "Electronics", "abc", "Bad product", "negative", "2023-01-03"])
        df, summary = load_and_clean(path, verbose=False)
        assert len(df) == 1

    def test_summary_has_required_keys(self, tmp_path):
        csv = _write_valid_csv(tmp_path / "reviews.csv")
        _, summary = load_and_clean(csv, verbose=False)
        for key in ["rows_loaded", "rows_cleaned", "rows_dropped",
                    "drop_reasons", "sentiment_distribution"]:
            assert key in summary, f"Summary missing key: {key}"

    def test_sentiment_distribution_counts_match_rows(self, tmp_path):
        csv = _write_valid_csv(tmp_path / "reviews.csv")
        df, summary = load_and_clean(csv, verbose=False)
        assert sum(summary["sentiment_distribution"].values()) == len(df)

    def test_real_dataset_loads(self):
        """Integration test against the actual synthetic dataset."""
        real_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "reviews.csv"
        if not real_path.exists():
            pytest.skip("Real dataset not found — run data/generate_dataset.py first")
        df, summary = load_and_clean(real_path, verbose=False)
        assert len(df) >= 500
        assert set(df["sentiment"].unique()).issubset(VALID_SENTIMENTS)
