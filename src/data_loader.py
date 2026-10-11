"""
src/data_loader.py
Loads and validates the raw reviews CSV, cleans data, and returns a quality report.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["review_id", "product_category", "rating", "review_text", "sentiment", "date"]
VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_RATINGS = {1, 2, 3, 4, 5}


def load_and_clean(
    filepath: str | Path,
    *,
    drop_duplicates: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load the reviews CSV, validate schema, clean bad records, and return a
    (cleaned_df, quality_summary) tuple.

    Parameters
    ----------
    filepath : str or Path
    drop_duplicates : bool
    verbose : bool

    Returns
    -------
    df : pd.DataFrame  — cleaned data
    summary : dict     — quality statistics

    Raises
    ------
    FileNotFoundError  — file does not exist
    ValueError         — missing columns or empty result
    """
    filepath = Path(filepath)

    # ── 1. File existence ──────────────────────────────────────────────────────
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            "Run `python data/generate_dataset.py` to create it."
        )

    # ── 2. Load ────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at {filepath}: {exc}") from exc

    rows_loaded = len(df)
    if rows_loaded == 0:
        raise ValueError(f"The CSV at {filepath} is empty.")

    # ── 3. Column validation ───────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    drop_reasons: dict[str, int] = {}

    # ── 4. Duplicate removal ───────────────────────────────────────────────────
    if drop_duplicates:
        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_dupes = n_before - len(df)
        if n_dupes:
            drop_reasons["duplicate_rows"] = n_dupes

    # ── 5. Missing / blank review_text ────────────────────────────────────────
    df["review_text"] = df["review_text"].astype(str).str.strip()
    bad_text = df["review_text"].isna() | df["review_text"].isin(["", "nan", "None"])
    if bad_text.sum():
        drop_reasons["missing_review_text"] = int(bad_text.sum())
        df = df[~bad_text].reset_index(drop=True)

    # ── 6. Validate sentiment labels ──────────────────────────────────────────
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    bad_sentiment = ~df["sentiment"].isin(VALID_SENTIMENTS)
    if bad_sentiment.sum():
        drop_reasons["invalid_sentiment_label"] = int(bad_sentiment.sum())
        df = df[~bad_sentiment].reset_index(drop=True)

    # ── 7. Validate ratings ───────────────────────────────────────────────────
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # Compare as int after coercion (coerce to float first, then check int value)
    bad_rating = df["rating"].isna() | ~df["rating"].apply(
        lambda x: (not pd.isna(x)) and (int(x) in VALID_RATINGS)
    )
    if bad_rating.sum():
        drop_reasons["invalid_rating"] = int(bad_rating.sum())
        df = df[~bad_rating].reset_index(drop=True)
    df["rating"] = df["rating"].astype(int)

    # ── 8. Validate dates ─────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_date = df["date"].isna()
    if bad_date.sum():
        drop_reasons["invalid_date"] = int(bad_date.sum())
        df = df[~bad_date].reset_index(drop=True)

    # ── 9. Final empty check ──────────────────────────────────────────────────
    if len(df) == 0:
        raise ValueError(
            "All rows were dropped during data cleaning. "
            "Check the dataset for required columns and valid values."
        )

    # ── 10. Build quality summary ─────────────────────────────────────────────
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    total_dropped = sum(drop_reasons.values())

    summary = {
        "rows_loaded": rows_loaded,
        "rows_cleaned": len(df),
        "rows_dropped": total_dropped,
        "drop_reasons": drop_reasons,
        "sentiment_distribution": sentiment_counts,
        "columns": list(df.columns),
        "date_range": (
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        ),
    }

    if verbose:
        logger.info(
            "Data loading complete: %d → %d rows (%d dropped)",
            rows_loaded, len(df), total_dropped,
        )
        for reason, count in drop_reasons.items():
            logger.warning("  Dropped %d rows: %s", count, reason)
        logger.info("  Sentiment distribution: %s", sentiment_counts)

    return df, summary
