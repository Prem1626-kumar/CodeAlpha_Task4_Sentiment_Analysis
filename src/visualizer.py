"""
src/visualizer.py
Generates all 8 required sentiment-analysis visualizations.

Colour convention
-----------------
    positive  → green  (#2ecc71)
    negative  → red    (#e74c3c)
    neutral   → steelblue (#3498db)
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Style constants ─────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#3498db",
}
SENTIMENT_ORDER = ["positive", "negative", "neutral"]
DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (10, 6)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_close(fig: plt.Figure, filepath: Path, dpi: int = DEFAULT_DPI) -> None:
    """Save figure and close it to free memory."""
    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", filepath)


# ── 1. Sentiment distribution (bar chart) ────────────────────────────────────
def plot_sentiment_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11)
    ax.set_title("Sentiment Distribution", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)

    out_path = out_dir / "sentiment_distribution.png"
    _save_close(fig, out_path)
    return out_path


# ── 2. Sentiment percentages (pie chart) ─────────────────────────────────────
def plot_sentiment_percentages(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(13)
    ax.set_title("Sentiment Percentages", fontsize=15, fontweight="bold", pad=20)

    out_path = out_dir / "sentiment_percentages.png"
    _save_close(fig, out_path)
    return out_path


# ── 3. Sentiment by rating (grouped bar) ──────────────────────────────────────
def plot_sentiment_by_rating(df: pd.DataFrame, out_dir: Path) -> Path:
    pivot = (
        df.groupby(["rating", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[s for s in SENTIMENT_ORDER if s in df["sentiment"].unique()])
    )

    x = np.arange(len(pivot.index))
    width = 0.25
    n_groups = len(pivot.columns)
    offsets = np.linspace(-(n_groups - 1) * width / 2, (n_groups - 1) * width / 2, n_groups)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    for i, sentiment in enumerate(pivot.columns):
        ax.bar(
            x + offsets[i],
            pivot[sentiment],
            width=width,
            label=sentiment.capitalize(),
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"★ {r}" for r in pivot.index], fontsize=11)
    ax.set_title("Sentiment Distribution by Star Rating", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Star Rating", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.legend(title="Sentiment", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    out_path = out_dir / "sentiment_by_rating.png"
    _save_close(fig, out_path)
    return out_path


# ── 4. Top terms by sentiment (3-panel horizontal bar) ───────────────────────
def plot_top_terms_by_sentiment(
    df: pd.DataFrame,
    out_dir: Path,
    text_col: str = "cleaned_text",
    top_n: int = 15,
) -> Path:
    sentiments = [s for s in SENTIMENT_ORDER if s in df["sentiment"].unique()]
    fig, axes = plt.subplots(1, len(sentiments), figsize=(6 * len(sentiments), 7))
    if len(sentiments) == 1:
        axes = [axes]

    for ax, sentiment in zip(axes, sentiments):
        subset = df[df["sentiment"] == sentiment][text_col].dropna()
        words = " ".join(subset).split()
        counter = Counter(w for w in words if len(w) > 2)
        top = counter.most_common(top_n)
        if not top:
            ax.set_title(f"{sentiment.capitalize()}\n(no data)")
            continue
        labels, vals = zip(*reversed(top))
        ax.barh(labels, vals, color=SENTIMENT_COLORS[sentiment], edgecolor="white")
        ax.set_title(f"{sentiment.capitalize()} — Top {top_n} Terms",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Most Frequent Terms by Sentiment", fontsize=14, fontweight="bold", y=1.01)

    out_path = out_dir / "top_terms_by_sentiment.png"
    _save_close(fig, out_path)
    return out_path


# ── 5. Positive vs. negative word frequency comparison ───────────────────────
def plot_word_freq_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    text_col: str = "cleaned_text",
    top_n: int = 20,
) -> Path:
    def _top_words(sentiment: str, n: int) -> dict:
        subset = df[df["sentiment"] == sentiment][text_col].dropna()
        words = " ".join(subset).split()
        return dict(Counter(w for w in words if len(w) > 2).most_common(n))

    pos_words = _top_words("positive", top_n)
    neg_words = _top_words("negative", top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    if pos_words:
        labels, vals = zip(*sorted(pos_words.items(), key=lambda x: x[1]))
        ax1.barh(labels, vals, color=SENTIMENT_COLORS["positive"], edgecolor="white")
    ax1.set_title(f"Positive — Top {top_n} Words", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Frequency", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)

    if neg_words:
        labels2, vals2 = zip(*sorted(neg_words.items(), key=lambda x: x[1]))
        ax2.barh(labels2, vals2, color=SENTIMENT_COLORS["negative"], edgecolor="white")
    ax2.set_title(f"Negative — Top {top_n} Words", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Frequency", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Positive vs. Negative Word Frequency Comparison",
                 fontsize=14, fontweight="bold")

    out_path = out_dir / "word_freq_comparison.png"
    _save_close(fig, out_path)
    return out_path


# ── 6. Sentiment by product category (stacked bar) ───────────────────────────
def plot_sentiment_by_category(df: pd.DataFrame, out_dir: Path) -> Path:
    pivot = (
        df.groupby(["product_category", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[s for s in SENTIMENT_ORDER if s in df["sentiment"].unique()])
    )
    pivot = pivot.sort_values(
        "positive" if "positive" in pivot.columns else pivot.columns[0],
        ascending=True,
    )

    colors = [SENTIMENT_COLORS[s] for s in pivot.columns]
    ax = pivot.plot(kind="barh", stacked=True, figsize=(12, 6), color=colors,
                    edgecolor="white", linewidth=0.6)
    ax.set_title("Sentiment Distribution by Product Category",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Number of Reviews", fontsize=12)
    ax.set_ylabel("Product Category", fontsize=12)
    ax.legend(
        [s.capitalize() for s in pivot.columns],
        title="Sentiment", fontsize=11, loc="lower right"
    )
    ax.spines[["top", "right"]].set_visible(False)

    out_path = out_dir / "sentiment_by_category.png"
    _save_close(ax.get_figure(), out_path)
    return out_path


# ── 7. Review length distribution by sentiment (box plot) ────────────────────
def plot_review_length_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    df = df.copy()
    df["word_count"] = df["review_text"].astype(str).str.split().str.len()

    sentiments = [s for s in SENTIMENT_ORDER if s in df["sentiment"].unique()]
    data = [df[df["sentiment"] == s]["word_count"].dropna().values for s in sentiments]
    colors = [SENTIMENT_COLORS[s] for s in sentiments]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    import matplotlib
    _mpl_version = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    _boxplot_label_kwarg = "tick_labels" if _mpl_version >= (3, 9) else "labels"
    bp = ax.boxplot(
        data,
        **{_boxplot_label_kwarg: [s.capitalize() for s in sentiments]},
        patch_artist=True,
        notch=False,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title("Review Length Distribution by Sentiment",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Word Count", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    out_path = out_dir / "review_length_distribution.png"
    _save_close(fig, out_path)
    return out_path


# ── 8. Sentiment trend over time (monthly line chart) ────────────────────────
def plot_sentiment_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    if "date" not in df.columns or df["date"].isna().all():
        logger.warning("No valid date column — generating alternative chart.")
        return _plot_sentiment_rating_heatmap(df, out_dir)

    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = (
        df.groupby(["month", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[s for s in SENTIMENT_ORDER if s in df["sentiment"].unique()])
    )
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 6))
    for sentiment in monthly.columns:
        ax.plot(
            monthly.index,
            monthly[sentiment],
            label=sentiment.capitalize(),
            color=SENTIMENT_COLORS[sentiment],
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.set_title("Monthly Sentiment Trend (2022–2024)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.legend(title="Sentiment", fontsize=11)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    out_path = out_dir / "sentiment_trend.png"
    _save_close(fig, out_path)
    return out_path


def _plot_sentiment_rating_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Alternative to time-trend when date data is unavailable."""
    pivot = (
        df.groupby(["rating", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Sentiment × Rating Heatmap (alternative to time-trend)",
                 fontsize=13, fontweight="bold")

    out_path = out_dir / "sentiment_trend.png"
    _save_close(fig, out_path)
    return out_path


# ── Main entry point ────────────────────────────────────────────────────────────
def generate_all_visualizations(
    df: pd.DataFrame,
    out_dir: Path,
    text_col: str = "cleaned_text",
) -> list[Path]:
    """
    Generate all 8 visualizations and return a list of saved file paths.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: sentiment, review_text, product_category, rating, date.
        Cleaned text is expected in `text_col`.
    out_dir : Path
        Directory where PNG files are saved.
    text_col : str
        Column name for preprocessed text (used in word-frequency charts).

    Returns
    -------
    List of Path objects for each saved chart.
    """
    _ensure_dir(out_dir)

    # Ensure cleaned_text column exists (fall back to review_text)
    if text_col not in df.columns:
        logger.warning("Column '%s' not found; falling back to 'review_text'.", text_col)
        df = df.copy()
        df[text_col] = df["review_text"].astype(str)

    saved: list[Path] = []

    chart_fns = [
        ("1 — Sentiment distribution",    plot_sentiment_distribution),
        ("2 — Sentiment percentages",     plot_sentiment_percentages),
        ("3 — Sentiment by rating",       plot_sentiment_by_rating),
        ("4 — Top terms by sentiment",    lambda d, o: plot_top_terms_by_sentiment(d, o, text_col)),
        ("5 — Word frequency comparison", lambda d, o: plot_word_freq_comparison(d, o, text_col)),
        ("6 — Sentiment by category",     plot_sentiment_by_category),
        ("7 — Review length distribution",plot_review_length_distribution),
        ("8 — Sentiment trend",           plot_sentiment_trend),
    ]

    for name, fn in chart_fns:
        try:
            path = fn(df, out_dir)
            saved.append(path)
            logger.info("✓ Chart %s → %s", name, path.name)
        except Exception as exc:
            logger.error("✗ Chart %s failed: %s", name, exc)

    return saved
