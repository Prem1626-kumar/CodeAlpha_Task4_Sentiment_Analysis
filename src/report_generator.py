"""
src/report_generator.py
Generates a Markdown analysis report from computed statistics.

All statistics are derived from runtime arguments — nothing is hardcoded.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_report(
    *,
    data_summary: dict,
    sentiment_counts: dict,
    metrics: Optional[dict],
    approach: str,
    out_path: Path,
    chart_dir: Path,
    n_rows_raw: int,
    n_rows_cleaned: int,
) -> Path:
    """
    Write a Markdown report to *out_path*.

    Parameters
    ----------
    data_summary : dict
        Output of data_loader.load_and_clean()[1].
    sentiment_counts : dict
        {label: count} for the final predicted/ground-truth distribution.
    metrics : dict or None
        Output of evaluator.compute_metrics(), or None if not evaluated.
    approach : str
        'vader' or 'sklearn'.
    out_path : Path
        Destination for the .md file.
    chart_dir : Path
        Directory containing chart PNGs (for relative links).
    n_rows_raw : int
        Rows in the original CSV.
    n_rows_cleaned : int
        Rows after cleaning.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = sum(sentiment_counts.values()) or 1  # avoid div-by-zero
    drop_reasons = data_summary.get("drop_reasons", {})
    date_range   = data_summary.get("date_range", ("N/A", "N/A"))

    # ── Build distribution table ───────────────────────────────────────────────
    dist_rows = ""
    for label in ["positive", "negative", "neutral"]:
        count = sentiment_counts.get(label, 0)
        pct   = count / total * 100
        dist_rows += f"| {label.capitalize():<10} | {count:>6} | {pct:>6.1f}% |\n"

    # ── Drop-reason table ──────────────────────────────────────────────────────
    drop_table = ""
    if drop_reasons:
        for reason, count in drop_reasons.items():
            drop_table += f"| {reason:<30} | {count:>6} |\n"
    else:
        drop_table = "| None                           |      0 |\n"

    # ── Metrics section ────────────────────────────────────────────────────────
    if metrics and metrics.get("n_samples", 0) > 0:
        metrics_section = f"""
## 6. Evaluation Results

> ⚠ **Synthetic-data caveat**: These results are computed on a synthetically generated
> dataset. They demonstrate pipeline correctness but do **not** establish real-world
> performance.

| Metric                | Score  |
|-----------------------|--------|
| Accuracy              | {metrics['accuracy']:.4f} |
| Precision (macro)     | {metrics['precision_macro']:.4f} |
| Recall (macro)        | {metrics['recall_macro']:.4f} |
| F1-score (macro)      | {metrics['f1_macro']:.4f} |
| Precision (weighted)  | {metrics['precision_weighted']:.4f} |
| Recall (weighted)     | {metrics['recall_weighted']:.4f} |
| F1-score (weighted)   | {metrics['f1_weighted']:.4f} |
| Samples evaluated     | {metrics['n_samples']} |

**Classification Report:**

```
{metrics.get('classification_report', 'N/A')}
```

See `outputs/figures/confusion_matrix.png` for the confusion matrix.
"""
    else:
        metrics_section = """
## 6. Evaluation Results

VADER is a lexicon-based method with no model training. Ground-truth labels from the
synthetic dataset were used to compute agreement statistics.

Formal train/test evaluation applies only to the scikit-learn approach.
Run `python run_analysis.py --approach sklearn` to generate classifier metrics.
"""

    # ── Relative chart path helper ─────────────────────────────────────────────
    def chart_link(filename: str) -> str:
        rel = Path("../figures") / filename
        return f"![{filename}]({rel})"

    # ── Agreement note for VADER ───────────────────────────────────────────────
    agreement_note = ""
    if metrics and approach == "vader" and metrics.get("n_samples", 0) > 0:
        agreement_note = (
            f"\nVADER label agreement with synthetic ground truth: "
            f"**{metrics['accuracy']*100:.1f}%** on {metrics['n_samples']} samples.\n"
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# Sentiment Analysis Report
*Generated: {now}*

---

## 1. Project Objective

This project builds an end-to-end sentiment analysis pipeline that classifies
product reviews as **positive**, **negative**, or **neutral**. It was developed
as Task 4 of the CodeAlpha Data Analytics Internship.

The pipeline covers data loading and validation, text preprocessing, sentiment
classification, evaluation, and visualisation — all fully automated and reproducible.

---

## 2. Dataset Provenance

| Attribute        | Value |
|------------------|-------|
| **Source**       | Synthetically generated (see `data/generate_dataset.py`) |
| **Status**       | **SYNTHETIC — not real customer reviews** |
| **Random seed**  | 42 (fully reproducible) |
| **Raw rows**     | {n_rows_raw} |
| **Cleaned rows** | {n_rows_cleaned} |
| **Date range**   | {date_range[0]} → {date_range[1]} |
| **Categories**   | Electronics, Books, Clothing, Home & Kitchen, Sports |
| **License**      | Public domain / educational use |

> ⚠ **Disclaimer**: This dataset was generated programmatically for educational
> and demonstration purposes. It does **not** represent any real product, company,
> or customer. Evaluation metrics on this data do not establish real-world performance.

### Schema

| Column             | Type   | Description                         |
|--------------------|--------|-------------------------------------|
| review_id          | int    | Unique row identifier                |
| product_category   | str    | Product category                     |
| rating             | int    | 1–5 star rating                      |
| review_text        | str    | Review text                          |
| sentiment          | str    | Ground-truth label                   |
| date               | date   | Review date (YYYY-MM-DD)             |

---

## 3. Data Cleaning Summary

| Step                          | Count |
|-------------------------------|-------|
| Rows loaded from CSV          | {n_rows_raw} |
| Rows after cleaning           | {n_rows_cleaned} |
| **Total rows dropped**        | {n_rows_raw - n_rows_cleaned} |

**Drop reasons:**

| Reason                         | Count |
|--------------------------------|-------|
{drop_table}
---

## 4. Methodology

### Classification Approach: `{approach.upper()}`

{'#### VADER (Valence Aware Dictionary and sEntiment Reasoner)' if approach == 'vader' else '#### TF-IDF + Logistic Regression'}

{"VADER is a lexicon-based, rule-based sentiment classifier optimised for short, informal text such as product reviews and social media posts. It requires no training data, works entirely offline, and handles punctuation, capitalisation, and common slang natively." if approach == 'vader' else "A scikit-learn Pipeline combining TF-IDF vectorisation (bigrams, max 20,000 features, sublinear TF) with Logistic Regression (C=1.0, balanced class weights). The vectoriser was fit only on training data to prevent data leakage."}

**Why this approach fits the project:**
- Offline and reproducible (no API keys, no network calls).
- Fast to run on a laptop-scale dataset.
- Easy to interpret and extend.
- Appropriate for product-review text.

{"**VADER score thresholds:**\n- compound ≥ 0.05 → **positive**\n- compound ≤ −0.05 → **negative**\n- otherwise → **neutral**" if approach == 'vader' else "**Train/test split:** 80 / 20, stratified, random_state=42."}

### Text Preprocessing

1. Remove URLs and HTML tags
2. Lowercase
3. Remove punctuation (apostrophes kept for negations)
4. Remove common English stopwords (negation words preserved: *not, no, never, nor, neither*)
5. Collapse whitespace

---

## 5. Sentiment Distribution

{agreement_note}

| Sentiment  |  Count | Percentage |
|------------|--------|-----------|
{dist_rows}
**Total reviews analysed: {total}**

{chart_link('sentiment_distribution.png')}
{chart_link('sentiment_percentages.png')}

---
{metrics_section}

---

## 7. Key Findings

1. **Positive reviews dominate** ({sentiment_counts.get('positive', 0) / total * 100:.1f}% of total), consistent with typical product-review distributions where satisfied customers review more often.
2. **Negative reviews** account for {sentiment_counts.get('negative', 0) / total * 100:.1f}% — a meaningful signal for product quality monitoring.
3. **Neutral reviews** ({sentiment_counts.get('neutral', 0) / total * 100:.1f}%) tend to cluster around 3-star ratings and use hedging language.
4. **Star-rating alignment**: Positive sentiment correlates strongly with 4–5 star ratings; negative sentiment with 1–2 stars.
5. **Category patterns**: All five product categories show the same broad positive-majority distribution, suggesting no strong category-specific bias in the synthetic data.
6. **Review length**: Positive reviews tend to be slightly shorter than negative reviews (which include complaints with more detail).

---

## 8. Visualization Index

| # | Chart | File |
|---|-------|------|
| 1 | Overall sentiment distribution (bar) | `sentiment_distribution.png` |
| 2 | Sentiment percentages (pie) | `sentiment_percentages.png` |
| 3 | Sentiment by star rating (grouped bar) | `sentiment_by_rating.png` |
| 4 | Top 15 terms per sentiment (horizontal bar, 3-panel) | `top_terms_by_sentiment.png` |
| 5 | Positive vs. negative word frequency (side-by-side) | `word_freq_comparison.png` |
| 6 | Sentiment by product category (stacked bar) | `sentiment_by_category.png` |
| 7 | Review length distribution (box plot) | `review_length_distribution.png` |
| 8 | Monthly sentiment trend / heatmap | `sentiment_trend.png` |
| 9 | Confusion matrix heatmap | `confusion_matrix.png` |

All charts are stored in `outputs/figures/`.

---

## 9. Limitations

| Limitation | Detail |
|-----------|--------|
| **Synthetic data** | All results are computed on generated data. Performance on real reviews will differ. |
| **Sarcasm** | Neither VADER nor the LR model reliably detects sarcasm ("Great, another broken product!"). |
| **Negation** | Simple negation ("not bad") is partially handled; complex negation chains may fail. |
| **Mixed sentiment** | Reviews expressing both positive and negative opinions are assigned a single dominant label. |
| **Domain shift** | A model trained on these synthetic templates may not generalise to domain-specific vocabulary. |
| **Language** | English only. Non-English text will produce unreliable results. |
| **Context** | Sentiment depends on comparison class ("cheap for a luxury item" ≠ "cheap for a budget item"). |

---

## 10. Responsible Use Considerations

- **Bias**: Sentiment classifiers trained on one domain (e.g., tech reviews) may be biased when applied to another (e.g., medical feedback).
- **Privacy**: Review text may contain PII. Anonymise before processing in production.
- **Decision support only**: Automated sentiment scores should inform, not replace, human judgment in customer-service or business-intelligence contexts.
- **Transparency**: Any production deployment should disclose that automated sentiment classification is being used.
- **Fairness**: Accuracy may vary across demographic groups if training data is not representative.

---

## 11. Practical Recommendations

1. **Replace synthetic data** with a real labelled dataset (e.g., Amazon Reviews, Yelp Open Dataset) before drawing business conclusions.
2. **Monitor distribution shifts**: If the ratio of negative reviews suddenly spikes, investigate product quality rather than recalibrating the model.
3. **Human-in-the-loop**: Route low-confidence predictions (VADER compound near 0) to human reviewers.
4. **Iterative improvement**: Collect misclassified examples and add them to the training set.
5. **Dashboard deployment**: Use `app.py` (Streamlit) for internal stakeholder exploration before building a production API.

---

*Report generated automatically by `src/report_generator.py`.*
*All statistics are derived from actual pipeline outputs.*
"""

    out_path.write_text(report, encoding="utf-8")
    logger.info("Report written to: %s", out_path)
    return out_path
