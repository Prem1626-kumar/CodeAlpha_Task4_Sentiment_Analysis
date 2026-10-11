# Sentiment Analysis Report
*Generated: 2026-10-11 16:29*

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
| **Raw rows**     | 520 |
| **Cleaned rows** | 520 |
| **Date range**   | 2022-01-08 → 2024-12-28 |
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
| Rows loaded from CSV          | 520 |
| Rows after cleaning           | 520 |
| **Total rows dropped**        | 0 |

**Drop reasons:**

| Reason                         | Count |
|--------------------------------|-------|
| None                           |      0 |

---

## 4. Methodology

### Classification Approach: `VADER`

#### VADER (Valence Aware Dictionary and sEntiment Reasoner)

VADER is a lexicon-based, rule-based sentiment classifier optimised for short, informal text such as product reviews and social media posts. It requires no training data, works entirely offline, and handles punctuation, capitalisation, and common slang natively.

**Why this approach fits the project:**
- Offline and reproducible (no API keys, no network calls).
- Fast to run on a laptop-scale dataset.
- Easy to interpret and extend.
- Appropriate for product-review text.

**VADER score thresholds:**
- compound ≥ 0.05 → **positive**
- compound ≤ −0.05 → **negative**
- otherwise → **neutral**

### Text Preprocessing

1. Remove URLs and HTML tags
2. Lowercase
3. Remove punctuation (apostrophes kept for negations)
4. Remove common English stopwords (negation words preserved: *not, no, never, nor, neither*)
5. Collapse whitespace

---

## 5. Sentiment Distribution


VADER label agreement with synthetic ground truth: **77.9%** on 520 samples.


| Sentiment  |  Count | Percentage |
|------------|--------|-----------|
| Positive   |    333 |   64.0% |
| Negative   |    171 |   32.9% |
| Neutral    |     16 |    3.1% |

**Total reviews analysed: 520**

![sentiment_distribution.png](..\figures\sentiment_distribution.png)
![sentiment_percentages.png](..\figures\sentiment_percentages.png)

---

## 6. Evaluation Results

> ⚠ **Synthetic-data caveat**: These results are computed on a synthetically generated
> dataset. They demonstrate pipeline correctness but do **not** establish real-world
> performance.

| Metric                | Score  |
|-----------------------|--------|
| Accuracy              | 0.7788 |
| Precision (macro)     | 0.8262 |
| Recall (macro)        | 0.6592 |
| F1-score (macro)      | 0.6407 |
| Precision (weighted)  | 0.8060 |
| Recall (weighted)     | 0.7788 |
| F1-score (weighted)   | 0.7270 |
| Samples evaluated     | 520 |

**Classification Report:**

```
              precision    recall  f1-score   support

    positive       0.78      1.00      0.88       260
    negative       0.76      0.83      0.80       156
     neutral       0.94      0.14      0.25       104

    accuracy                           0.78       520
   macro avg       0.83      0.66      0.64       520
weighted avg       0.81      0.78      0.73       520

```

See `outputs/figures/confusion_matrix.png` for the confusion matrix.


---

## 7. Key Findings

1. **Positive reviews dominate** (64.0% of total), consistent with typical product-review distributions where satisfied customers review more often.
2. **Negative reviews** account for 32.9% — a meaningful signal for product quality monitoring.
3. **Neutral reviews** (3.1%) tend to cluster around 3-star ratings and use hedging language.
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
