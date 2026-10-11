"""
generate_dataset.py
Generates a synthetic product-review dataset (520 rows) for sentiment analysis.
"""

import random
import os
import csv
from datetime import date, timedelta

import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────
N_ROWS = 520
CATEGORIES = ["Electronics", "Books", "Clothing", "Home & Kitchen", "Sports"]

SENTIMENT_WEIGHTS = {"positive": 0.50, "negative": 0.30, "neutral": 0.20}

# Date range: 2022-01-01 to 2024-12-31
START_DATE = date(2022, 1, 1)
END_DATE   = date(2024, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days  # 1095

# ── Template pools ─────────────────────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "I absolutely love this product! It exceeded all my expectations.",
    "Great quality and fast shipping. Would definitely buy again.",
    "This is exactly what I was looking for. Very happy with my purchase.",
    "Excellent value for the price. Works perfectly.",
    "Highly recommend this to anyone looking for a reliable product.",
    "Outstanding quality and very easy to use.",
    "Delivered on time and in perfect condition. Couldn't be happier.",
    "Amazing product that works exactly as described.",
    "Five stars! This product is absolutely fantastic.",
    "Best purchase I've made this year. Quality is top-notch.",
    "Very satisfied with this item. It performs beautifully.",
    "Superb craftsmanship and great customer service.",
    "Works like a charm. Definitely worth the money.",
    "I'm very impressed with the quality of this product.",
    "Fantastic product! My family loves it.",
    "Really pleased with this purchase. Meets all my needs.",
]

NEGATIVE_TEMPLATES = [
    "Very disappointed with this product. It broke after one week.",
    "Not worth the money at all. Poor quality.",
    "Terrible experience. The product stopped working immediately.",
    "Do not buy this. It is a complete waste of money.",
    "Worst purchase ever. I want my money back.",
    "The product arrived damaged and customer service was unhelpful.",
    "Extremely poor quality. Fell apart after first use.",
    "Does not work as advertised. Very misleading description.",
    "Broke within days. Very unreliable product.",
    "This is garbage. Save your money.",
    "Completely useless product. Don't bother.",
    "Had high hopes but this product failed to deliver.",
    "Poorly made and overpriced. Would not recommend.",
    "Stopped working after a week. Very frustrating.",
    "The worst product I have ever bought. Total disappointment.",
    "Cheap materials and poor construction. Avoid this.",
]

NEUTRAL_TEMPLATES = [
    "The product is okay. Nothing special but it does the job.",
    "Average quality. Expected more for the price.",
    "It works as described. Neither good nor bad.",
    "Decent product but there is room for improvement.",
    "Not bad, not great. Just average.",
    "It is what it is. Does the basic job.",
    "Fairly standard product. No complaints but not impressed either.",
    "Works fine for basic use. Do not expect anything extraordinary.",
    "Product is acceptable. Arrived on time.",
    "Met my basic expectations. Nothing more, nothing less.",
    "Reasonable quality for the price. Functional but unremarkable.",
    "Does what it says on the box. Neither a bargain nor a ripoff.",
    "Middling quality. Some aspects are good, some are lacking.",
    "Serviceable product. Would consider alternatives next time.",
    "It does the job adequately. No strong feelings either way.",
    "Standard product with no standout features.",
]

TEMPLATE_MAP = {
    "positive": POSITIVE_TEMPLATES,
    "negative": NEGATIVE_TEMPLATES,
    "neutral":  NEUTRAL_TEMPLATES,
}

# Category-specific suffix phrases (used 30 % of the time)
CATEGORY_SUFFIX = {
    "Electronics":    "Great for everyday tech use.",
    "Books":          "A wonderful addition to my library.",
    "Clothing":       "Fits perfectly and looks great.",
    "Home & Kitchen": "Makes a real difference in my home.",
    "Sports":         "Perfect for my active lifestyle.",
}

# ── Helper functions ───────────────────────────────────────────────────────────

def random_date() -> str:
    """Return a random date string between START_DATE and END_DATE."""
    offset = random.randint(0, DATE_RANGE_DAYS)
    return (START_DATE + timedelta(days=offset)).isoformat()


def build_review_text(sentiment: str, category: str) -> str:
    """Randomly combine 1–3 templates and optionally append a category suffix."""
    pool = TEMPLATE_MAP[sentiment]
    n_sentences = random.randint(1, 3)
    chosen = random.sample(pool, k=min(n_sentences, len(pool)))
    text = " ".join(chosen)
    if random.random() < 0.30:
        text += " " + CATEGORY_SUFFIX[category]
    return text


def pick_rating(sentiment: str) -> int:
    if sentiment == "positive":
        return random.choice([4, 5])
    elif sentiment == "negative":
        return random.choice([1, 2])
    else:  # neutral
        return 3


# ── Main generation ────────────────────────────────────────────────────────────

def main():
    # Determine per-sentiment counts that sum to N_ROWS
    n_positive = round(N_ROWS * SENTIMENT_WEIGHTS["positive"])  # 260
    n_negative = round(N_ROWS * SENTIMENT_WEIGHTS["negative"])  # 156
    n_neutral  = N_ROWS - n_positive - n_negative               # 104

    sentiments = (
        ["positive"] * n_positive +
        ["negative"] * n_negative +
        ["neutral"]  * n_neutral
    )
    random.shuffle(sentiments)

    rows = []
    for i, sentiment in enumerate(sentiments, start=1):
        category    = random.choice(CATEGORIES)
        rating      = pick_rating(sentiment)
        review_text = build_review_text(sentiment, category)
        review_date = random_date()
        rows.append({
            "review_id":        i,
            "product_category": category,
            "rating":           rating,
            "review_text":      review_text,
            "sentiment":        sentiment,
            "date":             review_date,
        })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reviews.csv")

    fieldnames = ["review_id", "product_category", "rating", "review_text", "sentiment", "date"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(rows)
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in rows:
        counts[r["sentiment"]] += 1

    print(f"\n{'='*50}")
    print(f"Dataset written to: {out_path}")
    print(f"{'='*50}")
    print(f"Total rows   : {total}")
    print(f"{'─'*50}")
    for sentiment, count in counts.items():
        pct = count / total * 100
        print(f"  {sentiment:<10}: {count:>4}  ({pct:.1f}%)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
