"""
run_analysis.py
Main end-to-end sentiment analysis pipeline.

Usage
-----
    python run_analysis.py                          # VADER (default)
    python run_analysis.py --approach sklearn       # TF-IDF + LR

This script:
    1. Loads and cleans the raw dataset.
    2. Preprocesses review text.
    3. Runs sentiment classification.
    4. Saves the processed dataset.
    5. Generates 8 visualisations.
    6. Evaluates predictions against ground-truth labels.
    7. Generates a Markdown report.
    8. Prints a summary to stdout.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
OUT_FIGS     = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
OUT_REPORT   = OUT_REPORTS / "sentiment_analysis_report.md"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeAlpha Task 4 — Sentiment Analysis Pipeline"
    )
    parser.add_argument(
        "--approach",
        choices=["vader", "sklearn"],
        default="vader",
        help="Sentiment classifier to use (default: vader)",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_RAW),
        help="Path to the raw reviews CSV",
    )
    parser.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="Skip lemmatisation (faster)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Ensure output directories exist ───────────────────────────────────────
    for d in [DATA_PROC, OUT_FIGS, OUT_REPORTS]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CodeAlpha Task 4 — Sentiment Analysis Pipeline")
    print("=" * 60)

    # ── Step 1: Load & clean data ──────────────────────────────────────────────
    print("\n[1/7] Loading and cleaning data ...")
    from src.data_loader import load_and_clean
    try:
        df, data_summary = load_and_clean(args.data, verbose=True)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Data loading failed: %s", exc)
        print(f"\n  ERROR: {exc}")
        print("  Tip: run `python data/generate_dataset.py` to create the dataset.")
        return 1

    print(f"  ✓ {data_summary['rows_cleaned']} rows loaded "
          f"({data_summary['rows_dropped']} dropped)")
    print(f"  ✓ Sentiment distribution: {data_summary['sentiment_distribution']}")

    # ── Step 2: Preprocess text ────────────────────────────────────────────────
    print("\n[2/7] Preprocessing review text ...")
    from src.preprocessor import preprocess_series
    do_lemmatize = not args.no_lemmatize
    try:
        df["cleaned_text"] = preprocess_series(
            df["review_text"],
            remove_stopwords=True,
            lemmatize=do_lemmatize,
        )
    except Exception as exc:
        logger.warning("Lemmatisation failed (%s); retrying without it.", exc)
        df["cleaned_text"] = preprocess_series(df["review_text"], remove_stopwords=True)
    print(f"  ✓ Preprocessing complete (lemmatize={do_lemmatize})")

    # ── Step 3: Sentiment classification ──────────────────────────────────────
    print(f"\n[3/7] Running sentiment analysis (approach={args.approach}) ...")
    from src.sentiment_analyzer import build_analyzer

    metrics = None
    try:
        analyzer = build_analyzer(args.approach)

        if args.approach == "vader":
            df["predicted_sentiment"] = analyzer.predict(df["review_text"].tolist())
            print(f"  ✓ VADER predictions generated for {len(df)} reviews")

        elif args.approach == "sklearn":
            result = analyzer.train_evaluate(df, text_col="cleaned_text", label_col="sentiment")
            # For sklearn: re-predict all rows with the fitted model for downstream use
            df["predicted_sentiment"] = analyzer.predict(df["cleaned_text"].tolist())
            print(f"  ✓ Sklearn model trained on {result['n_train']} and "
                  f"tested on {result['n_test']} samples")
            # Keep test-set arrays for proper evaluation
            from src.evaluator import compute_metrics
            metrics = compute_metrics(result["y_test"], result["y_pred"])
            print(f"  ✓ Test accuracy: {metrics['accuracy']:.4f}, "
                  f"F1 (macro): {metrics['f1_macro']:.4f}")

    except Exception as exc:
        logger.error("Sentiment analysis failed: %s", exc)
        print(f"\n  ERROR: {exc}")
        return 1

    # ── Step 4: Save processed data ────────────────────────────────────────────
    print("\n[4/7] Saving processed dataset ...")
    proc_path = DATA_PROC / "reviews_cleaned.csv"
    df.to_csv(proc_path, index=False, encoding="utf-8")
    print(f"  ✓ Saved to {proc_path}")

    # ── Step 5: Visualisations ─────────────────────────────────────────────────
    print("\n[5/7] Generating visualisations ...")
    from src.visualizer import generate_all_visualizations
    try:
        saved_charts = generate_all_visualizations(df, OUT_FIGS, text_col="cleaned_text")
        print(f"  ✓ {len(saved_charts)} charts saved to {OUT_FIGS}/")
    except Exception as exc:
        logger.warning("Visualisation step failed: %s", exc)
        saved_charts = []

    # ── Step 6: Evaluate ───────────────────────────────────────────────────────
    print("\n[6/7] Evaluating predictions ...")
    from src.evaluator import compute_metrics, plot_confusion_matrix

    if metrics is None:
        # VADER or any approach: compute agreement with ground-truth labels
        metrics = compute_metrics(
            df["sentiment"].tolist(),
            df["predicted_sentiment"].tolist(),
        )
    print(f"  ✓ Accuracy={metrics['accuracy']:.4f}, "
          f"F1(macro)={metrics['f1_macro']:.4f} on {metrics['n_samples']} samples")
    cm_path = plot_confusion_matrix(metrics, OUT_FIGS)
    if cm_path:
        print(f"  ✓ Confusion matrix saved to {cm_path}")

    # ── Step 7: Generate report ────────────────────────────────────────────────
    print("\n[7/7] Generating Markdown report ...")
    from src.report_generator import generate_report

    sentiment_counts = df["predicted_sentiment"].value_counts().to_dict()
    try:
        report_path = generate_report(
            data_summary=data_summary,
            sentiment_counts=sentiment_counts,
            metrics=metrics,
            approach=args.approach,
            out_path=OUT_REPORT,
            chart_dir=OUT_FIGS,
            n_rows_raw=data_summary["rows_loaded"],
            n_rows_cleaned=data_summary["rows_cleaned"],
        )
        print(f"  ✓ Report saved to {report_path}")
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Dataset      : {data_summary['rows_cleaned']} cleaned reviews")
    print(f"  Approach     : {args.approach.upper()}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)   : {metrics['f1_macro']:.4f}")
    print(f"  Charts       : {len(saved_charts)} saved to outputs/figures/")
    print(f"  Report       : outputs/reports/sentiment_analysis_report.md")
    print(f"  Processed CSV: data/processed/reviews_cleaned.csv")
    print("\n  ⚠  Results are on SYNTHETIC data — not real-world performance.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
