"""
tests/test_pipeline.py
End-to-end pipeline integration tests.
"""

import textwrap
from pathlib import Path

import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mini_csv(path: Path, n: int = 12) -> Path:
    """Write a minimal valid CSV with all three sentiment classes."""
    import csv as csv_module
    templates = [
        ("Electronics",    5, "Great product highly recommend excellent quality outstanding.", "positive"),
        ("Books",          1, "Terrible product broke immediately waste of money awful.", "negative"),
        ("Clothing",       3, "Average product nothing special does the job okay.", "neutral"),
        ("Sports",         4, "Very good product happy with purchase works well great.", "positive"),
        ("Home & Kitchen", 2, "Poor quality disappointed would not buy again bad.", "negative"),
        ("Electronics",    3, "Okay product meets basic expectations nothing remarkable fair.", "neutral"),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerow(["review_id", "product_category", "rating", "review_text", "sentiment", "date"])
        for i in range(n):
            cat, rating, text, sentiment = templates[i % len(templates)]
            month = (i % 9) + 1
            date = f"2023-{month:02d}-01"
            writer.writerow([i + 1, cat, rating, text, sentiment, date])
    return path


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_data_load_and_preprocess(self, tmp_path):
        csv = _make_mini_csv(tmp_path / "reviews.csv")
        from src.data_loader import load_and_clean
        from src.preprocessor import preprocess_series

        df, summary = load_and_clean(csv, verbose=False)
        assert len(df) > 0
        df["cleaned_text"] = preprocess_series(df["review_text"])
        assert "cleaned_text" in df.columns
        assert df["cleaned_text"].notna().all()

    def test_vader_predict_all_rows(self, tmp_path):
        csv = _make_mini_csv(tmp_path / "reviews.csv")
        from src.data_loader import load_and_clean
        from src.sentiment_analyzer import VaderSentimentAnalyzer

        df, _ = load_and_clean(csv, verbose=False)
        analyzer = VaderSentimentAnalyzer()
        preds = analyzer.predict(df["review_text"].tolist())
        assert len(preds) == len(df)
        valid = {"positive", "negative", "neutral"}
        for p in preds:
            assert p in valid

    def test_evaluator_runs_on_predictions(self, tmp_path):
        csv = _make_mini_csv(tmp_path / "reviews.csv")
        from src.data_loader import load_and_clean
        from src.sentiment_analyzer import VaderSentimentAnalyzer
        from src.evaluator import compute_metrics

        df, _ = load_and_clean(csv, verbose=False)
        analyzer = VaderSentimentAnalyzer()
        preds = analyzer.predict(df["review_text"].tolist())
        metrics = compute_metrics(df["sentiment"].tolist(), preds)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["n_samples"] == len(df)

    def test_visualizations_generated(self, tmp_path):
        csv = _make_mini_csv(tmp_path / "reviews.csv", n=30)
        from src.data_loader import load_and_clean
        from src.preprocessor import preprocess_series
        from src.visualizer import generate_all_visualizations

        df, _ = load_and_clean(csv, verbose=False)
        df["cleaned_text"] = preprocess_series(df["review_text"])
        out_dir = tmp_path / "figures"
        saved = generate_all_visualizations(df, out_dir, text_col="cleaned_text")
        assert len(saved) >= 8
        for p in saved:
            assert p.exists(), f"Chart not found: {p}"
            assert p.stat().st_size > 0, f"Chart is empty: {p}"

    def test_report_generated(self, tmp_path):
        csv = _make_mini_csv(tmp_path / "reviews.csv")
        from src.data_loader import load_and_clean
        from src.sentiment_analyzer import VaderSentimentAnalyzer
        from src.evaluator import compute_metrics
        from src.report_generator import generate_report

        df, data_summary = load_and_clean(csv, verbose=False)
        analyzer = VaderSentimentAnalyzer()
        df["predicted_sentiment"] = analyzer.predict(df["review_text"].tolist())
        metrics = compute_metrics(df["sentiment"].tolist(), df["predicted_sentiment"].tolist())

        report_path = tmp_path / "report.md"
        generate_report(
            data_summary=data_summary,
            sentiment_counts=df["predicted_sentiment"].value_counts().to_dict(),
            metrics=metrics,
            approach="vader",
            out_path=report_path,
            chart_dir=tmp_path / "figures",
            n_rows_raw=data_summary["rows_loaded"],
            n_rows_cleaned=data_summary["rows_cleaned"],
        )
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 500
        assert "Sentiment" in content
        assert "SYNTHETIC" in content.upper()

    def test_processed_csv_saved(self, tmp_path):
        csv = _make_mini_csv(tmp_path / "reviews.csv")
        from src.data_loader import load_and_clean
        from src.preprocessor import preprocess_series
        from src.sentiment_analyzer import VaderSentimentAnalyzer

        df, _ = load_and_clean(csv, verbose=False)
        df["cleaned_text"] = preprocess_series(df["review_text"])
        analyzer = VaderSentimentAnalyzer()
        df["predicted_sentiment"] = analyzer.predict(df["review_text"].tolist())

        out = tmp_path / "reviews_cleaned.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == len(df)

    def test_sklearn_train_evaluate(self, tmp_path):
        csv = _make_mini_csv(tmp_path / "reviews.csv", n=60)
        from src.data_loader import load_and_clean
        from src.preprocessor import preprocess_series
        from src.sentiment_analyzer import SklearnSentimentAnalyzer

        model_path = tmp_path / "model.pkl"
        df, _ = load_and_clean(csv, verbose=False)
        df["cleaned_text"] = preprocess_series(df["review_text"])
        analyzer = SklearnSentimentAnalyzer(model_path=model_path)
        result = analyzer.train_evaluate(df, text_col="cleaned_text", label_col="sentiment")
        assert "y_test" in result
        assert "y_pred" in result
        assert len(result["y_test"]) == len(result["y_pred"])
        assert model_path.exists()

    def test_empty_dataframe_no_crash_in_visualizer(self, tmp_path):
        """An empty DataFrame should be handled gracefully (charts may be empty)."""
        import pandas as pd
        from src.visualizer import generate_all_visualizations

        empty_df = pd.DataFrame(columns=[
            "review_id", "product_category", "rating",
            "review_text", "sentiment", "date", "cleaned_text"
        ])
        out_dir = tmp_path / "figures"
        # Should not raise, just produce fewer charts
        try:
            generate_all_visualizations(empty_df, out_dir)
        except Exception as exc:
            pytest.fail(f"Visualizer crashed on empty DataFrame: {exc}")
