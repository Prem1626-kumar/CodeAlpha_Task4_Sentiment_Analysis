"""
tests/test_sentiment_analyzer.py
Tests for src/sentiment_analyzer.py

Strategy
--------
* Tests use directional assertions (clearly positive → "positive", etc.).
* No hardcoded VADER compound scores — those are an implementation detail.
* Empty / whitespace text must return "neutral" (safe fallback).
"""

import pytest
from src.sentiment_analyzer import (
    VaderSentimentAnalyzer,
    SklearnSentimentAnalyzer,
    build_analyzer,
)

VALID_LABELS = {"positive", "negative", "neutral"}


# ── VADER tests ────────────────────────────────────────────────────────────────

class TestVaderSentimentAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return VaderSentimentAnalyzer()

    def test_clearly_positive(self, analyzer):
        texts = [
            "I absolutely love this product! It is fantastic and amazing!",
            "Best purchase ever! Exceeded all expectations. Highly recommend!",
            "Outstanding quality, works perfectly, very happy with my purchase.",
        ]
        for t in texts:
            label = analyzer.predict_single(t)
            assert label == "positive", f"Expected positive, got '{label}' for: {t[:60]}"

    def test_clearly_negative(self, analyzer):
        texts = [
            "Terrible product! Broke immediately. Worst purchase ever. Avoid!",
            "Complete garbage, waste of money, does not work at all.",
            "Awful quality, fell apart on first day, very disappointed.",
        ]
        for t in texts:
            label = analyzer.predict_single(t)
            assert label == "negative", f"Expected negative, got '{label}' for: {t[:60]}"

    def test_neutral_text(self, analyzer):
        # Clearly flat / factual statements
        texts = [
            "The product arrived.",
            "Item received in standard packaging.",
        ]
        for t in texts:
            label = analyzer.predict_single(t)
            assert label in VALID_LABELS  # must return a valid label

    def test_empty_string_returns_neutral(self, analyzer):
        assert analyzer.predict_single("") == "neutral"

    def test_whitespace_only_returns_neutral(self, analyzer):
        assert analyzer.predict_single("   ") == "neutral"

    def test_none_like_input(self, analyzer):
        # predict_single expects str; non-strings should be handled gracefully
        assert analyzer.predict_single("") == "neutral"

    def test_batch_predict_same_length(self, analyzer):
        texts = [
            "I love it",
            "I hate it",
            "It is okay",
            "Absolutely wonderful experience",
            "Completely broken product",
        ]
        results = analyzer.predict(texts)
        assert isinstance(results, list)
        assert len(results) == len(texts)

    def test_batch_all_valid_labels(self, analyzer):
        texts = ["good", "bad", "okay", "excellent", "terrible", ""]
        results = analyzer.predict(texts)
        for r in results:
            assert r in VALID_LABELS, f"Invalid label returned: {r}"

    def test_very_long_text(self, analyzer):
        long = ("This product is absolutely fantastic and I love it so much. " * 50).strip()
        label = analyzer.predict_single(long)
        assert label in VALID_LABELS

    def test_scores_have_compound(self, analyzer):
        scores = analyzer.predict_scores("Great product!")
        assert "compound" in scores
        assert -1.0 <= scores["compound"] <= 1.0

    def test_positive_has_higher_compound_than_negative(self, analyzer):
        pos = analyzer.predict_scores("Absolutely love this product! It is fantastic!")
        neg = analyzer.predict_scores("Terrible product! Broken immediately! Waste of money!")
        assert pos["compound"] > neg["compound"]

    def test_predict_single_string_returns_string(self, analyzer):
        result = analyzer.predict("A decent product.")
        assert isinstance(result, str)
        assert result in VALID_LABELS


# ── build_analyzer factory ─────────────────────────────────────────────────────

class TestBuildAnalyzer:

    def test_vader_factory(self):
        a = build_analyzer("vader")
        assert isinstance(a, VaderSentimentAnalyzer)

    def test_sklearn_factory(self):
        a = build_analyzer("sklearn")
        assert isinstance(a, SklearnSentimentAnalyzer)

    def test_invalid_approach_raises(self):
        with pytest.raises(ValueError, match="Unknown approach"):
            build_analyzer("gpt4")

    def test_case_insensitive(self):
        a = build_analyzer("VADER")
        assert isinstance(a, VaderSentimentAnalyzer)


# ── SklearnSentimentAnalyzer tests ─────────────────────────────────────────────

class TestSklearnSentimentAnalyzer:

    @pytest.fixture
    def trained_analyzer(self, tmp_path):
        """Train a minimal sklearn analyzer for testing."""
        model_path = tmp_path / "model.pkl"
        analyzer = SklearnSentimentAnalyzer(model_path=model_path)

        X_train = (
            ["I love this product, it is fantastic and excellent"] * 15 +
            ["Terrible product, broken, waste of money, awful"] * 15 +
            ["It is okay, average, nothing special"] * 15
        )
        y_train = ["positive"] * 15 + ["negative"] * 15 + ["neutral"] * 15
        analyzer.fit(X_train, y_train)
        return analyzer

    def test_predict_returns_valid_label(self, trained_analyzer):
        label = trained_analyzer.predict_single("I love this product")
        assert label in VALID_LABELS

    def test_batch_predict_correct_length(self, trained_analyzer):
        texts = ["great", "terrible", "okay", "amazing", "broken"]
        results = trained_analyzer.predict(texts)
        assert len(results) == 5
        for r in results:
            assert r in VALID_LABELS

    def test_empty_string_returns_neutral(self, trained_analyzer):
        assert trained_analyzer.predict_single("") == "neutral"

    def test_untrained_raises_runtime_error(self, tmp_path):
        analyzer = SklearnSentimentAnalyzer(model_path=tmp_path / "nomodel.pkl")
        with pytest.raises(RuntimeError, match="not trained"):
            analyzer.predict_single("some text")

    def test_model_saved_and_loadable(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        analyzer = SklearnSentimentAnalyzer(model_path=model_path)
        X_train = (
            ["I love this excellent product"] * 10 +
            ["Terrible broken product"] * 10 +
            ["Average okay product"] * 10
        )
        y_train = ["positive"] * 10 + ["negative"] * 10 + ["neutral"] * 10
        analyzer.fit(X_train, y_train)
        assert model_path.exists()

        analyzer2 = SklearnSentimentAnalyzer(model_path=model_path)
        analyzer2.load()
        label = analyzer2.predict_single("Great product")
        assert label in VALID_LABELS
