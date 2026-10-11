"""
src/sentiment_analyzer.py
Sentiment classification with two interchangeable backends.

Approach A — VADER (default)
    Valence Aware Dictionary and sEntiment Reasoner is a lexicon-based,
    rule-based tool specifically attuned to social-media and product-review text.
    It works entirely offline.  Compound score thresholds:
        ≥  0.05  →  positive
        ≤ -0.05  →  negative
        in between →  neutral
    Reference: Hutto & Gilbert (2014).

Approach B — TF-IDF + Logistic Regression (sklearn)
    A lightweight, interpretable text-classification pipeline.
    TfidfVectorizer converts cleaned text to sparse feature vectors and
    LogisticRegression learns a linear decision boundary.
    The vectorizer is fit ONLY on training data to prevent leakage.
    Model is persisted to outputs/model.pkl after training.

Limitations (both approaches)
------------------------------
* Sarcasm and irony are not detected.
* Mixed-sentiment reviews may be misclassified.
* Domain shift: performance degrades on text very different from training data.
* Language: English only.
* Context and pragmatics are ignored.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Literal, Union

import numpy as np

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]

# VADER compound-score thresholds
_VADER_POS_THRESHOLD = 0.05
_VADER_NEG_THRESHOLD = -0.05

# Default model output path (relative to project root)
_DEFAULT_MODEL_PATH = Path("outputs") / "model.pkl"


# ── VADER analyser ─────────────────────────────────────────────────────────────

class VaderSentimentAnalyzer:
    """
    Lexicon-based sentiment classifier using VADER.

    Parameters
    ----------
    pos_threshold : float
        Compound score above which text is "positive".
    neg_threshold : float
        Compound score below which text is "negative".
    """

    def __init__(
        self,
        pos_threshold: float = _VADER_POS_THRESHOLD,
        neg_threshold: float = _VADER_NEG_THRESHOLD,
    ) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        logger.info(
            "VaderSentimentAnalyzer initialised (pos≥%.2f, neg≤%.2f)",
            pos_threshold,
            neg_threshold,
        )

    def predict_single(self, text: str) -> SentimentLabel:
        """Classify a single string."""
        if not isinstance(text, str) or not text.strip():
            return "neutral"
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= self.pos_threshold:
            return "positive"
        if compound <= self.neg_threshold:
            return "negative"
        return "neutral"

    def predict_scores(self, text: str) -> dict:
        """Return the full VADER score dict for a single string."""
        if not isinstance(text, str) or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self._analyzer.polarity_scores(text)

    def predict(self, texts: Union[str, List[str]]) -> Union[SentimentLabel, List[SentimentLabel]]:
        """Classify one string or a list of strings."""
        if isinstance(texts, str):
            return self.predict_single(texts)
        return [self.predict_single(t) for t in texts]


# ── Sklearn TF-IDF + LR analyser ──────────────────────────────────────────────

class SklearnSentimentAnalyzer:
    """
    Trained TF-IDF + Logistic Regression classifier.

    Parameters
    ----------
    model_path : str or Path
        Where to persist / load the trained pipeline.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = _DEFAULT_MODEL_PATH,
        random_state: int = 42,
    ) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.model_path = Path(model_path)
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._classes: list[str] = []

        self._pipeline_factory = lambda: Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20_000,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                C=1.0,
                class_weight="balanced",
            )),
        ])

    def fit(self, X_train: List[str], y_train: List[str]) -> "SklearnSentimentAnalyzer":
        """Fit the pipeline on training data."""
        if len(X_train) == 0:
            raise ValueError("Cannot train on an empty training set.")
        self._pipeline = self._pipeline_factory()
        self._pipeline.fit(X_train, y_train)
        self._classes = list(self._pipeline.classes_)
        logger.info(
            "SklearnSentimentAnalyzer trained on %d samples, classes=%s",
            len(X_train), self._classes
        )
        self._save()
        return self

    def _save(self) -> None:
        import joblib
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, self.model_path)
        logger.info("Model saved to %s", self.model_path)

    def load(self) -> "SklearnSentimentAnalyzer":
        """Load a previously saved pipeline."""
        import joblib
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No saved model at {self.model_path}. Call fit() first."
            )
        self._pipeline = joblib.load(self.model_path)
        self._classes = list(self._pipeline.classes_)
        logger.info("Model loaded from %s", self.model_path)
        return self

    def predict_single(self, text: str) -> SentimentLabel:
        if self._pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        if not isinstance(text, str) or not text.strip():
            return "neutral"
        result = self._pipeline.predict([text])[0]
        return str(result)  # type: ignore[return-value]

    def predict(self, texts: Union[str, List[str]]) -> Union[SentimentLabel, List[SentimentLabel]]:
        if self._pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        if isinstance(texts, str):
            return self.predict_single(texts)
        if not texts:
            return []
        results = self._pipeline.predict(texts)
        return [str(r) for r in results]

    def train_evaluate(
        self,
        df: "pd.DataFrame",  # type: ignore[name-defined]
        text_col: str = "cleaned_text",
        label_col: str = "sentiment",
        test_size: float = 0.20,
    ) -> dict:
        """
        Split data, fit on train, predict on test, return metrics.

        Parameters
        ----------
        df : pd.DataFrame
        text_col : str
            Column containing preprocessed text.
        label_col : str
            Column containing ground-truth labels.
        test_size : float
            Fraction to hold out for testing.

        Returns
        -------
        dict with keys: y_test, y_pred, classes, and basic metrics.
        """
        import pandas as pd  # noqa: F401
        from sklearn.model_selection import train_test_split

        X = df[text_col].fillna("").tolist()
        y = df[label_col].tolist()

        classes = sorted(set(y))
        use_stratify = all(y.count(c) >= 2 for c in classes) if len(classes) > 1 else False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if use_stratify else None,
        )
        logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)

        return {
            "y_test": y_test,
            "y_pred": y_pred,
            "classes": self._classes,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }


# ── Convenience factory ────────────────────────────────────────────────────────

def build_analyzer(
    approach: str = "vader",
    **kwargs,
) -> Union[VaderSentimentAnalyzer, SklearnSentimentAnalyzer]:
    """
    Factory function.

    Parameters
    ----------
    approach : str
        "vader" (default) or "sklearn".
    **kwargs
        Forwarded to the chosen analyser constructor.

    Returns
    -------
    Analyser instance.
    """
    approach = approach.lower().strip()
    if approach == "vader":
        return VaderSentimentAnalyzer(**kwargs)
    if approach in ("sklearn", "lr", "tfidf"):
        return SklearnSentimentAnalyzer(**kwargs)
    raise ValueError(f"Unknown approach '{approach}'. Choose 'vader' or 'sklearn'.")
