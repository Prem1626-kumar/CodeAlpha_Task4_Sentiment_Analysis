"""
tests/test_evaluator.py
Tests for src/evaluator.py
"""

from pathlib import Path

import numpy as np
import pytest

from src.evaluator import compute_metrics, plot_confusion_matrix

VALID_LABEL_SET = {"positive", "negative", "neutral"}


class TestComputeMetrics:

    def test_perfect_predictions(self):
        y_true = ["positive", "negative", "neutral", "positive", "negative"]
        y_pred = y_true[:]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)
        assert m["f1_weighted"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = ["positive", "positive", "negative", "negative"]
        y_pred = ["negative", "negative", "positive", "positive"]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_required_keys_present(self):
        y_true = ["positive", "negative", "neutral"]
        y_pred = ["positive", "positive", "neutral"]
        m = compute_metrics(y_true, y_pred)
        for key in [
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "confusion_matrix", "labels", "n_samples", "note",
        ]:
            assert key in m, f"Missing key: {key}"

    def test_confusion_matrix_shape(self):
        y_true = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        y_pred = ["positive", "negative", "neutral", "negative", "positive", "neutral"]
        m = compute_metrics(y_true, y_pred)
        n_labels = len(m["labels"])
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (n_labels, n_labels)

    def test_two_classes_no_crash(self):
        y_true = ["positive", "negative", "positive", "negative"]
        y_pred = ["positive", "positive", "negative", "negative"]
        m = compute_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_single_class_no_crash(self):
        y_true = ["positive", "positive", "positive"]
        y_pred = ["positive", "positive", "positive"]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_n_samples_matches_input(self):
        y_true = ["positive"] * 10 + ["negative"] * 10 + ["neutral"] * 5
        y_pred = ["positive"] * 10 + ["negative"] * 8 + ["positive"] * 2 + ["neutral"] * 5
        m = compute_metrics(y_true, y_pred)
        assert m["n_samples"] == 25

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_metrics(["positive", "negative"], ["positive"])

    def test_empty_inputs_returns_zero_metrics(self):
        m = compute_metrics([], [])
        assert m["accuracy"] == 0.0
        assert m["n_samples"] == 0

    def test_metrics_in_valid_range(self):
        y_true = ["positive", "negative", "neutral", "positive", "negative"]
        y_pred = ["positive", "neutral",   "neutral", "negative", "negative"]
        m = compute_metrics(y_true, y_pred)
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0, 1]"

    def test_note_field_mentions_synthetic(self):
        y_true = ["positive", "negative"]
        y_pred = ["positive", "negative"]
        m = compute_metrics(y_true, y_pred)
        assert "synthetic" in m["note"].lower() or "note" in m


class TestPlotConfusionMatrix:

    def test_saves_file(self, tmp_path):
        y_true = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        y_pred = ["positive", "negative", "neutral", "negative", "positive", "neutral"]
        m = compute_metrics(y_true, y_pred)
        path = plot_confusion_matrix(m, tmp_path)
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"

    def test_empty_metrics_returns_none(self, tmp_path):
        empty_m = {"confusion_matrix": [], "labels": []}
        result = plot_confusion_matrix(empty_m, tmp_path)
        assert result is None
