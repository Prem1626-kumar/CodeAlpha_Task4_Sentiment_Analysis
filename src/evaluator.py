"""
src/evaluator.py
Evaluation metrics and confusion-matrix visualisation.

Note on synthetic data
----------------------
Metrics computed on the synthetic dataset do NOT establish real-world performance.
The data was generated with deterministic labels to match the review text, so
high accuracy on this dataset is expected and does not imply generalisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

LABEL_ORDER = ["positive", "negative", "neutral"]


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    *,
    labels: Optional[List[str]] = None,
) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : list of str
    y_pred : list of str
    labels : list of str, optional
        Ordered label list for the confusion matrix.

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_weighted, recall_weighted, f1_weighted,
        confusion_matrix (list of lists), labels (list),
        n_samples, note
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty y_true or y_pred — returning zero metrics.")
        return _zero_metrics()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length ({len(y_true)}) != y_pred length ({len(y_pred)})"
        )

    # Determine label order: use provided labels or LABEL_ORDER filtered to present classes
    present = sorted(set(y_true) | set(y_pred))
    if labels is None:
        labels = [l for l in LABEL_ORDER if l in present] + [l for l in present if l not in LABEL_ORDER]
    labels = [l for l in labels if l in present]

    # Handle edge case: only one class present
    avg_kwargs = {"zero_division": 0}

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro   = precision_score(y_true, y_pred, average="macro",    labels=labels, **avg_kwargs)
    recall_macro      = recall_score(   y_true, y_pred, average="macro",    labels=labels, **avg_kwargs)
    f1_macro          = f1_score(       y_true, y_pred, average="macro",    labels=labels, **avg_kwargs)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", labels=labels, **avg_kwargs)
    recall_weighted    = recall_score(   y_true, y_pred, average="weighted", labels=labels, **avg_kwargs)
    f1_weighted        = f1_score(       y_true, y_pred, average="weighted", labels=labels, **avg_kwargs)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = classification_report(y_true, y_pred, labels=labels, **avg_kwargs)

    metrics = {
        "accuracy":            round(float(accuracy),            4),
        "precision_macro":     round(float(precision_macro),     4),
        "recall_macro":        round(float(recall_macro),        4),
        "f1_macro":            round(float(f1_macro),            4),
        "precision_weighted":  round(float(precision_weighted),  4),
        "recall_weighted":     round(float(recall_weighted),     4),
        "f1_weighted":         round(float(f1_weighted),         4),
        "confusion_matrix":    cm.tolist(),
        "labels":              labels,
        "n_samples":           len(y_true),
        "classification_report": report,
        "note": (
            "⚠ Metrics are computed on synthetic data and do NOT reflect "
            "real-world model performance."
        ),
    }

    logger.info("Evaluation: accuracy=%.4f, F1(macro)=%.4f, n=%d",
                accuracy, f1_macro, len(y_true))
    return metrics


def _zero_metrics() -> dict:
    return {
        "accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0,
        "f1_macro": 0.0, "precision_weighted": 0.0, "recall_weighted": 0.0,
        "f1_weighted": 0.0, "confusion_matrix": [], "labels": [],
        "n_samples": 0, "classification_report": "",
        "note": "No samples to evaluate.",
    }


def plot_confusion_matrix(
    metrics: dict,
    out_dir: Path,
    *,
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> Optional[Path]:
    """
    Generate and save a confusion-matrix heatmap.

    Parameters
    ----------
    metrics : dict
        Output of compute_metrics().
    out_dir : Path
        Directory to save the PNG.

    Returns
    -------
    Path to the saved file, or None if the matrix is empty.
    """
    cm_data = metrics.get("confusion_matrix", [])
    labels  = metrics.get("labels", [])

    if not cm_data or not labels:
        logger.warning("Confusion matrix is empty — skipping heatmap.")
        return None

    cm = np.array(cm_data)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[l.capitalize() for l in labels],
        yticklabels=[l.capitalize() for l in labels],
        linewidths=0.5,
        linecolor="grey",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)

    out_path = out_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)
    return out_path
