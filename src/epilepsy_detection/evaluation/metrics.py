"""
Classification metrics and visualisation for seizure detection evaluation.

This module implements the ``confusion_metrics`` and ``draw_confusion_matrix``
helpers that were referenced but undefined in the original research notebook.
It also provides a higher-level :class:`Evaluator` class used by
:class:`~epilepsy_detection.pipeline.detection_pipeline.DetectionPipeline`.

Key metrics for seizure detection:
    - **Sensitivity** (recall): fraction of true ictal epochs correctly identified.
      Critical — a low sensitivity means missed seizures.
    - **Specificity**: fraction of true interictal epochs correctly classified.
      A low specificity means excess false alarms.
    - **F1-score**: harmonic mean of precision and sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as sk_metrics


@dataclass
class ConfusionMetrics:
    """Aggregated classification metrics for a binary prediction result.

    Attributes:
        accuracy: Proportion of correctly classified epochs.
        sensitivity: True positive rate (recall for the ictal class).
        specificity: True negative rate.
        precision: Positive predictive value.
        f1_score: Harmonic mean of precision and sensitivity.
        confusion_matrix: 2x2 numpy array ``[[TN, FP], [FN, TP]]``.
    """

    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1_score: float
    confusion_matrix: np.ndarray


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMetrics:
    """Compute the full set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0 / 1).
        y_pred: Model predictions (0 / 1).

    Returns:
        :class:`ConfusionMetrics` dataclass.
    """
    cm = sk_metrics.confusion_matrix(y_true, y_pred)

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class present in y_true
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return ConfusionMetrics(
        accuracy=float(sk_metrics.accuracy_score(y_true, y_pred)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1_score=float(f1),
        confusion_matrix=cm,
    )


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return sklearn's classification report as a nested dictionary.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.

    Returns:
        Dictionary with per-class precision / recall / F1 and weighted averages.
    """
    return sk_metrics.classification_report(y_true, y_pred, output_dict=True)


def draw_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> Path | None:
    """Render a confusion matrix heatmap with seaborn.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.
        output_path: If provided, save the figure to this path (PNG, 150 dpi).
        title: Plot title.

    Returns:
        The output path when the figure was saved, otherwise ``None``.
    """
    cm = sk_metrics.confusion_matrix(y_true, y_pred)
    labels = ["Interictal (0)", "Ictal (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None


class Evaluator:
    """Evaluate trained seizure detection models on labeled feature datasets."""

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
    ) -> ConfusionMetrics:
        """Compute binary classification metrics.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions.

        Returns:
            :class:`ConfusionMetrics` instance.
        """
        return confusion_metrics(np.asarray(y_true), np.asarray(y_pred))

    def full_report(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        report_dir: Path | None = None,
    ) -> dict:
        """Generate a complete evaluation report.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions.
            report_dir: If provided, save a confusion-matrix PNG to this directory.

        Returns:
            Dictionary with accuracy, sensitivity, specificity, precision,
            f1_score, and a per-class ``classification_report`` sub-dict.
        """
        cm_result = self.evaluate(y_true, y_pred)
        report = {
            "accuracy": cm_result.accuracy,
            "sensitivity": cm_result.sensitivity,
            "specificity": cm_result.specificity,
            "precision": cm_result.precision,
            "f1_score": cm_result.f1_score,
            "classification_report": classification_report_dict(
                np.asarray(y_true), np.asarray(y_pred)
            ),
        }
        if report_dir:
            draw_confusion_matrix(
                np.asarray(y_true),
                np.asarray(y_pred),
                output_path=Path(report_dir) / "confusion_matrix.png",
            )
        return report
