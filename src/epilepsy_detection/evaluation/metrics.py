"""Evaluation metrics and visualization (replaces missing notebook helpers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics


@dataclass
class ConfusionMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1_score: float
    confusion_matrix: np.ndarray


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMetrics:
    """Compute classification metrics from predictions."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return ConfusionMetrics(
        accuracy=float(metrics.accuracy_score(y_true, y_pred)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1_score=float(f1),
        confusion_matrix=cm,
    )


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return metrics.classification_report(y_true, y_pred, output_dict=True)


def draw_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> Path | None:
    """Plot and optionally save confusion matrix heatmap."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
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
    """Evaluate trained models on feature datasets."""

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
    ) -> ConfusionMetrics:
        return confusion_metrics(np.asarray(y_true), np.asarray(y_pred))

    def full_report(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        report_dir: Path | None = None,
    ) -> dict:
        """Return metrics dict and optionally save confusion matrix plot."""
        cm = self.evaluate(y_true, y_pred)
        report = {
            "accuracy": cm.accuracy,
            "sensitivity": cm.sensitivity,
            "specificity": cm.specificity,
            "precision": cm.precision,
            "f1_score": cm.f1_score,
            "classification_report": classification_report_dict(
                np.asarray(y_true), np.asarray(y_pred)
            ),
        }
        if report_dir:
            draw_confusion_matrix(
                np.asarray(y_true),
                np.asarray(y_pred),
                output_path=report_dir / "confusion_matrix.png",
            )
        return report
