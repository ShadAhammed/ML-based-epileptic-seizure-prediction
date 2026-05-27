"""
Class-imbalance strategies for seizure detection training.

EEG seizure datasets are heavily imbalanced: ictal epochs are a small
fraction of typical recordings.  This module provides two strategies used
in the original notebook:

* **SMOTE** - Synthetic Minority Over-sampling Technique.  Generates
  synthetic ictal samples to balance the training set before fitting
  XGBoost.

* **RUSBoost** - Random Under-Sampling Boosting.  An AdaBoost variant
  that under-samples the majority class at each boosting round.

Both wrappers preserve the DataFrame column names and index, making them
drop-in replacements inside :class:`~epilepsy_detection.training.trainer.SeizureTrainer`.
"""

from __future__ import annotations

import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from epilepsy_detection.config.settings import Settings


def apply_smote(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority (ictal) class using SMOTE.

    Args:
        x_train: Training feature matrix.
        y_train: Binary training labels (0 / 1).
        random_state: Random seed.  Defaults to :attr:`Settings.random_state`.

    Returns:
        Resampled ``(X, y)`` with balanced class counts.
    """
    seed = random_state if random_state is not None else Settings.load().random_state
    smote = SMOTE(random_state=seed)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    return (
        pd.DataFrame(x_res, columns=x_train.columns),
        pd.Series(y_res, name=y_train.name),
    )


def train_rusboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    base_estimator: XGBClassifier | None = None,
    random_state: int | None = None,
) -> RUSBoostClassifier:
    """Train a RUSBoost ensemble classifier on the training data.

    RUSBoost performs random under-sampling of the majority class at each
    boosting iteration, which is computationally cheaper than SMOTE for
    very large datasets.

    Args:
        x_train: Training feature matrix.
        y_train: Binary training labels.
        base_estimator: Weak learner for boosting.  Defaults to
            :class:`~xgboost.XGBClassifier`.
        random_state: Random seed.  Defaults to :attr:`Settings.random_state`.

    Returns:
        Fitted :class:`~imblearn.ensemble.RUSBoostClassifier`.
    """
    seed = random_state if random_state is not None else Settings.load().random_state
    base = base_estimator or XGBClassifier(random_state=seed)
    clf = RUSBoostClassifier(estimator=base, random_state=seed)
    clf.fit(x_train, y_train)
    return clf


def train_smote_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Apply SMOTE resampling then train a plain XGBoost classifier.

    This replicates the SMOTE branch of the research notebook.

    Args:
        x_train: Training feature matrix (already scaled and RFECV-reduced).
        y_train: Binary training labels.

    Returns:
        Fitted :class:`~xgboost.XGBClassifier`.
    """
    x_res, y_res = apply_smote(x_train, y_train)
    model = XGBClassifier(random_state=Settings.load().random_state)
    model.fit(x_res, y_res)
    return model
