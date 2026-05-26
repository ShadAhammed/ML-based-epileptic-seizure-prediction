"""Class imbalance handling with SMOTE and RUSBoost."""

from __future__ import annotations

import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from epilepsy_detection.config.settings import Settings


def apply_smote(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample minority class with SMOTE."""
    settings = Settings.load()
    smote = SMOTE(random_state=random_state or settings.random_state)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    return pd.DataFrame(x_res, columns=x_train.columns), pd.Series(y_res, name=y_train.name)


def train_rusboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    base_estimator: XGBClassifier | None = None,
    random_state: int | None = None,
) -> RUSBoostClassifier:
    """Train RUSBoost ensemble on resampled data."""
    settings = Settings.load()
    base = base_estimator or XGBClassifier(random_state=settings.random_state)
    clf = RUSBoostClassifier(
        estimator=base,
        random_state=random_state or settings.random_state,
    )
    clf.fit(x_train, y_train)
    return clf


def train_smote_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Apply SMOTE then train XGBoost (notebook SMOTE path)."""
    x_res, y_res = apply_smote(x_train, y_train)
    model = XGBClassifier(random_state=Settings.load().random_state)
    model.fit(x_res, y_res)
    return model
