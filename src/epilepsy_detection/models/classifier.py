"""Seizure classification with RFECV and XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from epilepsy_detection.config.settings import Settings


@dataclass
class FeatureSelectionResult:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    selected_features: list[str]
    scaler: MinMaxScaler
    rfe: RFECV


class SeizureClassifier:
    """Patient-agnostic seizure detector using scaled features, RFECV, and XGBoost."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.scaler: MinMaxScaler | None = None
        self.rfe: RFECV | None = None
        self.model: XGBClassifier | None = None
        self.selected_features: list[str] = []
        self.search_results_: dict[str, Any] = {}

    def prepare_features(
        self,
        features: pd.DataFrame,
        label_col: str = "Out",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split feature matrix and labels."""
        if label_col in features.columns:
            x = features.drop(columns=[label_col])
            y = features[label_col]
        else:
            x = features
            y = pd.Series(dtype=int)
        return x, y

    def select_features(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        estimator: XGBClassifier | None = None,
    ) -> FeatureSelectionResult:
        """Scale, split, and run RFECV feature selection."""
        s = self.settings
        estimator = estimator or XGBClassifier(random_state=s.random_state)

        self.scaler = MinMaxScaler()
        x_scaled = pd.DataFrame(
            self.scaler.fit_transform(x),
            columns=x.columns,
            index=x.index,
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled,
            y,
            test_size=s.test_size,
            random_state=s.random_state,
            stratify=y if y.nunique() > 1 else None,
        )

        self.rfe = RFECV(
            estimator=estimator,
            min_features_to_select=s.rfe_min_features,
            cv=s.rfe_cv_folds,
        )
        self.rfe.fit(x_train, y_train)

        support = self.rfe.support_
        self.selected_features = [c for c, sel in zip(x_train.columns, support) if sel]

        rfx_train = pd.DataFrame(
            self.rfe.transform(x_train),
            columns=self.selected_features,
            index=x_train.index,
        )
        rfx_test = pd.DataFrame(
            self.rfe.transform(x_test),
            columns=self.selected_features,
            index=x_test.index,
        )

        return FeatureSelectionResult(
            x_train=rfx_train,
            x_test=rfx_test,
            y_train=y_train,
            y_test=y_test,
            selected_features=self.selected_features,
            scaler=self.scaler,
            rfe=self.rfe,
        )

    def transform_for_inference(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler and RFECV to new data."""
        if self.scaler is None or self.rfe is None:
            raise RuntimeError("Classifier must be fitted before inference.")
        x_scaled = pd.DataFrame(
            self.scaler.transform(x),
            columns=x.columns,
            index=x.index,
        )
        return pd.DataFrame(
            self.rfe.transform(x_scaled),
            columns=self.selected_features,
            index=x.index,
        )

    def hyperparameter_search(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid: bool = True,
    ) -> XGBClassifier:
        """Run RandomizedSearchCV and optional GridSearchCV."""
        s = self.settings
        hp = s.hyperparameter_search
        params = hp.params or {
            "min_child_weight": [3, 5, 8],
            "gamma": [0.8, 1, 1.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_depth": [5, 7],
        }

        model = XGBClassifier(random_state=s.random_state)
        skf = StratifiedKFold(n_splits=hp.folds, shuffle=True, random_state=1001)
        cv_splits = list(skf.split(x_train, y_train))

        random_search = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=hp.n_iter,
            scoring="accuracy",
            n_jobs=-1,
            cv=cv_splits,
            random_state=1001,
        )
        random_search.fit(x_train, y_train)
        self.search_results_["random_search"] = {
            "best_score": random_search.best_score_,
            "best_params": random_search.best_params_,
            "best_estimator": random_search.best_estimator_,
        }

        best = random_search.best_estimator_
        if use_grid:
            grid_search = GridSearchCV(
                XGBClassifier(random_state=s.random_state),
                param_grid=params,
                scoring="accuracy",
                n_jobs=-1,
                cv=cv_splits,
            )
            grid_search.fit(x_train, y_train)
            self.search_results_["grid_search"] = {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                "best_estimator": grid_search.best_estimator_,
            }
            best = grid_search.best_estimator_

        self.model = best
        return best

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Fit XGBoost on pre-selected features."""
        self.model = XGBClassifier(random_state=self.settings.random_state)
        self.model.fit(x_train, y_train)
        return self.model

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        x_transformed = self.transform_for_inference(x) if self.scaler else x
        return self.model.predict(x_transformed)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        x_transformed = self.transform_for_inference(x)
        return self.model.predict_proba(x_transformed)
