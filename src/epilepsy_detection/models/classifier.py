"""
Seizure classifier: feature scaling, recursive feature elimination, and XGBoost.

This module implements the core ML pipeline described in the research notebook:

1. **MinMaxScaler** — normalise all features to [0, 1].
2. **RFECV** — recursively eliminate the least informative features using
   cross-validated XGBoost importance scores.
3. **XGBClassifier** — the final gradient-boosted tree classifier, optionally
   tuned via randomised and grid search.

The trained scaler and RFECV selector are persisted alongside the model so
that identical preprocessing is applied at inference time (no data leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from epilepsy_detection.config.settings import Settings


@dataclass
class FeatureSelectionResult:
    """Intermediate artefacts produced by the RFECV selection step.

    Attributes:
        x_train: Scaled, reduced training feature matrix.
        x_test: Scaled, reduced test feature matrix.
        y_train: Training labels.
        y_test: Test labels.
        selected_features: Names of features retained by RFECV.
        scaler: Fitted :class:`~sklearn.preprocessing.MinMaxScaler`.
        rfe: Fitted :class:`~sklearn.feature_selection.RFECV` object.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    selected_features: list[str]
    scaler: MinMaxScaler
    rfe: RFECV


class SeizureClassifier:
    """Patient-agnostic seizure classifier using MinMaxScaler + RFECV + XGBoost.

    The classifier is designed to work across patients (not patient-specific),
    matching the original research goal of building a generalised detector
    trained on the multi-subject CHB-MIT database.

    Args:
        settings: Application settings.  Loaded from ``config/default.yaml``
            when not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()

        # These attributes are set during training and required for inference.
        self.scaler: MinMaxScaler | None = None
        self.rfe: RFECV | None = None
        self.model: XGBClassifier | None = None
        self.selected_features: list[str] = []

        # Populated by hyperparameter_search() for post-hoc inspection.
        self.search_results_: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Feature preparation                                                  #
    # ------------------------------------------------------------------ #

    def prepare_features(
        self,
        features: pd.DataFrame,
        label_col: str = "Out",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split a feature DataFrame into X and y.

        The ``"Out"`` column is the label column added during training-data
        generation.  For inference, it is absent and an empty Series is
        returned for y.

        Args:
            features: Feature DataFrame (may or may not contain *label_col*).
            label_col: Name of the target column.

        Returns:
            Tuple ``(X, y)`` where ``y`` is empty when *label_col* is absent.
        """
        if label_col in features.columns:
            return features.drop(columns=[label_col]), features[label_col]
        return features.copy(), pd.Series(dtype=int)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def select_features(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        estimator: XGBClassifier | None = None,
    ) -> FeatureSelectionResult:
        """Scale features and apply RFECV to select the most informative subset.

        The same :class:`~sklearn.preprocessing.MinMaxScaler` and fitted
        :class:`~sklearn.feature_selection.RFECV` are stored on ``self`` and
        reused at inference time to guarantee identical preprocessing.

        Args:
            x: Raw feature matrix (unscaled).
            y: Binary target series (0 / 1).
            estimator: XGBoost estimator used by RFECV.  Defaults to a
                fresh :class:`~xgboost.XGBClassifier`.

        Returns:
            :class:`FeatureSelectionResult` containing the split and
            reduced train/test matrices.
        """
        s = self.settings
        estimator = estimator or XGBClassifier(random_state=s.random_state)

        # Fit scaler on training data only to avoid data leakage.
        self.scaler = MinMaxScaler()
        x_scaled = pd.DataFrame(
            self.scaler.fit_transform(x),
            columns=x.columns,
            index=x.index,
        )

        # Stratified split ensures ictal/interictal ratio is preserved.
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled,
            y,
            test_size=s.test_size,
            random_state=s.random_state,
            stratify=y if y.nunique() > 1 else None,
        )

        # RFECV eliminates features iteratively by importance, keeping at
        # least rfe_min_features (default 16) for the final model.
        self.rfe = RFECV(
            estimator=estimator,
            min_features_to_select=s.rfe_min_features,
            cv=s.rfe_cv_folds,
        )
        self.rfe.fit(x_train, y_train)

        self.selected_features = [
            col for col, kept in zip(x_train.columns, self.rfe.support_) if kept
        ]

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

    def hyperparameter_search(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid: bool = True,
    ) -> XGBClassifier:
        """Find optimal XGBoost hyperparameters via randomised and grid search.

        Mirrors the hyperparameter tuning cells in the research notebook.
        Results from both searches are stored in :attr:`search_results_` for
        post-training analysis.

        Args:
            x_train: Scaled, RFECV-reduced training matrix.
            y_train: Training labels.
            use_grid: If ``True`` (default), follow up the random search with
                a full grid search for refinement.

        Returns:
            The best-performing :class:`~xgboost.XGBClassifier`.
        """
        s = self.settings
        hp = s.hyperparameter_search
        params = hp.params  # dict loaded from config/default.yaml

        base_model = XGBClassifier(random_state=s.random_state)
        skf = StratifiedKFold(n_splits=hp.folds, shuffle=True, random_state=1001)
        cv_splits = list(skf.split(x_train, y_train))

        # --- Randomised search -------------------------------------------
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=params,
            n_iter=hp.n_iter,
            scoring="accuracy",
            n_jobs=-1,
            cv=cv_splits,
            random_state=1001,
            verbose=0,
        )
        random_search.fit(x_train, y_train)
        self.search_results_["random_search"] = {
            "best_score": random_search.best_score_,
            "best_params": random_search.best_params_,
        }
        best = random_search.best_estimator_

        # --- Grid search (optional refinement) ---------------------------
        if use_grid:
            grid_search = GridSearchCV(
                XGBClassifier(random_state=s.random_state),
                param_grid=params,
                scoring="accuracy",
                n_jobs=-1,
                cv=cv_splits,
                verbose=0,
            )
            grid_search.fit(x_train, y_train)
            self.search_results_["grid_search"] = {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
            }
            # Use the best result from whichever search performed better.
            if grid_search.best_score_ >= random_search.best_score_:
                best = grid_search.best_estimator_

        self.model = best
        return best

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Train XGBoost on pre-scaled, pre-selected features (no search).

        Args:
            x_train: Reduced training feature matrix.
            y_train: Binary training labels.

        Returns:
            The fitted :class:`~xgboost.XGBClassifier`.
        """
        self.model = XGBClassifier(random_state=self.settings.random_state)
        self.model.fit(x_train, y_train)
        return self.model

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def transform_for_inference(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler and RFECV transform to new feature data.

        This ensures the same preprocessing pipeline is used at inference
        as was used during training — avoiding train/test leakage.

        Args:
            x: Raw feature matrix with the same columns as the training data.

        Returns:
            Scaled and reduced feature matrix ready for :meth:`predict`.

        Raises:
            RuntimeError: If the classifier has not been fitted yet.
        """
        if self.scaler is None or self.rfe is None:
            raise RuntimeError(
                "Classifier must be fitted (or loaded from disk) before calling "
                "transform_for_inference()."
            )
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

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict binary seizure labels for a feature matrix.

        Args:
            x: Raw (unscaled) feature matrix.

        Returns:
            1-D integer array (0 = interictal, 1 = ictal).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(self.transform_for_inference(x))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return class probability estimates for a feature matrix.

        Args:
            x: Raw (unscaled) feature matrix.

        Returns:
            2-D array of shape ``(n_epochs, 2)``.  Column 1 is the
            probability of the ictal class.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(self.transform_for_inference(x))
