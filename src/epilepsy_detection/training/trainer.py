"""
Training orchestration and model persistence.

:class:`SeizureTrainer` is the single entry point for all training
activities.  It orchestrates feature loading, scaling, RFECV selection,
optional imbalance handling, hyperparameter search, and artifact persistence.

Persisted artifacts (saved with :mod:`joblib`) contain:

* The fitted :class:`~epilepsy_detection.models.classifier.SeizureClassifier`
  (includes scaler and RFECV selector — no preprocessing state is lost).
* The training strategy name.
* The original feature column list.
* The :class:`~epilepsy_detection.config.settings.Settings` used during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.models.classifier import SeizureClassifier
from epilepsy_detection.models.imbalance import train_rusboost, train_smote_xgboost


# Artifact file name written by SeizureTrainer.save()
_MODEL_FILENAME = "seizure_model.joblib"


@dataclass
class TrainingArtifacts:
    """Container for all artifacts produced by a completed training run.

    Attributes:
        classifier: Fully fitted :class:`SeizureClassifier` (scaler + RFE + model).
        strategy: Training strategy used: ``"xgboost"``, ``"smote"``, or
            ``"rusboost"``.
        model_path: Path to the persisted ``.joblib`` file on disk.
        feature_columns: Column names of the original (unscaled) feature matrix,
            used to verify compatibility with new EDF recordings at inference.
    """

    classifier: SeizureClassifier
    strategy: str
    model_path: Path
    feature_columns: list[str]


class SeizureTrainer:
    """Train, save, and load seizure detection models.

    Args:
        settings: Application settings.  Loaded from ``config/default.yaml``
            when not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.classifier = SeizureClassifier(self.settings)

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def load_features(self, path: str | Path) -> pd.DataFrame:
        """Load a feature file in Parquet, CSV, or Excel format.

        Args:
            path: File path.  Supported extensions: ``.parquet``, ``.csv``,
                ``.xlsx``, ``.xls``.

        Returns:
            Feature DataFrame with epoch-ID as the index.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, index_col=0)
        else:
            df = pd.read_csv(path, index_col=0)

        # Normalise index name to "ID" regardless of source format
        if "ID" in df.columns and df.index.name != "ID":
            df = df.set_index("ID")

        return df

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        features: pd.DataFrame,
        strategy: Literal["xgboost", "smote", "rusboost"] = "xgboost",
        run_hyperparameter_search: bool = True,
    ) -> TrainingArtifacts:
        """Run the full training pipeline and return artifacts.

        The pipeline is:

        1. Separate features (X) from labels (y).
        2. MinMaxScale and apply RFECV feature selection.
        3. Depending on *strategy*, optionally handle class imbalance.
        4. Optionally run randomised + grid hyperparameter search.

        Args:
            features: Labeled feature DataFrame (must contain an ``"Out"`` column).
            strategy: Imbalance handling strategy.
                - ``"xgboost"`` — standard XGBoost with hyperparameter search.
                - ``"smote"`` — SMOTE oversampling then XGBoost.
                - ``"rusboost"`` — RUSBoost ensemble.
            run_hyperparameter_search: If ``True`` and *strategy* is ``"xgboost"``,
                perform randomised + grid search.  Set ``False`` for a quick
                baseline fit.

        Returns:
            :class:`TrainingArtifacts` with the fitted classifier.
        """
        x, y = self.classifier.prepare_features(features)
        selection = self.classifier.select_features(x, y)

        if strategy == "smote":
            self.classifier.model = train_smote_xgboost(
                selection.x_train, selection.y_train
            )
        elif strategy == "rusboost":
            # RUSBoostClassifier is compatible with the predict/predict_proba
            # interface; assigned directly to classifier.model.
            self.classifier.model = train_rusboost(  # type: ignore[assignment]
                selection.x_train, selection.y_train
            )
        else:
            if run_hyperparameter_search:
                self.classifier.hyperparameter_search(selection.x_train, selection.y_train)
            else:
                self.classifier.fit(selection.x_train, selection.y_train)

        return TrainingArtifacts(
            classifier=self.classifier,
            strategy=strategy,
            model_path=Path(),  # Populated by save()
            feature_columns=list(x.columns),
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, artifacts: TrainingArtifacts, output_dir: str | Path) -> Path:
        """Serialise training artifacts to a ``.joblib`` file.

        The serialised payload includes the full :class:`SeizureClassifier`
        (with embedded scaler and RFECV selector), ensuring identical
        preprocessing at inference time.

        Args:
            artifacts: Artifacts returned by :meth:`fit`.
            output_dir: Directory to write ``seizure_model.joblib``.

        Returns:
            Absolute path of the written file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / _MODEL_FILENAME

        payload = {
            "classifier": artifacts.classifier,
            "strategy": artifacts.strategy,
            "feature_columns": artifacts.feature_columns,
            "settings": self.settings,
        }
        joblib.dump(payload, model_path)
        artifacts.model_path = model_path
        return model_path

    @staticmethod
    def load(model_path: str | Path) -> TrainingArtifacts:
        """Deserialise training artifacts from a ``.joblib`` file.

        Args:
            model_path: Path to the ``.joblib`` file written by :meth:`save`.

        Returns:
            :class:`TrainingArtifacts` ready for inference.
        """
        payload = joblib.load(model_path)
        return TrainingArtifacts(
            classifier=payload["classifier"],
            strategy=payload.get("strategy", "xgboost"),
            model_path=Path(model_path),
            feature_columns=payload.get("feature_columns", []),
        )
