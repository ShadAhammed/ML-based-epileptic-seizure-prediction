"""
Export a trained model from the research notebook to ``models/seizure_model.joblib``.

The legacy ``Epilepsy.ipynb`` trains ``Smodel`` (XGBoost + SMOTE) in memory but
never writes a file to disk.  The dashboard and CLI load a persisted artifact
produced by :func:`save_trained_model`.

Run the snippet in ``scripts/save_model_cell.py`` at the end of your notebook
(after the SMOTE training cell), or train from a labeled feature file with::

    python scripts/save_model.py --features path/to/features.xlsx
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.models.classifier import SeizureClassifier
from epilepsy_detection.training.trainer import SeizureTrainer, TrainingArtifacts

_DEFAULT_OUTPUT = Path("models/seizure_model.joblib")


def save_trained_model(
    scaler: MinMaxScaler,
    rfe: RFECV,
    model,
    feature_columns: list[str],
    selected_features: list[str] | None = None,
    strategy: str = "smote",
    output_path: str | Path | None = None,
) -> Path:
    """Persist notebook training state in the format expected by the application.

    Call this from a notebook cell after training, passing the objects that
    already exist in the kernel:

    * ``scaler`` — the :class:`~sklearn.preprocessing.MinMaxScaler` fit on ``X``
    * ``rfe`` — the fitted :class:`~sklearn.feature_selection.RFECV` object
    * ``Smodel`` (or ``model``) — the trained XGBoost / RUSBoost estimator
    * ``feature_columns`` — full list of column names in ``X`` (before RFE)
    * ``f`` — list of selected feature names returned by ``GetFeature``

    Args:
        scaler: Fitted MinMax scaler used on the training feature matrix.
        rfe: Fitted RFECV selector.
        model: Trained classifier (e.g. notebook ``Smodel``).
        feature_columns: All feature column names (unscaled, pre-RFE).
        selected_features: RFECV-selected feature names (notebook variable ``f``).
            Inferred from ``rfe.support_`` when omitted.
        strategy: Training strategy label stored in the artifact metadata.
        output_path: Destination ``.joblib`` path.  Defaults to
            ``models/seizure_model.joblib``.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path or _DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings = Settings.load()
    classifier = SeizureClassifier(settings)
    classifier.scaler = scaler
    classifier.rfe = rfe
    classifier.model = model

    if selected_features is not None:
        classifier.selected_features = list(selected_features)
    else:
        classifier.selected_features = [
            col for col, kept in zip(feature_columns, rfe.support_) if kept
        ]

    payload = {
        "classifier": classifier,
        "strategy": strategy,
        "feature_columns": list(feature_columns),
        "settings": settings,
    }
    joblib.dump(payload, output_path)
    return output_path


def save_via_trainer(
    features_path: str | Path,
    output_dir: str | Path = "models",
    strategy: str = "smote",
) -> Path:
    """Train and save using the same pipeline as the application (no notebook needed).

    Use this when you have a labeled feature Excel/Parquet file (with an
    ``Out`` column) exported from the notebook.

    Args:
        features_path: Path to labeled features.
        output_dir: Directory for ``seizure_model.joblib``.
        strategy: ``"xgboost"``, ``"smote"``, or ``"rusboost"``.

    Returns:
        Path to the saved model file.
    """
    from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline

    pipeline = DetectionPipeline()
    artifacts = pipeline.train(features_path, output_dir, strategy=strategy)
    return artifacts.model_path
