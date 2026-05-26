"""
End-to-end seizure detection pipeline.

This module is the primary public interface for the application.
:class:`DetectionPipeline` wires together every sub-system:

    EDF recording
        |
        v
    EpochFeatureExtractor   -- extracts statistical + band-energy features
        |
        v
    SeizureClassifier       -- applies saved scaler/RFECV/XGBoost model
        |
        v
    find_seizure_intervals  -- merges per-epoch predictions into windows
        |
        v
    DetectionResult         -- seizure start/end times + diagnostic charts

Training is deliberately separated into :class:`~epilepsy_detection.training.trainer.SeizureTrainer`
and is not required for running the detection application; it is used only
when rebuilding or updating the model from new labeled EEG data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.data.annotations import AnnotationParser
from epilepsy_detection.data.edf_loader import EDFLoader
from epilepsy_detection.detection.intervals import (
    DetectedSeizure,
    find_seizure_intervals,
    format_detection_report,
)
from epilepsy_detection.evaluation.metrics import Evaluator
from epilepsy_detection.features.epoch_features import EpochFeatureExtractor
from epilepsy_detection.training.trainer import SeizureTrainer, TrainingArtifacts


@dataclass
class DetectionResult:
    """Complete output of one seizure detection run.

    Attributes:
        edf_path: Path to the analysed EDF file.
        n_epochs: Total number of 1-second epochs in the recording.
        recording_seconds: Recording duration in seconds.
        per_epoch: DataFrame with columns ``epoch_id``, ``time_seconds``,
            ``predicted``, and ``probability_seizure``.
        seizures: List of detected ictal windows sorted by onset.
        report: Plain-text summary of detected seizures.
    """

    edf_path: Path
    n_epochs: int
    recording_seconds: int
    per_epoch: pd.DataFrame
    seizures: list[DetectedSeizure]
    report: str

    @property
    def seizure_detected(self) -> bool:
        """``True`` if at least one seizure was found."""
        return len(self.seizures) > 0


class DetectionPipeline:
    """Orchestrate EDF loading, feature extraction, model inference, and reporting.

    This is the main class that external code (dashboard, CLI, API) should
    interact with.  It exposes a single high-level method:
    :meth:`detect_from_edf` — just provide an EDF file and a model path.

    Training utilities (:meth:`train`, :meth:`extract_features`) are included
    for completeness but are not part of the normal detection workflow.

    Args:
        settings: Application settings.  Loaded automatically when omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.extractor = EpochFeatureExtractor(self.settings, EDFLoader())
        self.trainer = SeizureTrainer(self.settings)
        self.evaluator = Evaluator()
        self.annotations = AnnotationParser()
        self._artifacts: TrainingArtifacts | None = None

    # ------------------------------------------------------------------ #
    # Primary detection API                                                #
    # ------------------------------------------------------------------ #

    def detect_from_edf(
        self,
        edf_path: str | Path,
        model_path: str | Path,
        min_duration_epochs: int = 1,
    ) -> DetectionResult:
        """Analyse an EDF recording and return all predicted seizure windows.

        This is the single entry point for the dashboard, CLI ``detect``
        command, and REST API.  No seizure start/end input is required —
        the model decides where (if anywhere) seizure activity occurs.

        Args:
            edf_path: Path to the ``.edf`` recording to analyse.
            model_path: Path to a ``seizure_model.joblib`` file trained
                with :class:`~epilepsy_detection.training.trainer.SeizureTrainer`.
            min_duration_epochs: Minimum number of consecutive ictal epochs
                required to be reported as a seizure.  Increase to suppress
                brief false positives.

        Returns:
            :class:`DetectionResult` with per-epoch predictions, detected
            seizure windows, and a human-readable report.
        """
        edf_path = Path(edf_path)

        # Step 1 — Extract features (no labels; purely for inference).
        features = self.extractor.extract_from_edf(edf_path)

        # Step 2 — Load model (if not already loaded or if a different path).
        if self._artifacts is None or self._artifacts.model_path != Path(model_path):
            self.load_model(model_path)

        # Step 3 — Per-epoch prediction.
        per_epoch = self._run_inference(features)

        # Step 4 — Merge consecutive ictal epochs into seizure windows.
        seizures = find_seizure_intervals(
            per_epoch["epoch_id"],
            per_epoch["predicted"],
            epoch_seconds=self.settings.epoch_seconds,
            min_duration_epochs=min_duration_epochs,
        )

        n_epochs = len(per_epoch)
        recording_seconds = n_epochs * self.settings.epoch_seconds

        return DetectionResult(
            edf_path=edf_path,
            n_epochs=n_epochs,
            recording_seconds=recording_seconds,
            per_epoch=per_epoch,
            seizures=seizures,
            report=format_detection_report(seizures, n_epochs, recording_seconds),
        )

    # ------------------------------------------------------------------ #
    # Supporting methods                                                   #
    # ------------------------------------------------------------------ #

    def load_model(self, model_path: str | Path) -> TrainingArtifacts:
        """Load a pre-trained model from disk.

        Args:
            model_path: Path to the ``.joblib`` artifact file.

        Returns:
            The loaded :class:`TrainingArtifacts`.
        """
        self._artifacts = SeizureTrainer.load(model_path)
        return self._artifacts

    def predict(
        self,
        features_path: str | Path,
        model_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Predict seizure labels from a saved feature file.

        Args:
            features_path: Path to a feature file (parquet / csv / xlsx).
            model_path: Optional model path.  Uses the cached model when omitted.

        Returns:
            Per-epoch prediction DataFrame.
        """
        if model_path:
            self.load_model(model_path)
        if self._artifacts is None:
            raise RuntimeError("No model loaded.  Provide a model_path argument.")
        features = self.trainer.load_features(features_path)
        return self._run_inference(features)

    def evaluate(
        self,
        features_path: str | Path,
        model_path: str | Path | None = None,
        report_dir: str | Path | None = None,
    ) -> dict:
        """Evaluate model accuracy on a *labeled* feature file.

        The feature file must contain an ``"Out"`` column (0 / 1).  This
        method is used to reproduce the notebook evaluation cells.

        Args:
            features_path: Path to labeled feature file.
            model_path: Optional model path.
            report_dir: Optional directory for confusion-matrix plot.

        Returns:
            Dictionary with accuracy, sensitivity, specificity, precision,
            f1_score, and per-class classification report.

        Raises:
            ValueError: If the feature file has no ``"Out"`` label column.
        """
        predictions = self.predict(features_path, model_path)
        if "actual" not in predictions.columns:
            raise ValueError(
                "Feature file must contain an 'Out' column for evaluation.  "
                "This column is added automatically during labeled feature extraction."
            )
        report = self.evaluator.full_report(
            predictions["actual"],
            predictions["predicted"],
            report_dir=Path(report_dir) if report_dir else None,
        )
        report["predictions"] = predictions
        return report

    def extract_features(
        self,
        edf_path: str | Path,
        seizure_start: int | None = None,
        seizure_end: int | None = None,
        output_path: str | Path | None = None,
        use_seconds: bool = False,
        labeled: bool = False,
    ) -> pd.DataFrame:
        """Extract epoch features from an EDF file.

        In inference mode (default), no seizure window is needed.
        In training mode (``labeled=True`` with start/end), an ``"Out"``
        column is added for supervised learning.

        Args:
            edf_path: Path to the ``.edf`` file.
            seizure_start: Seizure onset (epoch or seconds; only for labeled mode).
            seizure_end: Seizure offset (epoch or seconds; only for labeled mode).
            output_path: Optional path to save the feature file.
            use_seconds: Interpret start/end as seconds instead of epoch indices.
            labeled: If ``True``, add the ``"Out"`` label column.

        Returns:
            Feature DataFrame.
        """
        interval = None
        if labeled and seizure_start is not None and seizure_end is not None:
            interval = (
                self.annotations.from_seconds(
                    seizure_start, seizure_end, epoch_seconds=self.settings.epoch_seconds
                )
                if use_seconds
                else self.annotations.from_epochs(seizure_start, seizure_end)
            )

        features = self.extractor.extract_from_edf(edf_path, interval)

        if output_path:
            self._save_features(features, output_path)

        return features

    def train(
        self,
        features_path: str | Path,
        output_dir: str | Path,
        strategy: Literal["xgboost", "smote", "rusboost"] = "xgboost",
    ) -> TrainingArtifacts:
        """Train a model from labeled features and save the artifact.

        Args:
            features_path: Path to a labeled feature file.
            output_dir: Directory to write ``seizure_model.joblib``.
            strategy: Imbalance handling strategy.

        Returns:
            Persisted :class:`TrainingArtifacts`.
        """
        features = self.trainer.load_features(features_path)
        artifacts = self.trainer.fit(features, strategy=strategy)
        self.trainer.save(artifacts, output_dir)
        self._artifacts = artifacts
        return artifacts

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_inference(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the loaded model to a feature matrix and return predictions.

        Args:
            features: Feature DataFrame produced by :class:`EpochFeatureExtractor`.

        Returns:
            DataFrame with columns: ``epoch_id``, ``time_seconds``,
            ``predicted``, and optionally ``probability_seizure`` and
            ``actual``.
        """
        assert self._artifacts is not None, "Model must be loaded before inference."

        clf = self._artifacts.classifier
        x, y = clf.prepare_features(features)
        x_selected = clf.transform_for_inference(x)
        model = clf.model

        preds = model.predict(x_selected)

        result = pd.DataFrame(
            {
                "epoch_id": x.index,
                "time_seconds": (x.index - 1) * self.settings.epoch_seconds,
                "predicted": preds,
            }
        )

        # Add seizure probability where the model supports it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_selected)
            result["probability_seizure"] = (
                proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            )

        # Carry over actual labels when present (for evaluation)
        if not y.empty:
            result["actual"] = y.values

        return result

    @staticmethod
    def _save_features(features: pd.DataFrame, path: str | Path) -> None:
        """Persist a feature DataFrame in the appropriate format.

        Args:
            features: Feature DataFrame to save.
            path: Output path.  Extension determines format:
                ``.parquet``, ``.xlsx``/``.xls``, or CSV (default).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            features.to_parquet(path)
        elif suffix in {".xlsx", ".xls"}:
            features.to_excel(path)
        else:
            features.to_csv(path)
