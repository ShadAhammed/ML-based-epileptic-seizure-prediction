"""End-to-end epilepsy detection pipeline (inference-first)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.data.annotations import AnnotationParser, SeizureInterval
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
    """Output of seizure detection on one EDF recording."""

    edf_path: Path
    n_epochs: int
    recording_seconds: int
    per_epoch: pd.DataFrame
    seizures: list[DetectedSeizure]
    report: str


class DetectionPipeline:
    """
    Detect ictal periods in EEG recordings using a pre-trained model.

    Primary flow (matches notebook inference cells 20–28):
      EDF → extract features → predict per epoch → report when–when seizure occurs

    Training is optional and kept for notebook/research use only.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.extractor = EpochFeatureExtractor(self.settings, EDFLoader())
        self.trainer = SeizureTrainer(self.settings)
        self.evaluator = Evaluator()
        self.annotations = AnnotationParser()
        self._artifacts: TrainingArtifacts | None = None

    def detect_from_edf(
        self,
        edf_path: str | Path,
        model_path: str | Path,
        min_duration_epochs: int = 1,
    ) -> DetectionResult:
        """
        Load EDF, extract features, run pre-trained model, return seizure time windows.

        This is the main application entry point — no seizure start/end input required.
        """
        edf_path = Path(edf_path)
        features = self.extractor.extract_from_edf_for_detection(edf_path)
        per_epoch = self.predict_features(features, model_path)
        seizures = find_seizure_intervals(
            per_epoch["epoch_id"],
            per_epoch["predicted"],
            epoch_seconds=self.settings.epoch_seconds,
            min_duration_epochs=min_duration_epochs,
        )
        n_epochs = len(per_epoch)
        recording_seconds = n_epochs * self.settings.epoch_seconds
        report = format_detection_report(seizures, n_epochs, recording_seconds)

        return DetectionResult(
            edf_path=edf_path,
            n_epochs=n_epochs,
            recording_seconds=recording_seconds,
            per_epoch=per_epoch,
            seizures=seizures,
            report=report,
        )

    def predict_features(
        self,
        features: pd.DataFrame,
        model_path: str | Path,
    ) -> pd.DataFrame:
        """Predict seizure label per epoch from a feature matrix (no labels required)."""
        self.load_model(model_path)
        assert self._artifacts is not None

        x, y = self._artifacts.classifier.prepare_features(features)
        x_selected = self._artifacts.classifier.transform_for_inference(x)
        model = self._artifacts.classifier.model
        preds = model.predict(x_selected)

        result = pd.DataFrame(
            {
                "epoch_id": x.index,
                "time_seconds": (x.index - 1) * self.settings.epoch_seconds,
                "predicted": preds,
            }
        )
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_selected)
            result["probability_seizure"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        if not y.empty and "Out" in features.columns:
            result["actual"] = y.values
        return result

    def extract_features(
        self,
        edf_path: str | Path,
        seizure_start: int | None = None,
        seizure_end: int | None = None,
        output_path: str | Path | None = None,
        use_seconds: bool = False,
        labeled: bool = False,
    ) -> pd.DataFrame:
        """
        Extract epoch features from EDF.

        For detection (default): no start/end needed.
        For training data creation (notebook SzData): pass start/end and labeled=True.
        """
        if labeled and seizure_start is not None and seizure_end is not None:
            if use_seconds:
                interval = self.annotations.from_seconds(
                    seizure_start,
                    seizure_end,
                    epoch_seconds=self.settings.epoch_seconds,
                )
            else:
                interval = self.annotations.from_epochs(seizure_start, seizure_end)
            features = self.extractor.extract_from_edf(edf_path, interval)
        else:
            features = self.extractor.extract_from_edf_for_detection(edf_path)

        if output_path:
            self._save_features(features, output_path)
        return features

    def train(
        self,
        features_path: str | Path,
        output_dir: str | Path,
        strategy: Literal["xgboost", "smote", "rusboost"] = "xgboost",
    ) -> TrainingArtifacts:
        """Train model (notebook workflow — not required for detection app)."""
        features = self.trainer.load_features(features_path)
        artifacts = self.trainer.fit(features, strategy=strategy)
        path = self.trainer.save(artifacts, output_dir)
        artifacts.model_path = path
        self._artifacts = artifacts
        return artifacts

    def load_model(self, model_path: str | Path) -> TrainingArtifacts:
        self._artifacts = SeizureTrainer.load(model_path)
        return self._artifacts

    def predict(
        self,
        features_path: str | Path,
        model_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Predict from a saved feature file."""
        features = self.trainer.load_features(features_path)
        return self.predict_features(features, model_path or self._require_model())

    def evaluate(
        self,
        features_path: str | Path,
        model_path: str | Path | None = None,
        report_dir: str | Path | None = None,
    ) -> dict:
        """Evaluate on labeled features (requires Out column)."""
        predictions = self.predict(features_path, model_path)
        if "actual" not in predictions.columns:
            raise ValueError("Evaluation requires labeled features with an 'Out' column.")
        report = self.evaluator.full_report(
            predictions["actual"],
            predictions["predicted"],
            report_dir=Path(report_dir) if report_dir else None,
        )
        report["predictions"] = predictions
        return report

    def _require_model(self) -> Path:
        if self._artifacts is None:
            raise RuntimeError("No model loaded. Provide model_path.")
        return self._artifacts.model_path

    @staticmethod
    def _save_features(features: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            features.to_parquet(path)
        elif suffix in {".xlsx", ".xls"}:
            features.to_excel(path)
        else:
            features.to_csv(path)
