"""End-to-end epilepsy detection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.data.annotations import AnnotationParser, SeizureInterval
from epilepsy_detection.data.edf_loader import EDFLoader
from epilepsy_detection.evaluation.metrics import Evaluator
from epilepsy_detection.features.epoch_features import EpochFeatureExtractor
from epilepsy_detection.training.trainer import SeizureTrainer, TrainingArtifacts


class DetectionPipeline:
    """Orchestrate feature extraction, training, prediction, and evaluation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.extractor = EpochFeatureExtractor(self.settings, EDFLoader())
        self.trainer = SeizureTrainer(self.settings)
        self.evaluator = Evaluator()
        self.annotations = AnnotationParser()
        self._artifacts: TrainingArtifacts | None = None

    def extract_features(
        self,
        edf_path: str | Path,
        seizure_start: int,
        seizure_end: int,
        output_path: str | Path | None = None,
        use_seconds: bool = False,
    ) -> pd.DataFrame:
        """Extract epoch features from EDF and optionally save."""
        if use_seconds:
            interval = self.annotations.from_seconds(
                seizure_start,
                seizure_end,
                epoch_seconds=self.settings.epoch_seconds,
            )
        else:
            interval = self.annotations.from_epochs(seizure_start, seizure_end)

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
        """Train model from feature file and persist."""
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
        """Predict seizure labels for feature file."""
        if model_path:
            self.load_model(model_path)
        if self._artifacts is None:
            raise RuntimeError("No model loaded. Provide model_path or train first.")

        features = self.trainer.load_features(features_path)
        x, y = self._artifacts.classifier.prepare_features(features)
        x_selected = self._artifacts.classifier.transform_for_inference(x)
        preds = self._artifacts.classifier.model.predict(x_selected)

        result = pd.DataFrame({"epoch_id": x.index, "actual": y.values, "predicted": preds})
        if hasattr(self._artifacts.classifier.model, "predict_proba"):
            proba = self._artifacts.classifier.model.predict_proba(x_selected)
            result["probability_seizure"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        return result

    def evaluate(
        self,
        features_path: str | Path,
        model_path: str | Path | None = None,
        report_dir: str | Path | None = None,
    ) -> dict:
        """Evaluate model on labeled feature set."""
        predictions = self.predict(features_path, model_path)
        report = self.evaluator.full_report(
            predictions["actual"],
            predictions["predicted"],
            report_dir=Path(report_dir) if report_dir else None,
        )
        report["predictions"] = predictions
        return report

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
