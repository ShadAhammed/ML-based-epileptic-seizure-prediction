"""Training orchestration and model persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from xgboost import XGBClassifier

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.models.classifier import SeizureClassifier
from epilepsy_detection.models.imbalance import train_rusboost, train_smote_xgboost


@dataclass
class TrainingArtifacts:
    classifier: SeizureClassifier
    strategy: str
    model_path: Path
    feature_columns: list[str]


class SeizureTrainer:
    """Train seizure detection models and persist artifacts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.classifier = SeizureClassifier(self.settings)

    def load_features(self, path: str | Path) -> pd.DataFrame:
        """Load features from Parquet, CSV, or Excel."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, index_col=0)
        else:
            df = pd.read_csv(path, index_col=0)
        if "ID" in df.columns and df.index.name != "ID":
            df = df.set_index("ID")
        return df

    def fit(
        self,
        features: pd.DataFrame,
        strategy: Literal["xgboost", "smote", "rusboost"] = "xgboost",
        run_hyperparameter_search: bool = True,
    ) -> TrainingArtifacts:
        """Full training pipeline: scale, RFECV, optional imbalance handling."""
        x, y = self.classifier.prepare_features(features)
        selection = self.classifier.select_features(x, y)

        if strategy == "smote":
            model = train_smote_xgboost(selection.x_train, selection.y_train)
            self.classifier.model = model
        elif strategy == "rusboost":
            model = train_rusboost(selection.x_train, selection.y_train)
            self.classifier.model = model  # type: ignore[assignment]
        else:
            if run_hyperparameter_search:
                self.classifier.hyperparameter_search(selection.x_train, selection.y_train)
            else:
                self.classifier.fit(selection.x_train, selection.y_train)

        return TrainingArtifacts(
            classifier=self.classifier,
            strategy=strategy,
            model_path=Path(),
            feature_columns=list(x.columns),
        )

    def save(self, artifacts: TrainingArtifacts, output_dir: str | Path) -> Path:
        """Persist classifier state to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "seizure_model.joblib"
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
        """Load persisted training artifacts."""
        payload = joblib.load(model_path)
        return TrainingArtifacts(
            classifier=payload["classifier"],
            strategy=payload.get("strategy", "xgboost"),
            model_path=Path(model_path),
            feature_columns=payload.get("feature_columns", []),
        )
