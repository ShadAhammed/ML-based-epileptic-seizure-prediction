"""Smoke tests for training and prediction pipeline."""

import numpy as np
import pandas as pd
import pytest

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline


def _synthetic_features(n_samples: int = 80, n_features: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_samples, n_features))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    rng.shuffle(y)
    cols = [f"MeanCh{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(x, columns=cols)
    df.index.name = "ID"
    df.index = range(1, n_samples + 1)
    df["Out"] = y
    return df


@pytest.fixture
def features_path(tmp_path):
    path = tmp_path / "features.parquet"
    _synthetic_features().to_parquet(path)
    return path


def test_train_predict_evaluate(features_path, tmp_path):
    settings = Settings(
        rfe_min_features=8,
        rfe_cv_folds=2,
        test_size=0.3,
        random_state=42,
    )
    settings.hyperparameter_search.folds = 2
    settings.hyperparameter_search.n_iter = 2
    settings.hyperparameter_search.params = {
        "max_depth": [3, 5],
        "subsample": [0.8],
    }

    pipeline = DetectionPipeline(settings)
    model_dir = tmp_path / "models"
    artifacts = pipeline.train(features_path, model_dir, strategy="xgboost")
    assert artifacts.model_path.exists()

    preds = pipeline.predict(features_path, artifacts.model_path)
    assert "predicted" in preds.columns
    assert len(preds) > 0

    report = pipeline.evaluate(features_path, artifacts.model_path, tmp_path / "reports")
    assert "accuracy" in report
    assert 0.0 <= report["accuracy"] <= 1.0
