"""FastAPI endpoint tests."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from epilepsy_detection.api.app import create_app
from epilepsy_detection.config.settings import Settings
from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline


def _train_fixture_model(tmp_path) -> str:
    rng = np.random.default_rng(0)
    n = 60
    x = rng.standard_normal((n, 16))
    y = np.array([0] * 30 + [1] * 30)
    cols = [f"Feat{i}" for i in range(16)]
    df = pd.DataFrame(x, columns=cols)
    df["Out"] = y
    df.index = range(1, n + 1)
    path = tmp_path / "features.parquet"
    df.to_parquet(path)

    settings = Settings(rfe_min_features=8, rfe_cv_folds=2)
    settings.hyperparameter_search.folds = 2
    settings.hyperparameter_search.n_iter = 1
    settings.hyperparameter_search.params = {"max_depth": [3]}

    pipeline = DetectionPipeline(settings)
    model_dir = tmp_path / "models"
    artifacts = pipeline.train(path, model_dir, strategy="xgboost")
    return str(artifacts.model_path)


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(tmp_path, monkeypatch):
    model_path = _train_fixture_model(tmp_path)
    monkeypatch.setenv("EPILEPSY_MODEL_DIR", str(tmp_path / "models"))

    app = create_app()
    with TestClient(app) as client:
        features = pd.read_parquet(tmp_path / "features.parquet")
        csv_bytes = features.to_csv().encode()

        response = client.post(
            "/predict",
            files={"file": ("features.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["n_epochs"] > 0
        assert len(data["predictions"]) > 0
