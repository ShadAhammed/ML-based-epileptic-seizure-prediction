"""FastAPI REST service for seizure prediction."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelInfoResponse(BaseModel):
    loaded: bool
    model_path: str | None
    strategy: str | None
    n_features: int


class PredictResponse(BaseModel):
    n_epochs: int
    predictions: list[dict[str, Any]]


_app_state: dict[str, Any] = {"pipeline": None, "model_path": None}


def _get_model_path() -> Path | None:
    model_dir = Path(os.environ.get("EPILEPSY_MODEL_DIR", "models"))
    default = model_dir / "seizure_model.joblib"
    if default.exists():
        return default
    joblibs = list(model_dir.glob("*.joblib"))
    return joblibs[0] if joblibs else None


def create_app() -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        model_path = _get_model_path()
        if model_path:
            pipeline = DetectionPipeline()
            pipeline.load_model(model_path)
            _app_state["pipeline"] = pipeline
            _app_state["model_path"] = str(model_path)
        yield

    application = FastAPI(
        title="Epilepsy Detection API",
        description="REST API for ML-based ictal period detection from EEG features.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        from epilepsy_detection import __version__

        return HealthResponse(status="ok", version=__version__)

    @application.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        pipeline = _app_state.get("pipeline")
        if pipeline and pipeline._artifacts:
            art = pipeline._artifacts
            return ModelInfoResponse(
                loaded=True,
                model_path=_app_state.get("model_path"),
                strategy=art.strategy,
                n_features=len(art.feature_columns),
            )
        return ModelInfoResponse(loaded=False, model_path=None, strategy=None, n_features=0)

    @application.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)) -> PredictResponse:
        pipeline: DetectionPipeline | None = _app_state.get("pipeline")
        if pipeline is None:
            model_path = _get_model_path()
            if not model_path:
                raise HTTPException(status_code=503, detail="No trained model available.")
            pipeline = DetectionPipeline()
            pipeline.load_model(model_path)
            _app_state["pipeline"] = pipeline

        content = await file.read()
        suffix = Path(file.filename or "data.csv").suffix.lower()

        try:
            if suffix == ".parquet":
                features = pd.read_parquet(io.BytesIO(content))
            elif suffix in {".xlsx", ".xls"}:
                features = pd.read_excel(io.BytesIO(content), index_col=0)
            else:
                features = pd.read_csv(io.BytesIO(content), index_col=0)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid feature file: {exc}") from exc

        tmp_path = Path("data/cache/api_upload.csv")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(tmp_path)

        try:
            preds = pipeline.predict(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        records = preds.to_dict(orient="records")
        return PredictResponse(n_epochs=len(records), predictions=records)

    return application


app = create_app()
