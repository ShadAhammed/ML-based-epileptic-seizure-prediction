"""Command-line interface for epilepsy detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import uvicorn

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline

app = typer.Typer(
    name="epilepsy",
    help="ML-based epileptic seizure detection from scalp EEG (CHB-MIT).",
    no_args_is_help=True,
)


def _pipeline(config: Path | None) -> DetectionPipeline:
    settings = Settings.load(config) if config else Settings.load()
    return DetectionPipeline(settings)


@app.command("extract-features")
def extract_features(
    edf: Path = typer.Option(..., "--edf", help="Path to EDF recording"),
    start: int = typer.Option(..., "--start", help="Seizure start (epoch index or seconds)"),
    end: int = typer.Option(..., "--end", help="Seizure end (epoch index or seconds)"),
    output: Path = typer.Option(..., "--output", help="Output feature file path"),
    use_seconds: bool = typer.Option(False, "--use-seconds", help="Interpret start/end as seconds"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config YAML path"),
) -> None:
    """Extract epoch-level features from an EDF file."""
    pipeline = _pipeline(config)
    features = pipeline.extract_features(edf, start, end, output, use_seconds=use_seconds)
    typer.echo(f"Extracted {len(features)} epochs -> {output}")


@app.command("train")
def train(
    features: Path = typer.Option(..., "--features", help="Feature file (parquet/csv/xlsx)"),
    output_dir: Path = typer.Option(Path("models"), "--output-dir", help="Model output directory"),
    strategy: str = typer.Option("xgboost", "--strategy", help="xgboost, smote, or rusboost"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config YAML path"),
) -> None:
    """Train seizure detection model."""
    pipeline = _pipeline(config)
    artifacts = pipeline.train(features, output_dir, strategy=strategy)  # type: ignore[arg-type]
    typer.echo(f"Model saved to {artifacts.model_path}")


@app.command("predict")
def predict(
    model: Path = typer.Option(..., "--model", help="Trained model joblib path"),
    features: Path = typer.Option(..., "--features", help="Feature file for prediction"),
    output: Path = typer.Option(Path("predictions.csv"), "--output", help="Predictions output"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config YAML path"),
) -> None:
    """Run inference on feature file."""
    pipeline = _pipeline(config)
    preds = pipeline.predict(features, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    typer.echo(f"Predictions saved to {output}")


@app.command("evaluate")
def evaluate(
    model: Path = typer.Option(..., "--model", help="Trained model joblib path"),
    features: Path = typer.Option(..., "--features", help="Labeled feature file"),
    report_dir: Path = typer.Option(Path("reports"), "--report-dir", help="Report output directory"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config YAML path"),
) -> None:
    """Evaluate model and write metrics report."""
    pipeline = _pipeline(config)
    report = pipeline.evaluate(features, model, report_dir)
    typer.echo(f"Accuracy: {report['accuracy']:.4f}")
    typer.echo(f"Sensitivity: {report['sensitivity']:.4f}")
    typer.echo(f"Specificity: {report['specificity']:.4f}")
    if report_dir:
        typer.echo(f"Reports written to {report_dir}")


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    model_dir: Path = typer.Option(Path("models"), "--model-dir"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start FastAPI prediction service."""
    import os

    os.environ["EPILEPSY_MODEL_DIR"] = str(model_dir.resolve())
    uvicorn.run(
        "epilepsy_detection.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command("gui")
def launch_gui() -> None:
    """Launch optional desktop GUI."""
    from epilepsy_detection.gui.app import run_gui

    run_gui()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
