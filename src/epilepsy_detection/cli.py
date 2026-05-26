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
    help="Detect ictal (seizure) periods in scalp EEG using a pre-trained model.",
    no_args_is_help=True,
)

train_app = typer.Typer(help="Training commands (notebook/research — not required for detection).")
app.add_typer(train_app, name="train-cmd")


def _pipeline(config: Path | None) -> DetectionPipeline:
    settings = Settings.load(config) if config else Settings.load()
    return DetectionPipeline(settings)


@app.command("detect")
def detect(
    edf: Path = typer.Option(..., "--edf", help="EDF recording to analyze"),
    model: Path = typer.Option(..., "--model", help="Pre-trained model (.joblib)"),
    output: Path = typer.Option(
        Path("reports/detection_result.csv"),
        "--output",
        help="Per-epoch predictions CSV",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Config YAML path"),
) -> None:
    """Detect when–when seizures occur in an EDF (main application command)."""
    pipeline = _pipeline(config)
    result = pipeline.detect_from_edf(edf, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.per_epoch.to_csv(output, index=False)
    typer.echo(result.report)
    typer.echo(f"\nPer-epoch output: {output}")


@app.command("extract-features")
def extract_features(
    edf: Path = typer.Option(..., "--edf", help="Path to EDF recording"),
    output: Path = typer.Option(..., "--output", help="Output feature file path"),
    start: Optional[int] = typer.Option(
        None, "--start", help="Seizure start (only for labeled training export)"
    ),
    end: Optional[int] = typer.Option(
        None, "--end", help="Seizure end (only for labeled training export)"
    ),
    use_seconds: bool = typer.Option(False, "--use-seconds"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Extract features from EDF (unlabeled by default; use --start/--end for training labels)."""
    pipeline = _pipeline(config)
    labeled = start is not None and end is not None
    features = pipeline.extract_features(
        edf, start, end, output, use_seconds=use_seconds, labeled=labeled
    )
    typer.echo(f"Extracted {len(features)} epochs -> {output}")


@train_app.command("fit")
def train_fit(
    features: Path = typer.Option(..., "--features"),
    output_dir: Path = typer.Option(Path("models"), "--output-dir"),
    strategy: str = typer.Option("xgboost", "--strategy"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Train model from labeled features (notebook workflow)."""
    pipeline = _pipeline(config)
    artifacts = pipeline.train(features, output_dir, strategy=strategy)  # type: ignore[arg-type]
    typer.echo(f"Model saved to {artifacts.model_path}")


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    model_dir: Path = typer.Option(Path("models"), "--model-dir"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start FastAPI service."""
    import os

    os.environ["EPILEPSY_MODEL_DIR"] = str(model_dir.resolve())
    uvicorn.run("epilepsy_detection.api.app:app", host=host, port=port, reload=reload)


@app.command("gui")
def launch_gui() -> None:
    """Launch seizure detection desktop GUI."""
    from epilepsy_detection.gui.app import run_gui

    run_gui()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
