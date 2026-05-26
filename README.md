# Epilepsy Detection

Professional ML software for **ictal (seizure) period detection** from scalp EEG, refactored from the original CHB-MIT research notebook into a modular OOP Python package.

Based on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (22 pediatric subjects, 182 annotated seizures).

## Features

- **OOP package** (`epilepsy_detection`): data loading, feature extraction, training, evaluation
- **CLI** (`epilepsy` command): extract features, train, predict, evaluate, serve API, launch GUI
- **REST API** (FastAPI): health check, model info, batch prediction from feature files
- **Desktop GUI** (tkinter): file pickers and pipeline actions without editing code
- **Class imbalance**: XGBoost (default), SMOTE + XGBoost, RUSBoost strategies
- **Reproducible training**: persisted scaler + RFECV + model via `joblib`

## Author

Abu Shad Ahammed — [abu.ahammed@uni-siegen.de](mailto:abu.ahammed@uni-siegen.de)  
Chair of Embedded Systems, Universität Siegen

## Installation

```bash
git clone https://github.com/ShadAhammed/ML-based-epileptic-seizure-prediction.git
cd ML-based-epileptic-seizure-prediction
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

Copy environment template:

```bash
copy .env.example .env
```

## Data setup

1. Register at [PhysioNet](https://physionet.org/) and request access to [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/).
2. Download EDF recordings into `EPILEPSY_DATA_DIR` (default: `./data/raw`).
3. Use `*-summary.txt` seizure annotations or provide start/end epoch indices manually.

Raw EEG files are **not** committed to this repository.

## Quick start

### Extract features from EDF

```bash
epilepsy extract-features --edf data/raw/chb01/chb01_03.edf --start 2382 --end 2447 --output data/features.parquet
```

### Train model

```bash
epilepsy train --features data/features.parquet --output-dir models/ --strategy xgboost
```

Strategies: `xgboost` (default), `smote`, `rusboost`

### Predict

```bash
epilepsy predict --model models/seizure_model.joblib --features data/features.parquet --output predictions.csv
```

### Evaluate

```bash
epilepsy evaluate --model models/seizure_model.joblib --features data/features.parquet --report-dir reports/
```

### REST API

```bash
epilepsy serve-api --model-dir models/ --port 8000
```

Endpoints:

- `GET /health`
- `GET /model/info`
- `POST /predict` (upload `.csv`, `.parquet`, or `.xlsx` feature file)

### Desktop GUI

```bash
epilepsy gui
```

## Project structure

```
config/default.yaml          # Sample rate, bands, hyperparameters
src/epilepsy_detection/
  config/                    # Settings
  data/                      # EDF loader, annotations
  features/                  # Epoch feature extraction
  models/                    # Classifier, SMOTE, RUSBoost
  training/                  # Trainer, model persistence
  evaluation/                # Metrics and plots
  pipeline/                  # End-to-end orchestration
  api/                       # FastAPI app
  gui/                       # Desktop GUI
  cli.py                     # Typer CLI
notebooks/legacy/            # Original research notebook
tests/                       # Unit and smoke tests
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design details.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest tests -v
```

## Legacy notebook

The original Jupyter workflow is preserved at [`notebooks/legacy/Epilepsy.ipynb`](notebooks/legacy/Epilepsy.ipynb).

## License

MIT License — see [LICENSE](LICENSE).

## Version history

- **1.0.0** — OOP refactor: package, CLI, API, GUI, tests, CI
- **0.1** — Initial notebook release
