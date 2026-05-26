# Epilepsy Detection

Machine learning based software for **ictal (seizure) period detection** from scalp EEG, refactored from the original CHB-MIT research notebook.

> **No clinical data in this repository.** This is intentional: EEG recordings are sensitive and must not be published on public GitHub. The code, tests, and docs are complete; you obtain the dataset separately under [PhysioNet](https://physionet.org/) terms. See **[docs/DATA.md](docs/DATA.md)** for step-by-step download and local setup instructions.

Designed for use with the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (22 pediatric subjects, 182 annotated seizures).

## Purpose

**Detect when seizures occur in an EDF recording** — not train models (that stays in the legacy notebook).

1. Load an EDF file with EEG data  
2. Extract per-second features (same logic as notebook `SzData`, without manual seizure window input)  
3. Run a **pre-trained** classifier  
4. Report **from–to** time windows where seizure activity is detected  

## Features

- **Detection pipeline**: EDF → features → predict → seizure intervals (seconds + epochs)
- **Web dashboard**: upload EDF → **Detect Seizures** → intervals, timeline, CSV export (`epilepsy dashboard`)
- **CLI**: `epilepsy detect --edf recording.edf --model models/seizure_model.joblib`
- **REST API** (FastAPI): upload feature files for batch prediction
- **Training** (optional): `epilepsy train-cmd fit` — for notebook/research only

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


**To run on real EEG:** register and get credentialed on [PhysioNet](https://physionet.org/), then download [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) into a local folder such as `./data/raw/` (gitignored). Full instructions: **[docs/DATA.md](docs/DATA.md)**.

**To evaluate the project without downloading data:** run `pytest tests -v` — tests use synthetic signals only.

### Local data layout (after download)

```
data/raw/          # CHB-MIT EDF + *-summary.txt (you download)
data/features/     # extracted Parquet/CSV (generated locally)
models/            # trained joblib artifacts (generated locally)
reports/           # evaluation outputs (generated locally)
```

## Quick start (detection)

### 1. Install and open dashboard

```bash
pip install -e .
epilepsy dashboard
```

Opens **http://localhost:8501** — upload an EDF and click **Detect Seizures**.

On Windows you can also double-click `run_dashboard.bat`.

### 2. Pre-trained model

The notebook trains the model **in memory only** — you must save it once:

**Option A** — new cell at end of `notebooks/legacy/Epilepsy.ipynb` (run [`notebooks/save_model_cell.py`](notebooks/save_model_cell.py))

**Option B** — from a labeled feature file:

```bash
python scripts/save_model.py --features path/to/features.xlsx --strategy smote
```

This creates `models/seizure_model.joblib` (local only, not in git). See [notebooks/README.md](notebooks/README.md).

### 3. Detect seizures (CLI)

```bash
epilepsy detect --edf data/raw/chb01/chb01_03.edf --model models/seizure_model.joblib
```

Example output:

```
=== Seizure detection result ===
Detected 1 seizure period(s):

  1. Seizure: 2382s – 2447s (epochs 2382–2447, duration 65s)
```

### Optional: training (local notebook, not in repo)

```bash
epilepsy extract-features --edf file.edf --start 2382 --end 2447 --output labeled.parquet
epilepsy train-cmd fit --features labeled.parquet --output-dir models/
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
  dashboard/                 # Streamlit web dashboard
  cli.py                     # Typer CLI
notebooks/                   # Local research notebooks (gitignored)
tests/                       # Unit and smoke tests
```

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system design
- [docs/DATA.md](docs/DATA.md) — where to obtain CHB-MIT data (required for full pipeline)
- [examples/quickstart.md](examples/quickstart.md) — end-to-end workflow

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest tests -v
```

## License

MIT License — see [LICENSE](LICENSE).

## Version history

- **1.0.0** — OOP refactor: package, CLI, API, GUI, tests, CI
- **0.1** — Initial notebook release
