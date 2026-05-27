# EpilepsyDetector

**Upload an EDF recording. Find out if a seizure is present - and exactly when.**

EpilepsyDetector is a machine learning application that analyses scalp EEG recordings in EDF format and reports whether a seizure occurred, together with precise start and end times for each detected event.

## What it does

1. Load any EDF recording
2. Extract per-second signal features across all EEG channels
3. Run a trained XGBoost classifier on every second of the recording
4. Report seizure windows with start time, end time, and duration

No manual annotation or configuration needed. Open the dashboard, upload your file, click **Detect Seizures**.

## Get started

```bash
git clone https://github.com/ShadAhammed/EpilepsyDetector.git
cd EpilepsyDetector
python -m venv .venv
.venv\Scripts\activate        # Windows
# .venv/bin/activate          # Linux / macOS
pip install -e .
epilepsy dashboard
```

On Windows you can also double-click `run_dashboard.bat`.

The dashboard opens at **http://localhost:8501**.

## Training your own model

EpilepsyDetector ships without a bundled model because the training data (CHB-MIT EEG recordings) must be obtained through PhysioNet credentialing. Once you have EDF files with known seizure times, build the model in one command:

```bash
python scripts/build_chbmit_training.py --data-dir data --train --strategy smote
```

This reads your EDF files, extracts features, trains with SMOTE oversampling, and writes `models/seizure_model.joblib`. The dashboard will load it automatically.

Alternatively, if you have a labeled feature file (CSV or Excel with an `Out` column):

```bash
python scripts/save_model.py --features features.xlsx --strategy smote
```

## CLI usage

Detect seizures without the dashboard:

```bash
epilepsy detect --edf recording.edf --model models/seizure_model.joblib
```

Example output:

```
=== Seizure Detection Report ===
Recording: 3600 epochs (3600 seconds)

Result: 1 seizure period(s) detected

  1. Seizure: 2996s to 3036s (epochs 2996 to 3036, duration 40s)
```

Extract features from a labeled recording:

```bash
epilepsy extract-features --edf recording.edf --start 2996 --end 3036 --use-seconds --output features.parquet
```

## EEG data

The application works with any standard EDF file. It was trained and validated on the
[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (22 pediatric subjects, 182 seizures).
To reproduce the training or evaluate on new subjects, register and download from [PhysioNet](https://physionet.org/). See [docs/DATA.md](docs/DATA.md) for step-by-step instructions.

No EEG data is included in this repository.

## Project structure

```
src/epilepsy_detection/
  config/        # Settings (sample rate, filter bands, model path)
  data/          # EDF loader and seizure annotation parser
  features/      # Per-epoch feature extraction (23 channels x 10 features)
  models/        # Classifier, SMOTE, RUSBoost
  training/      # Model training and persistence
  evaluation/    # Metrics and reports
  pipeline/      # End-to-end orchestration
  api/           # FastAPI prediction endpoint
  cli.py         # Command-line interface
dashboard/       # Streamlit web dashboard
scripts/         # Training and model export helpers
tests/           # Unit and smoke tests (no EDF data required)
```

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest tests -v
```

## Author

Abu Shad Ahammed - [abu.ahammed@uni-siegen.de](mailto:abu.ahammed@uni-siegen.de)  
Chair of Embedded Systems, Universitat Siegen

## License

MIT - see [LICENSE](LICENSE).
