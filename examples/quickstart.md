# Quickstart

## 1. Install

```bash
pip install -e .
```

## 2. Prepare data

This repo does **not** ship clinical EEG data. See **[docs/DATA.md](../docs/DATA.md)** for:

- PhysioNet registration and credentialing
- CHB-MIT download links and folder layout
- What must stay local (never committed to git)

After download, place EDF files under `data/raw/` and set in `.env`:

```
EPILEPSY_DATA_DIR=./data/raw
EPILEPSY_MODEL_DIR=./models
```

## 3. Extract features

Using epoch indices (as in the legacy notebook example `2382–2447`):

```bash
epilepsy extract-features \
  --edf data/raw/chb01/chb01_03.edf \
  --start 2382 \
  --end 2447 \
  --output data/features.parquet
```

## 4. Train

```bash
epilepsy train --features data/features.parquet --output-dir models/
```

## 5. Evaluate

```bash
epilepsy evaluate \
  --model models/seizure_model.joblib \
  --features data/features.parquet \
  --report-dir reports/
```

## 6. API

```bash
epilepsy serve-api --model-dir models/
curl http://127.0.0.1:8000/health
```

Upload features for prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/features.parquet"
```

## 7. GUI

```bash
epilepsy gui
```

Use Browse to select EDF and feature files, set seizure window, then run Extract / Train / Predict / Evaluate.
