# Quickstart — seizure detection

## What this software does

**Input:** EDF recording + pre-trained model  
**Output:** When–when seizure periods occur (start/end in seconds and epochs)

## 1. Install

```bash
git clone https://github.com/ShadAhammed/ML-based-epileptic-seizure-prediction.git
cd ML-based-epileptic-seizure-prediction
pip install -e .
```

## 2. Open the dashboard

```bash
epilepsy dashboard
```

Browser opens at **http://localhost:8501**:

1. Set model path (default `models/seizure_model.joblib`)
2. Upload your `.edf` file
3. Click **Detect Seizures**
4. View intervals, timeline chart, download CSV

**Windows:** double-click `run_dashboard.bat`

## 3. Pre-trained model

Train with your **local** notebook under `notebooks/` (not in git). Save to:

```
models/seizure_model.joblib
```

See [docs/DATA.md](../docs/DATA.md) for CHB-MIT download steps.

## 4. CLI (alternative)

```bash
epilepsy detect --edf data/raw/chb01/chb01_03.edf --model models/seizure_model.joblib
```
