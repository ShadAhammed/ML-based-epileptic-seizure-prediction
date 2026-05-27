# Quickstart — seizure detection

## What this software does

**Input:** EDF recording + pre-trained model  
**Output:** When–when seizure periods occur (start/end in seconds and epochs)

## 1. Install

```bash
git clone https://github.com/ShadAhammed/EpilepsyDetector.git
cd EpilepsyDetector
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

## 3. Create the model file (required once)

The notebook trains in memory; the dashboard needs a file on disk.

**From the notebook** (after SMOTE cell): paste and run [`scripts/save_model_cell.py`](../scripts/save_model_cell.py) in a new cell.

**From a feature file:**

```bash
python scripts/save_model.py --features path/to/features.xlsx --strategy smote
```

Output: `models/seizure_model.joblib`

The `notebooks/` folder is gitignored (local research only).

## 4. CLI (alternative)

```bash
epilepsy detect --edf data/raw/chb01/chb01_03.edf --model models/seizure_model.joblib
```
