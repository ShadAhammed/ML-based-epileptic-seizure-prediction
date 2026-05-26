# Notebooks (local only)

Research notebooks are **gitignored** and stay on your machine.

## Why the dashboard says "Model not found"

The notebook **trains the model in RAM** (`Smodel`, `scaler`, `f`) but **does not save** `models/seizure_model.joblib` to disk. The dashboard only loads that file.

## Option A — Save from the notebook (recommended)

1. Open `legacy/Epilepsy.ipynb` and run all training cells through **cell 26** (SMOTE + `Smodel`).
2. Add a **new cell** at the end and paste the contents of [`save_model_cell.py`](save_model_cell.py), then run it.
3. Confirm the file exists: `models/seizure_model.joblib`
4. Restart the dashboard: `epilepsy dashboard`

## Option B — Save from a feature file (no notebook kernel)

If you have a labeled Excel/Parquet file with an `Out` column (from `SzData` or feature export):

```bash
python scripts/save_model.py --features path/to/your_features.xlsx --strategy smote
```

## What gets saved

The `.joblib` file contains:

- `MinMaxScaler` (same preprocessing as the notebook)
- `RFECV` feature selector
- Trained classifier (`Smodel` / XGBoost)
- Feature column names for validation at inference

This is **not** committed to GitHub (see `.gitignore`).
