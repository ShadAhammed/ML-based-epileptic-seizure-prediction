# Quickstart — seizure detection

## What this software does

**Input:** EDF recording + pre-trained model  
**Output:** When–when seizure periods occur (start/end in seconds and epochs)

Training is done in the legacy notebook — not in the main app flow.

## 1. Install

```bash
pip install -e .
```

## 2. Pre-trained model

After training in `notebooks/legacy/Epilepsy.ipynb`, place your saved model at:

```
models/seizure_model.joblib
```

See [docs/DATA.md](../docs/DATA.md) for obtaining CHB-MIT EDF files.

## 3. Detect (CLI)

```bash
epilepsy detect \
  --edf data/raw/chb01/chb01_03.edf \
  --model models/seizure_model.joblib
```

## 4. Detect (GUI)

```bash
epilepsy gui
```

1. Browse → your `.edf` file  
2. Browse → `models/seizure_model.joblib`  
3. Click **Detect Seizures**  
4. Read the **from–to** windows in the results panel  

## 5. Output files

| File | Content |
|------|---------|
| Console / GUI | Seizure windows (seconds + epochs) |
| `reports/detection_result.csv` | Per-epoch prediction (0/1 + probability) |
