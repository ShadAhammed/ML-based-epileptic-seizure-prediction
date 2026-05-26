# Data guide (public repository)

This repository is **source code only**. It does not and must not contain patient EEG recordings, extracted feature files from real subjects, or trained models fit on identifiable clinical data.

## Why there is no data in this repo

The [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) contains **de-identified but sensitive** pediatric clinical recordings. Distributing or re-hosting that data on a public GitHub repository would:

- Violate [PhysioNet usage requirements](https://physionet.org/about/licenses/)
- Raise privacy and ethical concerns for clinical EEG
- Be inappropriate for a public portfolio repo

Reviewers and employers can evaluate this project from the **architecture, code quality, tests, and documentation**. You run the full pipeline locally after obtaining data through the official channel below.

## Where to obtain the dataset

| Item | Link |
|------|------|
| Dataset | [CHB-MIT Scalp EEG Database v1.0.0](https://physionet.org/content/chbmit/1.0.0/) |
| Registration | [PhysioNet account](https://physionet.org/register/) |
| Credentialing | Complete the [Credentialed Health Data Use Agreement](https://physionet.org/settings/credentialing/) on PhysioNet |
| Citation | Goldberger et al., PhysioBank, PhysioToolkit, and PhysioNet (2000) — see dataset page |

### Download steps

1. Sign in at [PhysioNet](https://physionet.org/).
2. Open [chbmit/1.0.0](https://physionet.org/content/chbmit/1.0.0/) and click **Access the data** (after credentialing is approved).
3. Download subject folders (e.g. `chb01`, `chb02`, …) containing:
   - `*.edf` — EEG recordings
   - `*-summary.txt` — seizure onset/offset annotations
4. Extract archives locally into a directory **outside** git tracking, for example:

```
./data/raw/
  chb01/
    chb01_01.edf
    chb01_02.edf
    chb01-summary.txt
  chb02/
    ...
```

5. Copy `.env.example` to `.env` and set:

```env
EPILEPSY_DATA_DIR=./data/raw
```

### Optional: command-line download (PhysioNet)

If you use the [PhysioNet download tools](https://physionet.org/about/physionet-tools/), you can fetch files after configuring your credentials (see PhysioNet documentation). Example pattern:

```bash
# Install: pip install wfdb
# Follow PhysioNet instructions for your OS; then download specific files, e.g.:
# wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/chb01/
```

Always follow the license and citation requirements on the dataset page.

## What stays local (never commit)

These paths are listed in [`.gitignore`](../.gitignore) and must remain on your machine only:

| Path | Contents |
|------|----------|
| `data/raw/` | Downloaded EDF files |
| `data/features/` | Extracted feature Parquet/Excel from real recordings |
| `data/cache/` | API upload cache |
| `models/` | Trained `*.joblib` models |
| `reports/` | Evaluation outputs, plots |
| `.env` | Local paths and secrets |

## Running without clinical data

**Unit and smoke tests** use synthetic EEG-like signals and random feature matrices — no PhysioNet download required:

```bash
pip install -e ".[dev]"
pytest tests -v
```

To demo the CLI/API/GUI on real data, you must complete the download steps above.

## Using annotations

Seizure windows can be specified manually (epoch indices or seconds) or parsed from CHB-MIT `*-summary.txt` files via `AnnotationParser` in the package. Example summary lines:

```
Seizure Start Time: 2382 seconds
Seizure End Time: 2447 seconds
```

## Questions for recruiters / reviewers

- **Code and design**: see [ARCHITECTURE.md](ARCHITECTURE.md) and `src/epilepsy_detection/`.
- **Reproducibility**: `config/default.yaml`, `requirements.txt`, GitHub Actions CI.
- **Data**: obtained by the reviewer under PhysioNet terms; not redistributed in this repo.

Research notebooks live under `notebooks/` locally and are **gitignored** — use them to train `models/seizure_model.joblib` for the dashboard.

For questions about the original research context, contact the author listed in [README.md](../README.md).
