#!/usr/bin/env python
"""
Build labeled training features from local CHB-MIT EDF files + summary annotations,
then train and save models/seizure_model.joblib (notebook SMOTE workflow).

Usage::

    python scripts/build_chbmit_training.py
    python scripts/build_chbmit_training.py --data-dir data --train --strategy smote
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.data.annotations import AnnotationParser
from epilepsy_detection.features.epoch_features import EpochFeatureExtractor
from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline

# EDF basename -> subject folder (for *-summary.txt lookup)
_SUBJECT_RE = re.compile(r"^(chb\d+)")


def _subject_id(edf_name: str) -> str:
    match = _SUBJECT_RE.match(edf_name.lower().replace(".edf", ""))
    if not match:
        raise ValueError(f"Cannot infer subject from EDF name: {edf_name}")
    return match.group(1)


def _find_edfs(data_dir: Path) -> list[Path]:
    edfs = sorted(data_dir.glob("*.edf"))
    if not edfs:
        edfs = sorted(data_dir.rglob("*.edf"))
    return edfs


def _summary_path(data_dir: Path, subject: str) -> Path | None:
    candidates = [
        data_dir / "raw" / subject / f"{subject}-summary.txt",
        data_dir / subject / f"{subject}-summary.txt",
        data_dir / f"{subject}-summary.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def build_features(
    data_dir: Path,
    output_path: Path,
    records_file: Path | None = None,
) -> pd.DataFrame:
    """Extract and concatenate labeled epoch features (notebook ``FData`` style)."""
    parser = AnnotationParser()
    extractor = EpochFeatureExtractor()
    edfs = _find_edfs(data_dir)

    if records_file and records_file.exists():
        allowed = {
            line.strip().split("/")[-1]
            for line in records_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        edfs = [p for p in edfs if p.name in allowed]

    if not edfs:
        raise FileNotFoundError(f"No EDF files found under {data_dir}")

    summary_cache: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []

    for edf_path in edfs:
        subject = _subject_id(edf_path.name)
        if subject not in summary_cache:
            summary_path = _summary_path(data_dir, subject)
            if summary_path is None:
                print(f"  skip {edf_path.name}: no {subject}-summary.txt")
                continue
            summary_cache[subject] = parser.load_summary_by_file(summary_path)

        intervals = summary_cache[subject].get(edf_path.name, [])
        if not intervals:
            print(f"  skip {edf_path.name}: no seizures in summary")
            continue

        print(f"  extract {edf_path.name} ({len(intervals)} seizure interval(s))")
        chunk = extractor.extract_from_edf(edf_path, intervals)
        frames.append(chunk)

    if not frames:
        raise RuntimeError("No labeled features extracted. Check EDF paths and summary files.")

    # Use common columns only (channel count differs across CHB-MIT subjects).
    combined = pd.concat(frames, axis=0, join="inner")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        combined.to_parquet(output_path)
    elif suffix in {".xlsx", ".xls"}:
        combined.to_excel(output_path)
    else:
        combined.to_csv(output_path)

    n_pos = int(combined["Out"].sum())
    print(f"Saved {len(combined)} epochs ({n_pos} ictal) -> {output_path}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CHB-MIT training data and model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing EDF files (and data/raw/*/ summaries)",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "data" / "RECORDS-WITH-SEIZURES",
        help="Optional list of seizure EDF paths to include",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=ROOT / "data" / "features" / "training_features.parquet",
        help="Output labeled feature file",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Where to write seizure_model.joblib",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model after feature extraction",
    )
    parser.add_argument(
        "--strategy",
        default="smote",
        choices=["xgboost", "smote", "rusboost"],
        help="Training strategy (default: smote, matching notebook)",
    )
    parser.add_argument(
        "--only-local-edfs",
        action="store_true",
        default=True,
        help="Only process EDF files present under data-dir (default: true)",
    )
    args = parser.parse_args()

    records = None
    if args.only_local_edfs:
        local_names = {p.name for p in _find_edfs(args.data_dir)}
        records = args.data_dir / ".local_edf_filter.tmp"
        records.write_text("\n".join(sorted(local_names)), encoding="utf-8")
        records_file = records
    else:
        records_file = args.records if args.records.exists() else None

    print("Building training features from EDF + CHB-MIT summaries...")
    build_features(args.data_dir, args.features_out, records_file=records_file)

    if records and records.exists():
        records.unlink()

    if args.train:
        print(f"Training ({args.strategy})...")
        pipeline = DetectionPipeline(Settings.load())
        artifacts = pipeline.train(args.features_out, args.model_dir, strategy=args.strategy)
        print(f"Model saved: {artifacts.model_path}")


if __name__ == "__main__":
    main()
