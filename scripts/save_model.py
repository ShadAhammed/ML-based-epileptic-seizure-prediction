#!/usr/bin/env python
"""
Train and save models/seizure_model.joblib from a labeled feature file.

The notebook trains in memory only; this script produces the file the
dashboard expects.

Usage::

    python scripts/save_model.py --features data/my_features.xlsx
    python scripts/save_model.py --features data/features.parquet --strategy smote

The feature file must contain an ``Out`` column (0 = interictal, 1 = ictal),
as produced by the notebook ``SzData`` function or ``epilepsy extract-features``
with ``--start`` / ``--end``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without pip install -e .
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epilepsy_detection.notebook_export import save_via_trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create models/seizure_model.joblib from labeled features."
    )
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="Labeled feature file (.xlsx, .csv, or .parquet) with Out column",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        type=Path,
        help="Output directory (default: models/)",
    )
    parser.add_argument(
        "--strategy",
        default="smote",
        choices=["xgboost", "smote", "rusboost"],
        help="Training strategy (default: smote, matching notebook cell 26)",
    )
    args = parser.parse_args()

    if not args.features.exists():
        print(f"Error: feature file not found: {args.features}", file=sys.stderr)
        sys.exit(1)

    print(f"Training from: {args.features}")
    print(f"Strategy: {args.strategy}")
    path = save_via_trainer(args.features, args.output_dir, strategy=args.strategy)
    print(f"Saved: {path}")
    print("You can now run:  epilepsy dashboard")


if __name__ == "__main__":
    main()
