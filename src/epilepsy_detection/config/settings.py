"""
Application settings for the epilepsy detection package.

Settings are resolved in this priority order:
    1. Environment variables prefixed with ``EPILEPSY_``
    2. A ``.env`` file in the working directory
    3. The ``config/default.yaml`` file
    4. Hardcoded defaults

Example usage::

    # Load from default YAML + env
    settings = Settings.load()

    # Load from a specific config file
    settings = Settings.load(config_path=Path("my_config.yaml"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class HyperparameterSearchConfig:
    """XGBoost hyperparameter search configuration used during model training.

    Attributes:
        folds: Number of cross-validation folds for StratifiedKFold.
        n_iter: Number of random parameter combinations to try in RandomizedSearchCV.
        params: Dictionary mapping XGBoost parameter names to candidate value lists.
            Defaults to the grid used in the original research notebook.
    """

    folds: int = 3
    n_iter: int = 5
    params: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "min_child_weight": [3, 5, 8],
            "gamma": [0.8, 1, 1.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_depth": [5, 7],
        }
    )


class Settings(BaseSettings):
    """Runtime settings with environment-variable overrides.

    All attributes can be overridden by exporting ``EPILEPSY_<ATTR_UPPER>=value``.
    Boolean flags use standard Pydantic parsing (``true`` / ``false`` / ``1`` / ``0``).

    Attributes:
        data_dir: Root directory for raw EDF recordings and cached features.
        model_dir: Directory where trained ``.joblib`` model artifacts are stored.
        reports_dir: Output directory for evaluation reports and plots.
        sample_rate: EEG sampling frequency in Hz (CHB-MIT is 256 Hz).
        epoch_seconds: Length of one analysis epoch in seconds.
        baseline_seconds: Length of the normal (non-ictal) baseline window at the
            start of each recording, used to compute per-channel ratio features.
        band_edges: Frequency band edge list [f0, f1, f2, ...] that defines
            ``len(edges) - 1`` FIR bandpass filters. Default = [0.5, 12.5, 25] Hz
            gives two bands: delta/theta (0.5–12.5 Hz) and alpha/beta (12.5–25 Hz).
        fir_filter_length: Number of FIR coefficients (taps). Must be odd for a
            Nuttall-windowed symmetric filter.
        fir_window: Scipy window name passed to :func:`scipy.signal.firwin`.
        rfe_min_features: Minimum number of features to retain after RFECV selection.
        rfe_cv_folds: Number of cross-validation folds inside RFECV.
        test_size: Fraction of the feature dataset held out for evaluation.
        random_state: Global random seed for reproducibility.
        hyperparameter_search: Nested config for XGBoost hyperparameter search.
    """

    model_config = SettingsConfigDict(
        env_prefix="EPILEPSY_",
        env_file=".env",
        extra="ignore",
    )

    # ---- Directory paths ------------------------------------------------
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    config_path: Path = Path("config/default.yaml")

    # ---- Signal processing parameters -----------------------------------
    sample_rate: int = 256
    epoch_seconds: int = 1
    baseline_seconds: int = 10
    band_edges: list[float] = Field(default=[0.5, 12.5, 25.0])
    fir_filter_length: int = 1001
    fir_window: str = "nuttall"

    # ---- Model / training parameters ------------------------------------
    rfe_min_features: int = 16
    rfe_cv_folds: int = 3
    test_size: float = 0.3
    random_state: int = 42
    hyperparameter_search: HyperparameterSearchConfig = Field(
        default_factory=HyperparameterSearchConfig
    )

    # ---- Derived properties ---------------------------------------------

    @property
    def samples_per_epoch(self) -> int:
        """Number of samples in one epoch: ``sample_rate * epoch_seconds``."""
        return self.sample_rate * self.epoch_seconds

    @property
    def baseline_epochs(self) -> int:
        """Number of baseline epochs at the start of each recording."""
        return self.baseline_seconds // self.epoch_seconds

    # ---- Factory --------------------------------------------------------

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Settings":
        """Create a Settings instance from YAML merged with environment variables.

        Args:
            config_path: Optional path to a YAML config file. Falls back to
                ``config/default.yaml`` when not provided.

        Returns:
            A fully resolved :class:`Settings` instance.
        """
        path = config_path or Path("config/default.yaml")
        data: dict[str, Any] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

        # Extract nested hyperparameter block before Pydantic sees it
        hp_raw = data.pop("hyperparameter_search", None)
        settings = cls(**{k: v for k, v in data.items() if v is not None})

        if hp_raw:
            settings.hyperparameter_search = HyperparameterSearchConfig(
                folds=hp_raw.get("folds", 3),
                n_iter=hp_raw.get("n_iter", 5),
                params=hp_raw.get("params", {}),
            )

        if config_path:
            settings.config_path = config_path

        return settings
