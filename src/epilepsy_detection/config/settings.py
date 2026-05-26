"""Application settings loaded from YAML and environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class HyperparameterSearchConfig:
    folds: int = 3
    n_iter: int = 5
    params: dict[str, list[Any]] = field(default_factory=dict)


class Settings(BaseSettings):
    """Runtime settings with env overrides."""

    model_config = SettingsConfigDict(
        env_prefix="EPILEPSY_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    config_path: Path = Path("config/default.yaml")

    sample_rate: int = 256
    epoch_seconds: int = 1
    baseline_seconds: int = 10
    band_edges: list[float] = Field(default=[0.5, 12.5, 25.0])
    fir_filter_length: int = 1001
    fir_window: str = "nuttall"
    rfe_min_features: int = 16
    rfe_cv_folds: int = 3
    test_size: float = 0.3
    random_state: int = 42
    hyperparameter_search: HyperparameterSearchConfig = Field(
        default_factory=HyperparameterSearchConfig
    )

    @classmethod
    def load(cls, config_path: Path | None = None) -> Settings:
        """Load settings from YAML merged with environment."""
        path = config_path or Path("config/default.yaml")
        data: dict[str, Any] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

        hp = data.pop("hyperparameter_search", None)
        settings = cls(**{k: v for k, v in data.items() if v is not None})
        if hp:
            settings.hyperparameter_search = HyperparameterSearchConfig(
                folds=hp.get("folds", 3),
                n_iter=hp.get("n_iter", 5),
                params=hp.get("params", {}),
            )
        if config_path:
            settings.config_path = config_path
        return settings

    @property
    def samples_per_epoch(self) -> int:
        return self.sample_rate * self.epoch_seconds

    @property
    def baseline_epochs(self) -> int:
        return self.baseline_seconds // self.epoch_seconds
