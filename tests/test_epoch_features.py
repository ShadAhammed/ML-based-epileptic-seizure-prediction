"""Tests for epoch feature extraction."""

import numpy as np
import pandas as pd

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.data.annotations import SeizureInterval
from epilepsy_detection.features.epoch_features import EpochFeatureExtractor


def _synthetic_eeg(n_channels: int = 2, n_seconds: int = 30, sample_rate: int = 256) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_samples = n_seconds * sample_rate
    data = rng.standard_normal((n_channels, n_samples))
    return pd.DataFrame(data)


def test_epoch_feature_columns():
    settings = Settings(
        sample_rate=256,
        epoch_seconds=1,
        baseline_seconds=10,
        band_edges=[0.5, 12.5, 25.0],
    )
    extractor = EpochFeatureExtractor(settings)
    matrix = _synthetic_eeg(n_channels=2, n_seconds=30)
    interval = SeizureInterval(start_epoch=20, end_epoch=25)
    features = extractor.extract_from_matrix(matrix, interval)

    assert "Out" in features.columns
    assert features.loc[20, "Out"] == 1
    assert features.loc[10, "Out"] == 0
    assert "MeanCh1" in features.columns
    assert "Energy2" in str(features.columns)


def test_label_all_epochs_in_interval():
    settings = Settings(sample_rate=256, baseline_seconds=10)
    extractor = EpochFeatureExtractor(settings)
    matrix = _synthetic_eeg(n_seconds=15)
    interval = SeizureInterval(start_epoch=5, end_epoch=7)
    features = extractor.extract_from_matrix(matrix, interval)

    for epoch in [5, 6, 7]:
        assert features.loc[epoch, "Out"] == 1
    assert features.loc[1, "Out"] == 0
