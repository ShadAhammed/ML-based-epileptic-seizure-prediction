"""Tests for seizure window detection from predictions."""

import numpy as np

from epilepsy_detection.detection.intervals import find_seizure_intervals, format_detection_report
from epilepsy_detection.features.epoch_features import EpochFeatureExtractor
from epilepsy_detection.config.settings import Settings
import pandas as pd


def test_find_single_seizure_window():
    epochs = np.arange(1, 11)
    preds = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    intervals = find_seizure_intervals(epochs, preds)
    assert len(intervals) == 2
    assert intervals[0].start_epoch == 3
    assert intervals[0].end_epoch == 5
    assert intervals[0].start_seconds == 2
    assert intervals[0].end_seconds == 5
    assert intervals[1].start_epoch == 8
    assert intervals[1].end_epoch == 9


def test_unlabeled_feature_extraction():
    settings = Settings(sample_rate=256, baseline_seconds=10)
    extractor = EpochFeatureExtractor(settings)
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(rng.standard_normal((2, 256 * 20)))
    features = extractor.extract_from_matrix(matrix, seizure_interval=None)
    assert "Out" not in features.columns
    assert len(features) == 20


def test_format_empty_report():
    report = format_detection_report([], 100, 100)
    assert "No seizure" in report
