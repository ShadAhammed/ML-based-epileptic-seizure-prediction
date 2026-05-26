"""Epoch-level statistical and band-energy features (ported from SzData)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.data.annotations import SeizureInterval
from epilepsy_detection.data.edf_loader import EDFLoader
from epilepsy_detection.features.extractor import FeatureExtractor


class EpochFeatureExtractor(FeatureExtractor):
    """Extract per-epoch, per-channel features from EDF recordings."""

    def __init__(self, settings: Settings | None = None, loader: EDFLoader | None = None) -> None:
        self.settings = settings or Settings.load()
        self.loader = loader or EDFLoader(preload=False)
        self._band_edges = np.array(self.settings.band_edges, dtype=float)

    def extract_from_edf(
        self,
        edf_path: str | Path,
        seizure_interval: SeizureInterval | None = None,
    ) -> pd.DataFrame:
        """Extract features from an EDF file (optional labels for training only)."""
        _, matrix = self.loader.get_data_matrix(edf_path)
        data = pd.DataFrame(matrix)
        return self.extract_from_matrix(data, seizure_interval)

    def extract_from_edf_for_detection(self, edf_path: str | Path) -> pd.DataFrame:
        """Extract features from full EDF with no seizure labels (inference)."""
        return self.extract_from_edf(edf_path, seizure_interval=None)

    def extract_from_matrix(
        self,
        data: pd.DataFrame,
        seizure_interval: SeizureInterval | None = None,
    ) -> pd.DataFrame:
        """Extract features from a channel x samples DataFrame."""
        s = self.settings
        samples_per_epoch = s.samples_per_epoch
        n_epochs = int(data.shape[1] / samples_per_epoch)
        n_channels = data.shape[0]

        baseline_start = 0
        baseline_end = s.baseline_epochs * samples_per_epoch
        sz_epochs = set(seizure_interval.epoch_range) if seizure_interval else set()

        rows: list[dict] = []
        for epoch_idx in range(n_epochs):
            row: dict = {}
            epoch_start = epoch_idx * samples_per_epoch
            epoch_end = (epoch_idx + 1) * samples_per_epoch

            for ch in range(n_channels):
                norm_data = data.iloc[ch, baseline_start:baseline_end]
                epoch_data = data.iloc[ch, epoch_start:epoch_end]

                mean_normal = float(np.mean(norm_data))
                mean_data = float(np.mean(epoch_data))
                median_normal = float(np.median(norm_data))
                median_data = float(np.median(epoch_data))
                std_normal = float(np.std(norm_data)) or 1e-12
                std_data = float(np.std(epoch_data))
                max_normal = float(np.amax(norm_data)) or 1e-12
                max_data = float(np.amax(epoch_data))

                energies = self._band_energies(epoch_data.values)

                ch_num = ch + 1
                row[f"MeanCh{ch_num}"] = mean_data
                row[f"MeanDiff{ch_num}"] = mean_data / mean_normal if mean_normal else 0.0
                row[f"MedianCh{ch_num}"] = median_data
                row[f"MedianDiff{ch_num}"] = median_data / median_normal if median_normal else 0.0
                row[f"StdCh{ch_num}"] = std_data
                row[f"StdDiff{ch_num}"] = std_data / std_normal
                row[f"MaxCh{ch_num}"] = max_data
                row[f"MaxDiff{ch_num}"] = max_data / max_normal
                row[f"Energy1{ch_num}"] = energies[0]
                row[f"Energy2{ch_num}"] = energies[1]

            row["ID"] = epoch_idx + 1
            if seizure_interval is not None:
                row["Out"] = 1 if (epoch_idx + 1) in sz_epochs else 0
            rows.append(row)

        result = pd.DataFrame(rows)
        result = result.set_index("ID")
        return result

    def extract(
        self,
        edf_path: str | Path,
        seizure_interval: SeizureInterval | None = None,
    ) -> pd.DataFrame:
        return self.extract_from_edf(edf_path, seizure_interval)

    def _band_energies(self, epoch_samples: np.ndarray) -> list[float]:
        energies: list[float] = []
        nyq = self.settings.sample_rate
        for k in range(len(self._band_edges) - 1):
            coeffs = firwin(
                self.settings.fir_filter_length,
                [self._band_edges[k], self._band_edges[k + 1]],
                window=self.settings.fir_window,
                pass_zero=False,
                fs=nyq,
            )
            filtered = lfilter(coeffs, 1.0, epoch_samples)
            energies.append(float(np.sum(np.abs(filtered) ** 2)))
        return energies
