"""
Epoch-level statistical and band-energy feature extraction.

This module is a production-quality port of the ``SzData`` function from
the original research notebook.  Each 1-second epoch of the recording is
characterised by 10 features per EEG channel:

==========  ============================================================
Feature     Description
==========  ============================================================
MeanCh      Absolute mean amplitude of the epoch.
MeanDiff    Ratio of epoch mean to 10-second baseline mean.
MedianCh    Absolute median amplitude of the epoch.
MedianDiff  Ratio of epoch median to baseline median.
StdCh       Standard deviation of the epoch.
StdDiff     Ratio of epoch std to baseline std.
MaxCh       Peak absolute amplitude of the epoch.
MaxDiff     Ratio of epoch max to baseline max.
Energy1     Signal energy in the low-frequency FIR band (0.5–12.5 Hz).
Energy2     Signal energy in the high-frequency FIR band (12.5–25 Hz).
==========  ============================================================

The first ``baseline_seconds`` (default 10) of the recording are assumed
to be seizure-free and serve as the baseline reference for all ratio
features.  This assumption matches the CHB-MIT recordings used during
training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.data.annotations import AnnotationParser, SeizureInterval
from epilepsy_detection.data.edf_loader import EDFLoader
from epilepsy_detection.features.extractor import FeatureExtractor

# Small constant added to denominators to prevent division-by-zero when the
# baseline signal is flat (e.g. an artificially generated test signal).
_EPSILON = 1e-12


class EpochFeatureExtractor(FeatureExtractor):
    """Extract per-epoch, per-channel features from EDF recordings.

    This extractor is the primary feature engineering component.  It operates
    in two modes:

    * **Inference mode** (``seizure_interval=None``): extracts features for
      every epoch without an ``"Out"`` label column.  This is the path used
      by the detection dashboard and CLI.
    * **Training mode** (``seizure_interval`` provided): identical extraction
      plus an ``"Out"`` column (0 / 1) for supervised learning.

    Args:
        settings: Application settings.  Loaded from ``config/default.yaml``
            when not provided.
        loader: EDF loader instance.  Defaults to :class:`~epilepsy_detection.data.edf_loader.EDFLoader`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        loader: EDFLoader | None = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.loader = loader or EDFLoader(preload=False)
        # Pre-compute band edges as a numpy array for performance
        self._band_edges = np.array(self.settings.band_edges, dtype=float)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        edf_path: str | Path,
        seizure_interval: SeizureInterval | None = None,
    ) -> pd.DataFrame:
        """Extract epoch features from an EDF file (FeatureExtractor interface).

        Args:
            edf_path: Path to the ``.edf`` recording.
            seizure_interval: Optional ictal label window.

        Returns:
            Feature DataFrame with epoch-ID index.
        """
        return self.extract_from_edf(edf_path, seizure_interval)

    def extract_from_edf(
        self,
        edf_path: str | Path,
        seizure_interval: SeizureInterval | None = None,
    ) -> pd.DataFrame:
        """Load an EDF file and extract epoch-level features.

        Args:
            edf_path: Path to the ``.edf`` recording.
            seizure_interval: Ictal window for training labels.  Pass
                ``None`` (default) for inference.

        Returns:
            Feature DataFrame with shape ``(n_epochs, n_features)``.
        """
        _, data_matrix = self.loader.get_data_matrix(edf_path)
        data = pd.DataFrame(data_matrix)
        return self.extract_from_matrix(data, seizure_interval)

    def extract_from_matrix(
        self,
        data: pd.DataFrame,
        seizure_interval: SeizureInterval | Sequence[SeizureInterval] | None = None,
    ) -> pd.DataFrame:
        """Extract features from a pre-loaded channel-x-samples DataFrame.

        This method is the computational core.  It is called by
        :meth:`extract_from_edf` and is also exposed for unit testing with
        synthetic signal matrices.

        Args:
            data: DataFrame of shape ``(n_channels, n_samples)``.  Each row
                is one EEG channel; each column is one sample point.
            seizure_interval: Optional ictal label window.

        Returns:
            Feature DataFrame indexed by 1-based epoch ID.
        """
        s = self.settings
        samples_per_epoch = s.samples_per_epoch
        n_epochs = int(data.shape[1] / samples_per_epoch)
        n_channels = data.shape[0]

        # Baseline window: first N seconds of the recording (assumed seizure-free).
        # The baseline statistics are used to compute ratio features for every epoch,
        # capturing relative changes rather than absolute signal amplitudes.
        baseline_end = s.baseline_epochs * samples_per_epoch
        # Guard: if the recording is shorter than the baseline window, use
        # whatever is available.
        baseline_end = min(baseline_end, data.shape[1])

        # Pre-compute baseline statistics for each channel to avoid
        # repeating the computation inside the epoch loop.
        baseline_stats = self._compute_baseline_stats(data, 0, baseline_end, n_channels)

        # Epoch label set - empty for inference, populated for training.
        sz_epochs = AnnotationParser.seizure_epoch_set(seizure_interval)

        rows: list[dict] = []
        for epoch_idx in range(n_epochs):
            epoch_start = epoch_idx * samples_per_epoch
            epoch_end = epoch_start + samples_per_epoch
            row = self._extract_epoch_row(
                data, epoch_start, epoch_end, n_channels, baseline_stats
            )
            row["ID"] = epoch_idx + 1

            # Only add the label column in training mode
            if seizure_interval is not None:
                row["Out"] = 1 if (epoch_idx + 1) in sz_epochs else 0

            rows.append(row)

        result = pd.DataFrame(rows).set_index("ID")
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_baseline_stats(
        self,
        data: pd.DataFrame,
        start: int,
        end: int,
        n_channels: int,
    ) -> list[dict[str, float]]:
        """Pre-compute per-channel baseline statistics.

        Extracting these once per recording (rather than per epoch) reduces
        the O(n_epochs * n_channels) computation to O(n_channels).
        """
        stats = []
        for ch in range(n_channels):
            baseline = data.iloc[ch, start:end].values.astype(float)
            stats.append(
                {
                    "mean": float(np.mean(baseline)),
                    "median": float(np.median(baseline)),
                    "std": float(np.std(baseline)) or _EPSILON,
                    "max": float(np.amax(np.abs(baseline))) or _EPSILON,
                }
            )
        return stats

    def _extract_epoch_row(
        self,
        data: pd.DataFrame,
        epoch_start: int,
        epoch_end: int,
        n_channels: int,
        baseline_stats: list[dict[str, float]],
    ) -> dict:
        """Compute feature values for one epoch across all channels."""
        row: dict = {}
        for ch in range(n_channels):
            epoch_signal = data.iloc[ch, epoch_start:epoch_end].values.astype(float)
            bl = baseline_stats[ch]

            mean_val = float(np.mean(epoch_signal))
            median_val = float(np.median(epoch_signal))
            std_val = float(np.std(epoch_signal))
            max_val = float(np.amax(np.abs(epoch_signal)))

            # Band-pass energy features using pre-designed FIR filters
            energies = self._band_energies(epoch_signal)

            ch_num = ch + 1
            row[f"MeanCh{ch_num}"] = mean_val
            row[f"MeanDiff{ch_num}"] = mean_val / bl["mean"] if abs(bl["mean"]) > _EPSILON else 0.0
            row[f"MedianCh{ch_num}"] = median_val
            row[f"MedianDiff{ch_num}"] = (
                median_val / bl["median"] if abs(bl["median"]) > _EPSILON else 0.0
            )
            row[f"StdCh{ch_num}"] = std_val
            row[f"StdDiff{ch_num}"] = std_val / bl["std"]
            row[f"MaxCh{ch_num}"] = max_val
            row[f"MaxDiff{ch_num}"] = max_val / bl["max"]
            row[f"Energy1{ch_num}"] = energies[0]
            row[f"Energy2{ch_num}"] = energies[1]

        return row

    def _band_energies(self, epoch_samples: np.ndarray) -> list[float]:
        """Compute signal energy in each frequency band via FIR filtering.

        A Nuttall-windowed FIR bandpass filter is applied (matching the
        notebook implementation: ``firwin(1001, [...], window='nuttall',
        pass_zero=False, nyq=256)``).  Energy is computed as the sum of
        squared absolute values of the filtered signal.

        Args:
            epoch_samples: 1-D float array of one epoch's raw samples.

        Returns:
            List of energy values, one per band defined by ``band_edges``.
        """
        energies: list[float] = []
        nyq = float(self.settings.sample_rate)

        for k in range(len(self._band_edges) - 1):
            # Design a symmetric FIR bandpass filter for the k-th band.
            # ``fs=nyq`` is used instead of the deprecated ``nyq=`` kwarg
            # (SciPy >= 1.12 removed the ``nyq`` argument).
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
