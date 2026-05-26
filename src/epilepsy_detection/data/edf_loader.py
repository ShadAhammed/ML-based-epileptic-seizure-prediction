"""
EDF file loading via MNE-Python.

The European Data Format (EDF) is the standard file format used by the
CHB-MIT Scalp EEG Database and most clinical EEG acquisition systems.
This module wraps MNE's EDF reader to provide a simple, testable interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mne


class EDFLoader:
    """Load scalp EEG recordings from EDF/EDF+ files.

    Args:
        preload: If ``True``, load all data into RAM on open.
            ``False`` (default) uses memory-mapped access and is preferred
            for large recordings where only a subset of the data is needed.
    """

    def __init__(self, preload: bool = False) -> None:
        self.preload = preload

    def load(self, path: str | Path) -> mne.io.BaseRaw:
        """Open an EDF file and return an MNE :class:`~mne.io.BaseRaw` object.

        Args:
            path: Path to the ``.edf`` file.

        Returns:
            MNE Raw object with channel info and data accessor.

        Raises:
            FileNotFoundError: If the file does not exist at *path*.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"EDF file not found: {path}")
        return mne.io.read_raw_edf(path, preload=self.preload, verbose=False)

    def get_data_matrix(self, path: str | Path) -> tuple[mne.io.BaseRaw, np.ndarray]:
        """Open an EDF file and return the raw object together with the data matrix.

        Args:
            path: Path to the ``.edf`` file.

        Returns:
            A 2-tuple ``(raw, data)`` where *data* is a
            ``float64`` array of shape ``(n_channels, n_samples)``.
        """
        raw = self.load(path)
        # Materialise data as numpy array; avoids repeated disk reads during
        # epoch iteration even when preload=False.
        data: np.ndarray = raw.get_data()
        return raw, data

    def recording_info(self, path: str | Path) -> dict:
        """Return a lightweight metadata summary without loading signal data.

        Args:
            path: Path to the ``.edf`` file.

        Returns:
            Dictionary with keys: ``n_channels``, ``sample_rate``,
            ``duration_seconds``, ``channel_names``.
        """
        raw = mne.io.read_raw_edf(Path(path), preload=False, verbose=False)
        return {
            "n_channels": len(raw.ch_names),
            "sample_rate": int(raw.info["sfreq"]),
            "duration_seconds": raw.n_times / raw.info["sfreq"],
            "channel_names": raw.ch_names,
        }
