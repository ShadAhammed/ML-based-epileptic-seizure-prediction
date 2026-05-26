"""EDF file loading via MNE."""

from __future__ import annotations

from pathlib import Path

import mne


class EDFLoader:
    """Load scalp EEG recordings from EDF files."""

    def __init__(self, preload: bool = False) -> None:
        self.preload = preload

    def load(self, path: str | Path) -> mne.io.BaseRaw:
        """Load an EDF file and return an MNE Raw object."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"EDF file not found: {path}")
        return mne.io.read_raw_edf(path, preload=self.preload, verbose=False)

    def get_data_matrix(self, path: str | Path) -> tuple[mne.io.BaseRaw, list[list[float]]]:
        """Return raw object and channel x sample data matrix."""
        raw = self.load(path)
        return raw, raw.get_data().tolist()
