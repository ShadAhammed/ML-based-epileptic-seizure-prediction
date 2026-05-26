"""
Abstract base class for EEG feature extractors.

New feature extraction strategies should subclass :class:`FeatureExtractor`
and implement :meth:`extract`.  This enforces a consistent DataFrame output
contract across all feature implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from epilepsy_detection.data.annotations import SeizureInterval


class FeatureExtractor(ABC):
    """Abstract base for epoch-level EEG feature extractors.

    All concrete implementations must return a :class:`pandas.DataFrame` with:

    * A 1-based integer index named ``"ID"`` (epoch number).
    * One column per extracted feature.
    * An optional ``"Out"`` column (``0`` interictal / ``1`` ictal) present
      only when *seizure_interval* is provided (training mode).
    """

    @abstractmethod
    def extract(
        self,
        edf_path: str | Path,
        seizure_interval: SeizureInterval | None = None,
    ) -> pd.DataFrame:
        """Extract epoch-level features from an EDF recording.

        Args:
            edf_path: Path to the ``.edf`` recording.
            seizure_interval: Optional ictal window used to add an ``"Out"``
                label column.  Pass ``None`` for inference (no labels).

        Returns:
            DataFrame with shape ``(n_epochs, n_features)`` and an integer
            epoch index.
        """
