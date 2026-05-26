"""Abstract feature extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureExtractor(ABC):
    """Base class for EEG feature extraction."""

    @abstractmethod
    def extract(self, *args, **kwargs) -> pd.DataFrame:
        """Extract features and return a labeled DataFrame."""
