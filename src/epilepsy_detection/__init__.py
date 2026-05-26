"""Epilepsy detection: ML-based ictal period detection from scalp EEG."""

from epilepsy_detection.config.settings import Settings
from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline

__version__ = "1.0.0"
__all__ = ["DetectionPipeline", "Settings", "__version__"]
