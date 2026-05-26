"""Merge per-epoch predictions into seizure time windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epilepsy_detection.data.annotations import SeizureInterval


@dataclass(frozen=True)
class DetectedSeizure:
    """A contiguous predicted ictal period."""

    start_epoch: int
    end_epoch: int
    start_seconds: int
    end_seconds: int
    duration_seconds: int

    def __str__(self) -> str:
        return (
            f"Seizure: {self.start_seconds}s – {self.end_seconds}s "
            f"(epochs {self.start_epoch}–{self.end_epoch}, "
            f"duration {self.duration_seconds}s)"
        )


def find_seizure_intervals(
    epoch_ids: np.ndarray | pd.Index,
    predicted: np.ndarray,
    epoch_seconds: int = 1,
    min_duration_epochs: int = 1,
) -> list[DetectedSeizure]:
    """Find contiguous epochs predicted as seizure (label 1)."""
    epochs = np.asarray(epoch_ids, dtype=int)
    preds = np.asarray(predicted, dtype=int)

    intervals: list[tuple[int, int]] = []
    start: int | None = None

    for epoch_id, pred in zip(epochs, preds):
        if pred == 1:
            if start is None:
                start = int(epoch_id)
        elif start is not None:
            end_epoch = int(epoch_id) - 1
            if end_epoch - start + 1 >= min_duration_epochs:
                intervals.append((start, end_epoch))
            start = None

    if start is not None:
        end_epoch = int(epochs[-1])
        if end_epoch - start + 1 >= min_duration_epochs:
            intervals.append((start, end_epoch))

    detected: list[DetectedSeizure] = []
    for start_epoch, end_epoch in intervals:
        start_sec = (start_epoch - 1) * epoch_seconds
        end_sec = end_epoch * epoch_seconds
        detected.append(
            DetectedSeizure(
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                start_seconds=start_sec,
                end_seconds=end_sec,
                duration_seconds=end_sec - start_sec,
            )
        )
    return detected


def format_detection_report(
    seizures: list[DetectedSeizure],
    recording_epochs: int,
    recording_seconds: int,
) -> str:
    """Human-readable summary of detected seizure windows."""
    lines = [
        "=== Seizure detection result ===",
        f"Recording length: {recording_epochs} epochs (~{recording_seconds} seconds)",
        "",
    ]
    if not seizures:
        lines.append("No seizure activity detected in this recording.")
        return "\n".join(lines)

    lines.append(f"Detected {len(seizures)} seizure period(s):\n")
    for i, sz in enumerate(seizures, 1):
        lines.append(f"  {i}. {sz}")
    return "\n".join(lines)


def to_seizure_intervals(seizures: list[DetectedSeizure]) -> list[SeizureInterval]:
    return [
        SeizureInterval(start_epoch=s.start_epoch, end_epoch=s.end_epoch) for s in seizures
    ]
