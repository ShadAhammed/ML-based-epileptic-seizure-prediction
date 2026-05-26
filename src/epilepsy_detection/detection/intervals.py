"""
Post-processing of per-epoch predictions into human-readable seizure windows.

After the XGBoost classifier labels each 1-second epoch as ictal (1) or
interictal (0), this module groups consecutive ictal epochs into contiguous
:class:`DetectedSeizure` intervals and formats them for display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epilepsy_detection.data.annotations import SeizureInterval


@dataclass(frozen=True)
class DetectedSeizure:
    """A contiguous predicted ictal (seizure) time window.

    All time values are in **seconds** from the start of the recording.

    Attributes:
        start_epoch: First predicted ictal epoch (1-based).
        end_epoch: Last predicted ictal epoch (1-based, inclusive).
        start_seconds: Onset time in seconds from recording start.
        end_seconds: Offset time in seconds from recording start.
        duration_seconds: Total ictal duration in seconds.
    """

    start_epoch: int
    end_epoch: int
    start_seconds: int
    end_seconds: int
    duration_seconds: int

    def __str__(self) -> str:
        return (
            f"Seizure: {self.start_seconds}s to {self.end_seconds}s "
            f"(epochs {self.start_epoch} to {self.end_epoch}, "
            f"duration {self.duration_seconds}s)"
        )

    def to_dict(self) -> dict:
        """Return a plain dictionary representation suitable for JSON serialisation."""
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
        }


def find_seizure_intervals(
    epoch_ids: np.ndarray | pd.Index,
    predicted: np.ndarray,
    epoch_seconds: int = 1,
    min_duration_epochs: int = 1,
) -> list[DetectedSeizure]:
    """Merge per-epoch binary predictions into contiguous ictal windows.

    Args:
        epoch_ids: 1-based epoch identifiers, typically the DataFrame index.
        predicted: Binary prediction array (0 = interictal, 1 = ictal).
        epoch_seconds: Duration of one epoch in seconds.
        min_duration_epochs: Minimum run length to count as a detected
            seizure.  Use values > 1 to filter out single-epoch false
            positives.

    Returns:
        Sorted list of :class:`DetectedSeizure` objects.
    """
    epochs = np.asarray(epoch_ids, dtype=int)
    preds = np.asarray(predicted, dtype=int)

    # Walk through the prediction sequence and collect (start, end) pairs
    # of contiguous ictal runs.
    intervals: list[tuple[int, int]] = []
    run_start: int | None = None

    for epoch_id, pred in zip(epochs, preds):
        if pred == 1:
            if run_start is None:
                run_start = int(epoch_id)
        elif run_start is not None:
            run_end = int(epoch_id) - 1
            if run_end - run_start + 1 >= min_duration_epochs:
                intervals.append((run_start, run_end))
            run_start = None

    # Handle a run that extends to the very end of the recording.
    if run_start is not None:
        run_end = int(epochs[-1])
        if run_end - run_start + 1 >= min_duration_epochs:
            intervals.append((run_start, run_end))

    detected: list[DetectedSeizure] = []
    for start_epoch, end_epoch in intervals:
        # Epoch IDs are 1-based: epoch 1 = seconds 0..1, epoch N = seconds N-1..N.
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
    """Generate a plain-text summary of detected seizure windows.

    Args:
        seizures: List of detected seizure objects.
        recording_epochs: Total number of epochs in the recording.
        recording_seconds: Recording duration in seconds.

    Returns:
        Multi-line human-readable report string.
    """
    lines = [
        "=== Seizure Detection Report ===",
        f"Recording: {recording_epochs} epochs ({recording_seconds} seconds)",
        "",
    ]
    if not seizures:
        lines.append("Result: No seizure activity detected.")
        return "\n".join(lines)

    lines.append(f"Result: {len(seizures)} seizure period(s) detected\n")
    for i, sz in enumerate(seizures, 1):
        lines.append(f"  {i}. {sz}")

    total_ictal = sum(s.duration_seconds for s in seizures)
    lines.append(f"\nTotal ictal duration: {total_ictal}s")

    return "\n".join(lines)


def to_seizure_intervals(seizures: list[DetectedSeizure]) -> list[SeizureInterval]:
    """Convert a list of :class:`DetectedSeizure` to :class:`SeizureInterval` objects."""
    return [
        SeizureInterval(start_epoch=s.start_epoch, end_epoch=s.end_epoch)
        for s in seizures
    ]
