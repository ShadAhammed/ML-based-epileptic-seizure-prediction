"""
Seizure interval annotations for CHB-MIT and manual labeling.

The CHB-MIT database ships ``*-summary.txt`` files containing lines such as::

    Seizure Start Time: 2382 seconds
    Seizure End Time: 2447 seconds

:class:`AnnotationParser` can ingest these files as well as accept explicit
epoch or second ranges supplied by the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SeizureInterval:
    """An inclusive seizure window expressed in 1-based epoch indices.

    Epoch numbering matches the convention used throughout the original
    research notebook: epoch 1 corresponds to seconds 0–1, epoch 2 to
    seconds 1–2, and so on.

    Args:
        start_epoch: First ictal epoch (inclusive, 1-based).
        end_epoch: Last ictal epoch (inclusive, 1-based).
    """

    start_epoch: int
    end_epoch: int

    def contains(self, epoch_id: int) -> bool:
        """Return ``True`` if *epoch_id* falls inside this interval."""
        return self.start_epoch <= epoch_id <= self.end_epoch

    @property
    def epoch_range(self) -> range:
        """Iterable of all epoch IDs inside this interval."""
        return range(self.start_epoch, self.end_epoch + 1)

    def __str__(self) -> str:
        return f"SeizureInterval(epochs {self.start_epoch}–{self.end_epoch})"


class AnnotationParser:
    """Convert seizure time information into :class:`SeizureInterval` objects.

    Supports three input modes:

    * **Explicit seconds** - ``from_seconds(start_sec, end_sec)``
    * **Explicit epochs** - ``from_epochs(start_epoch, end_epoch)``
    * **CHB-MIT summary file** - ``load_summary_file(path)``
    """

    def from_seconds(
        self,
        start_sec: int,
        end_sec: int,
        epoch_seconds: int = 1,
    ) -> SeizureInterval:
        """Create a :class:`SeizureInterval` from absolute second offsets.

        Args:
            start_sec: Seizure onset in seconds from recording start.
            end_sec: Seizure offset in seconds from recording start.
            epoch_seconds: Epoch duration in seconds (default 1).

        Returns:
            Corresponding :class:`SeizureInterval` in epoch indices.
        """
        return SeizureInterval(
            start_epoch=start_sec // epoch_seconds,
            end_epoch=end_sec // epoch_seconds,
        )

    def from_epochs(self, start_epoch: int, end_epoch: int) -> SeizureInterval:
        """Create a :class:`SeizureInterval` directly from epoch indices.

        Args:
            start_epoch: First ictal epoch (1-based).
            end_epoch: Last ictal epoch (1-based, inclusive).

        Returns:
            A frozen :class:`SeizureInterval`.
        """
        return SeizureInterval(start_epoch=start_epoch, end_epoch=end_epoch)

    def load_summary_file(self, path: str | Path) -> list[SeizureInterval]:
        """Parse a CHB-MIT ``*-summary.txt`` file into a list of intervals.

        The parser is lenient: it handles both "Start Time" and "Start:"
        variants and ignores unrecognised lines.

        Args:
            path: Path to the summary text file.

        Returns:
            Ordered list of :class:`SeizureInterval` objects.
        """
        path = Path(path)
        intervals: list[SeizureInterval] = []
        current_start: int | None = None

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            start_match = re.search(
                r"Seizure(?:\s+\d+)?\s+Start[^:]*:\s*(\d+)", line, re.I
            )
            end_match = re.search(
                r"Seizure(?:\s+\d+)?\s+End[^:]*:\s*(\d+)", line, re.I
            )

            if start_match:
                current_start = int(start_match.group(1))
            elif end_match and current_start is not None:
                intervals.append(
                    SeizureInterval(
                        start_epoch=current_start,
                        end_epoch=int(end_match.group(1)),
                    )
                )
                current_start = None

        return intervals

    def load_summary_by_file(self, path: str | Path) -> dict[str, list[SeizureInterval]]:
        """Parse a CHB-MIT summary file into per-EDF seizure intervals.

        Args:
            path: Path to ``*-summary.txt``.

        Returns:
            Mapping from EDF basename (e.g. ``chb01_03.edf``) to seizure intervals
            in that recording.  Files with no seizures map to an empty list.
        """
        path = Path(path)
        by_file: dict[str, list[SeizureInterval]] = {}
        current_file: str | None = None
        current_start: int | None = None

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            file_match = re.search(r"File\s+Name:\s*(\S+)", line, re.I)
            if file_match:
                current_file = file_match.group(1)
                by_file.setdefault(current_file, [])
                current_start = None
                continue

            if current_file is None:
                continue

            start_match = re.search(
                r"Seizure(?:\s+\d+)?\s+Start[^:]*:\s*(\d+)", line, re.I
            )
            end_match = re.search(
                r"Seizure(?:\s+\d+)?\s+End[^:]*:\s*(\d+)", line, re.I
            )

            if start_match:
                current_start = int(start_match.group(1))
            elif end_match and current_start is not None:
                by_file[current_file].append(
                    SeizureInterval(
                        start_epoch=current_start,
                        end_epoch=int(end_match.group(1)),
                    )
                )
                current_start = None

        return by_file

    @staticmethod
    def seizure_epoch_set(
        intervals: SeizureInterval | Sequence[SeizureInterval] | None,
    ) -> set[int]:
        """Collect all 1-based epoch IDs labeled ictal across one or more intervals."""
        if intervals is None:
            return set()
        if isinstance(intervals, SeizureInterval):
            return set(intervals.epoch_range)
        merged: set[int] = set()
        for interval in intervals:
            merged.update(interval.epoch_range)
        return merged
