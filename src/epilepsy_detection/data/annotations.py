"""Seizure interval annotations for CHB-MIT and manual labeling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeizureInterval:
    """Inclusive seizure window in epoch indices (1-based, matching notebook)."""

    start_epoch: int
    end_epoch: int

    def contains(self, epoch_id: int) -> bool:
        return self.start_epoch <= epoch_id <= self.end_epoch

    @property
    def epoch_range(self) -> range:
        return range(self.start_epoch, self.end_epoch + 1)


def parse_chb_mit_summary_line(line: str) -> SeizureInterval | None:
    """Parse lines like 'Seizure Start: 2382 seconds' from CHB-MIT summary files."""
    start_match = re.search(r"Seizure\s+Start:\s*(\d+)\s*seconds", line, re.I)
    end_match = re.search(r"Seizure\s+End:\s*(\d+)\s*seconds", line, re.I)
    if start_match and end_match:
        return SeizureInterval(
            start_epoch=int(start_match.group(1)),
            end_epoch=int(end_match.group(1)),
        )
    return None


class AnnotationParser:
    """Load seizure intervals from summary files or explicit ranges."""

    def from_seconds(
        self,
        start_sec: int,
        end_sec: int,
        epoch_seconds: int = 1,
    ) -> SeizureInterval:
        start_epoch = start_sec // epoch_seconds
        end_epoch = end_sec // epoch_seconds
        return SeizureInterval(start_epoch=start_epoch, end_epoch=end_epoch)

    def from_epochs(self, start_epoch: int, end_epoch: int) -> SeizureInterval:
        return SeizureInterval(start_epoch=start_epoch, end_epoch=end_epoch)

    def load_summary_file(self, path: str | Path) -> list[SeizureInterval]:
        """Parse CHB-MIT *-summary.txt style files."""
        path = Path(path)
        intervals: list[SeizureInterval] = []
        current_start: int | None = None

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            start_match = re.search(r"Seizure\s+Start:\s*(\d+)", line, re.I)
            end_match = re.search(r"Seizure\s+End:\s*(\d+)", line, re.I)
            if start_match:
                current_start = int(start_match.group(1))
            elif end_match and current_start is not None:
                intervals.append(
                    SeizureInterval(start_epoch=current_start, end_epoch=int(end_match.group(1)))
                )
                current_start = None
        return intervals
