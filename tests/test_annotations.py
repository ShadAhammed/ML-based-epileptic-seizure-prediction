"""Tests for CHB-MIT summary parsing."""

from __future__ import annotations

from pathlib import Path

from epilepsy_detection.data.annotations import AnnotationParser


def test_numbered_seizure_lines_in_summary_by_file() -> None:
    summary = Path("data/raw/chb08/chb08-summary.txt")
    if not summary.exists():
        return

    by_file = AnnotationParser().load_summary_by_file(summary)
    assert by_file["chb08_05.edf"], "expected seizure intervals for chb08_05.edf"
    assert by_file["chb08_05.edf"][0].start_epoch == 2856
    assert by_file["chb08_05.edf"][0].end_epoch == 3046
