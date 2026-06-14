"""Tests for fragmentation guard and merge helper in speaker selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "speaker_selection",
    _ROOT / "processors" / "speaker_selection.py",
)
_speaker_selection = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_speaker_selection)

_merge_short_fragments = _speaker_selection._merge_short_fragments
group_into_answers = _speaker_selection.group_into_answers


def test_short_answer_boundaries_guard_cost_0_100():
    candidates = [(0.0, 2.0), (3.5, 5.6), (6.1, 7.0), (8.5, 16.0)]

    merged = _merge_short_fragments(candidates, 4.0)

    assert merged == [(0.0, 7.0), (8.5, 16.0)]


def test_group_into_answers_merge_fragments():
    segments = [(0.0, 2.2), (3.5, 6.1), (7.0, 7.8)]
    grouped = group_into_answers(
        segments,
        silence_gap_sec=3.0,
        min_answer_duration_sec=3.0,
    )
    assert grouped == [(0.0, 7.8)]
