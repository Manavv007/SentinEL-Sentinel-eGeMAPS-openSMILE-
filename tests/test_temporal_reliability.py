"""`short_answer_blocks_ambiguous` edge-case coverage."""

from __future__ import annotations

import config
from engine.temporal_reliability import short_answer_blocks_ambiguous


def test_borderline_moderate_count_returns_true(monkeypatch):
    monkeypatch.setattr(config, "TEMPORAL_SHORT_ANSWER_SEC", 10.0)
    monkeypatch.setattr(config, "TEMPORAL_SHORT_MIN_WINDOWS", 3)
    monkeypatch.setattr(config, "SHORT_ANSWER_AMBIGUOUS_MIN_MODERATE_WINDOWS", 4)
    monkeypatch.setattr(config, "SHORT_ANSWER_AMBIGUOUS_COMPOSITE_RATIO", 0.95)
    monkeypatch.setattr(config, "SHORT_ANSWER_AMBIGUOUS_MIN_PEAK", 0.45)

    short_params = dict(
        duration_sec=6.0,
        window_count=3,
        strong_count=0,
        moderate_count=2,
        composite=0.3,
        peak_suspicion=0.2,
        margin=1.0,
    )
    assert short_answer_blocks_ambiguous(**short_params) is True
