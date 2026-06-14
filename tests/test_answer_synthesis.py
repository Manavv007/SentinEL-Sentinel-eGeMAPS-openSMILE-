"""Short-answer reliability guard for `engine.answer_synthesis.synthesize_final_decision`."""

from __future__ import annotations

from engine.answer_synthesis import synthesize_final_decision


def test_short_single_window_forces_clear_low():
    status, confidence, reasons = synthesize_final_decision(
        temporal_summary={
            "answer_duration_sec": 2.5,
            "window_count": 1,
        },
        behavioral={},
    )
    assert status == "CLEAR"
    assert confidence == "LOW"
    assert any("insufficient duration" in reason for reason in reasons)


def test_short_multi_window_forces_clear_low():
    status, confidence, reasons = synthesize_final_decision(
        temporal_summary={
            "answer_duration_sec": 4.2,
            "window_count": 3,
        },
        behavioral={},
    )
    assert status == "CLEAR"
    assert confidence == "LOW"
