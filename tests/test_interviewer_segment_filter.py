"""Tests for interviewer segment detection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from engine.interviewer_segment_filter import (
    assess_kept_segment_quality,
    filter_segments_by_transcript,
    infer_alternate_speaker,
    is_interviewer_transcript,
    is_off_topic_or_wrong_slice,
    is_too_short_answer,
    needs_speaker_track_recovery,
)

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "speaker_selection",
    _ROOT / "processors" / "speaker_selection.py",
)
_speaker_selection = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_speaker_selection)


def test_interviewer_intro_detected():
    text = (
        "We'll be going through a series of questions focused on backend development. "
        "To start, could you briefly introduce yourself?"
    )
    assert is_interviewer_transcript(text)


def test_saren_ai_bot_intro_detected():
    text = (
        "Hello and welcome. I'm Saren, and I'll be conducting your interview today. "
        "We'll be going through a series of questions focused on HDSC backend developer. "
        "To start, could you briefly introduce yourself and share what specifically "
        "attracted you to this role?"
    )
    assert is_interviewer_transcript(text)


def test_truncated_ai_question_detected():
    assert is_interviewer_transcript(
        "Considering scalability, how would you design a back-end service to"
    )


def test_off_topic_volleyball_detected():
    assert is_off_topic_or_wrong_slice(
        "Please translate the page about particle gas and fuels are mixtures."
    )


def test_candidate_irrelevant_or_reused_content_is_kept_for_scoring():
    texts = [
        (
            "Beyond development, I am involved in research and what excites me about "
            "a forward deployment engineer role is the opportunity to work closely with users. "
            "I also worked on a hybrid AI navigation system for a planetary exploration rover."
        ),
        (
            "I am planning to pursue a master's degree in design because I'm passionate "
            "about combining technology with creativity."
        ),
        "Let's talk about something else. Please ask some other question about cricket.",
    ]
    for text in texts:
        assert not is_off_topic_or_wrong_slice(text)
        assert not is_interviewer_transcript(text)


def test_actual_failure_shape_keeps_recovered_candidate_track():
    answers = [
        {"answer_id": 0, "start_sec": 26.0, "end_sec": 94.0},
        {"answer_id": 1, "start_sec": 102.0, "end_sec": 160.0},
        {"answer_id": 2, "start_sec": 169.0, "end_sec": 264.0},
        {"answer_id": 3, "start_sec": 277.0, "end_sec": 350.0},
        {"answer_id": 4, "start_sec": 357.0, "end_sec": 398.0},
        {"answer_id": 5, "start_sec": 406.0, "end_sec": 455.0},
    ]
    transcripts = [
        {"transcript": "Yeah, sure. I'm Deo Kansara, and I am currently pursuing my BTech in Computer Science Engineering."},
        {"transcript": "Yeah. So primarily for sensitive financial data at rest, I will be using encryption techniques."},
        {"transcript": "Beyond development, I am involved in research. What excites me about a forward deployment engineer role is opportunity to work closely with users."},
        {"transcript": "So for database transaction isolation, we would be using ACID properties called atomicity and consistency."},
        {"transcript": "Yeah. So looking at it, I am planning to pursue a master's degree in design because I'm passionate about technology and creativity."},
        {"transcript": "Let's talk about something else. Please ask some other question about cricket."},
    ]
    kept, excluded, mask = filter_segments_by_transcript(answers, transcripts)
    assert len(kept) == len(answers)
    assert excluded == []
    assert mask == [True] * len(answers)


def test_assess_quality_fails_when_interviewer_in_kept():
    ok, meta = assess_kept_segment_quality(
        [{"transcript": "Hello and welcome. I'm Saren conducting your interview today."}]
    )
    assert not ok
    assert meta.get("reason") == "interviewer_in_kept"


def test_interviewer_question_detected():
    assert is_interviewer_transcript(
        "Understood. How would you ensure encryption of sensitive financial data both at rest and in transit?"
    )


def test_candidate_answer_not_interviewer():
    text = (
        "At my last role I implemented AES-256 for data at rest using AWS KMS "
        "and TLS 1.3 for transit after we had a key rotation incident in Q2."
    )
    assert not is_interviewer_transcript(text)


def test_needs_speaker_track_recovery():
    assert needs_speaker_track_recovery(12, 14, 2)
    assert not needs_speaker_track_recovery(1, 10, 5)


def test_filter_segments_by_transcript():
    answers = [
        {"answer_id": 0, "start_sec": 1.0, "end_sec": 10.0},
        {"answer_id": 1, "start_sec": 20.0, "end_sec": 40.0},
    ]
    transcripts = [
        {"transcript": "How would you design a rate limiter for banking APIs?"},
        {
            "transcript": (
                "We used a token bucket per customer ID in Redis with a 429 fallback "
                "when traffic spiked during payroll week."
            )
        },
    ]
    kept, excluded, mask = filter_segments_by_transcript(answers, transcripts)
    assert len(excluded) == 1
    assert len(kept) == 1
    assert kept[0]["answer_id"] == 0
    assert mask == [False, True]


def test_refine_candidate_speaker_swaps_opener():
    segments = [
        (0.0, 20.0, "SPEAKER_A"),
        (21.0, 45.0, "SPEAKER_B"),
        (46.0, 52.0, "SPEAKER_A"),
        (53.0, 80.0, "SPEAKER_B"),
    ]
    totals = {"SPEAKER_A": 32.0, "SPEAKER_B": 51.0}
    refined, swapped = _speaker_selection.refine_candidate_speaker(
        segments, "SPEAKER_A", totals
    )
    assert swapped
    assert refined == "SPEAKER_B"


def test_too_short_answer_filtered():
    assert is_too_short_answer("Sure.")
    answers = [{"answer_id": 0, "start_sec": 1.0, "end_sec": 5.0}]
    transcripts = [{"transcript": "Sure."}]
    kept, excluded, mask = filter_segments_by_transcript(answers, transcripts)
    assert len(kept) == 0
    assert excluded[0]["reason"] == "too_short"
    assert mask == [False]


def test_infer_alternate_speaker():
    meta = {
        "chosen_speaker": "SPEAKER_01",
        "speaker_total_sec": {"SPEAKER_00": 400.0, "SPEAKER_01": 170.0},
    }
    assert infer_alternate_speaker(meta) == "SPEAKER_00"


def test_build_candidate_turns_keeps_responses_only():
    segments = [
        (0.0, 18.0, "AI"),
        (19.0, 45.0, "HUMAN"),
        (46.0, 52.0, "AI"),
        (53.0, 75.0, "HUMAN"),
    ]
    turns, meta = _speaker_selection.build_candidate_turns(segments, strategy="responder")
    assert meta.get("chosen_speaker") == "HUMAN"
    assert len(turns) >= 1
    assert all(end - start >= 4.0 for start, end in turns)
