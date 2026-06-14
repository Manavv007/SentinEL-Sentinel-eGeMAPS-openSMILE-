"""Unit tests for LLM judge tie-breaker (no live API)."""

from __future__ import annotations

import pytest

from engine.llm_judge import (
    apply_llm_judge_to_answers,
    map_llm_verdict_to_status,
    normalize_llm_response,
    sanitize_public_reasons,
)


class MockProvider:
    name = "mock"
    model = "mock-model"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = 0

    def complete_json(self, *, system: str, user: str) -> dict:
        self.calls += 1
        return dict(self.response)


@pytest.mark.parametrize(
    "raw,expected_verdict,expected_likelihood",
    [
        (
            {
                "script_reading_likelihood": 0.9,
                "confidence": "HIGH",
                "verdict": "PROBABLE_SCRIPT_READING",
                "is_interviewer_speech": False,
                "reasons": ["textbook tone"],
            },
            "PROBABLE_SCRIPT_READING",
            0.9,
        ),
        (
            {
                "script_reading_likelihood": "0.15",
                "confidence": "high",
                "verdict": "clear",
                "is_interviewer_speech": False,
                "reasons": "personal story",
            },
            "CLEAR",
            0.15,
        ),
    ],
)
def test_normalize_llm_response(raw, expected_verdict, expected_likelihood):
    out = normalize_llm_response(raw)
    assert out["verdict"] == expected_verdict
    assert out["script_reading_likelihood"] == expected_likelihood


def test_map_interviewer_speech_to_clear():
    status, lines = map_llm_verdict_to_status(
        {
            "script_reading_likelihood": 0.95,
            "confidence": "HIGH",
            "verdict": "PROBABLE_SCRIPT_READING",
            "is_interviewer_speech": True,
            "reasons": ["question form"],
        },
        status_before="AMBIGUOUS",
    )
    assert status == "CLEAR"
    assert lines[0].startswith("LLM judge:")
    assert any("interviewer" in line.lower() for line in lines)


def test_sanitize_public_reasons_filters_first_person_excuse():
    out = sanitize_public_reasons(
        [
            "Uses first-person narrative (I use) indicating personal experience",
            "Reads like a textbook definition without project specifics",
            "No concrete tradeoffs or situational detail mentioned",
        ]
    )
    assert not any("first-person" in r.lower() for r in out)
    assert len(out) >= 2


def test_map_promote_ambiguous_to_probable():
    status, _ = map_llm_verdict_to_status(
        {
            "script_reading_likelihood": 0.85,
            "confidence": "HIGH",
            "verdict": "PROBABLE_SCRIPT_READING",
            "is_interviewer_speech": False,
            "reasons": [],
        },
        status_before="AMBIGUOUS",
        promote_min=0.72,
        min_confidence="MEDIUM",
    )
    assert status == "PROBABLE_SCRIPT_READING"


def test_map_demote_ambiguous_to_clear_requires_high_confidence():
    status, _ = map_llm_verdict_to_status(
        {
            "script_reading_likelihood": 0.1,
            "confidence": "MEDIUM",
            "verdict": "CLEAR",
            "is_interviewer_speech": False,
            "reasons": [],
        },
        status_before="AMBIGUOUS",
        clear_max=0.28,
    )
    assert status == "AMBIGUOUS"

    status_high, _ = map_llm_verdict_to_status(
        {
            "script_reading_likelihood": 0.1,
            "confidence": "HIGH",
            "verdict": "CLEAR",
            "is_interviewer_speech": False,
            "reasons": [],
        },
        status_before="AMBIGUOUS",
        clear_max=0.28,
    )
    assert status_high == "CLEAR"


def test_map_does_not_override_non_ambiguous():
    status, lines = map_llm_verdict_to_status(
        {
            "script_reading_likelihood": 0.1,
            "confidence": "HIGH",
            "verdict": "CLEAR",
            "is_interviewer_speech": False,
            "reasons": [],
        },
        status_before="PROBABLE_SCRIPT_READING",
    )
    assert status == "PROBABLE_SCRIPT_READING"
    assert lines == []


def test_apply_llm_judge_skips_clear_and_probable(monkeypatch):
    monkeypatch.setattr("config.ENABLE_LLM_JUDGE", True, raising=False)
    provider = MockProvider(
        {
            "script_reading_likelihood": 0.9,
            "confidence": "HIGH",
            "verdict": "PROBABLE_SCRIPT_READING",
            "is_interviewer_speech": False,
            "reasons": ["script"],
        }
    )
    answers = [
        {"answer_id": 1, "status": "CLEAR", "transcript": "I built a project at work."},
        {
            "answer_id": 2,
            "status": "PROBABLE_SCRIPT_READING",
            "transcript": "Machine learning is a subset of artificial intelligence.",
        },
    ]
    apply_llm_judge_to_answers(answers, provider=provider)
    assert provider.calls == 0
    assert "llm_judge" not in answers[0]
    assert "llm_judge" not in answers[1]


def test_apply_llm_judge_promotes_ambiguous(monkeypatch):
    monkeypatch.setattr("config.ENABLE_LLM_JUDGE", True, raising=False)
    provider = MockProvider(
        {
            "script_reading_likelihood": 0.88,
            "confidence": "HIGH",
            "verdict": "PROBABLE_SCRIPT_READING",
            "is_interviewer_speech": False,
            "reasons": ["polished textbook prose"],
        }
    )
    answers = [
        {
            "answer_id": 3,
            "status": "AMBIGUOUS",
            "transcript": (
                "Gradient descent is an optimization algorithm used to minimize "
                "the loss function by iteratively updating parameters."
            ),
            "signals": {"acoustic": 0.4, "linguistic": 0.5},
            "contrastive": {"decision_explanation": []},
        }
    ]
    apply_llm_judge_to_answers(answers, provider=provider)
    assert provider.calls == 1
    assert answers[0]["status"] == "PROBABLE_SCRIPT_READING"
    assert answers[0]["llm_judge"]["ran"] is True
    assert answers[0]["llm_judge"]["verdict"] == "PROBABLE_SCRIPT_READING"
    assert len(answers[0]["llm_judge"]["reasons"]) <= 3


def test_apply_llm_judge_interviewer_flag_clears(monkeypatch):
    monkeypatch.setattr("config.ENABLE_LLM_JUDGE", True, raising=False)
    provider = MockProvider(
        {
            "script_reading_likelihood": 0.8,
            "confidence": "HIGH",
            "verdict": "PROBABLE_SCRIPT_READING",
            "is_interviewer_speech": True,
            "reasons": ["Could you describe your experience"],
        }
    )
    answers = [
        {
            "answer_id": 4,
            "status": "AMBIGUOUS",
            "transcript": "Could you describe your experience with distributed systems?",
            "contrastive": {},
        }
    ]
    apply_llm_judge_to_answers(answers, provider=provider)
    assert answers[0]["status"] == "CLEAR"
    assert answers[0]["llm_judge"]["verdict"] == "CLEAR"
