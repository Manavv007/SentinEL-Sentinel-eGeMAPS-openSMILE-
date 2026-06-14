"""Tests for LLM JSON parsing helpers."""

from __future__ import annotations

import pytest

from engine.llm_providers.json_utils import parse_llm_json


def test_parse_llm_json_plain():
    out = parse_llm_json('{"verdict": "CLEAR", "reasons": ["ok"]}')
    assert out["verdict"] == "CLEAR"


def test_parse_llm_json_fenced():
    out = parse_llm_json('```json\n{"verdict": "AMBIGUOUS"}\n```')
    assert out["verdict"] == "AMBIGUOUS"


def test_parse_llm_json_repairs_truncated_string():
    raw = '{\n  "verdict": "CLEAR",\n  "reasons": ["short reason'
    out = parse_llm_json(raw)
    assert out.get("verdict") == "CLEAR"
