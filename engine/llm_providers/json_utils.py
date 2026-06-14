"""Robust JSON parsing for LLM judge responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(content: str) -> dict[str, Any]:
    """Parse model output that should be JSON; tolerate markdown fences and minor damage."""
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty LLM response", text, 0)

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text) or re.search(r"\{[\s\S]*", text)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            # Truncated JSON — close open string and braces heuristically
            repaired = candidate.rstrip()
            if repaired.count('"') % 2 == 1:
                repaired += '"'
            open_brackets = repaired.count("[") - repaired.count("]")
            if open_brackets > 0:
                repaired += "]" * open_brackets
            open_braces = repaired.count("{") - repaired.count("}")
            if open_braces > 0:
                repaired += "}" * open_braces
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed

    raise json.JSONDecodeError("Could not parse LLM JSON", text, 0)
