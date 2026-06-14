"""LLM provider protocol for structured JSON completion."""

from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    """Pluggable backend for the LLM judge."""

    name: str
    model: str

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        """Return parsed JSON object from the model."""
        ...
