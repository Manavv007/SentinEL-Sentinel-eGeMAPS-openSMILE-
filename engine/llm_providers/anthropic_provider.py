"""Anthropic Messages API provider."""

from __future__ import annotations

import re
from typing import Any

import httpx

import config
from engine.llm_providers.json_utils import parse_llm_json


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or config.LLM_MODEL
        self.api_key = (api_key or config.ANTHROPIC_API_KEY or "").strip()
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        url = f"{config.ANTHROPIC_BASE_URL.rstrip('/')}/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": config.LLM_JUDGE_MAX_TOKENS,
            "temperature": config.LLM_JUDGE_TEMPERATURE,
            "system": system + "\n\nRespond with valid JSON only, no markdown.",
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=config.LLM_JUDGE_TIMEOUT_SEC) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return parse_llm_json(text)
