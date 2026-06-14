"""OpenAI Chat Completions provider (JSON mode)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

import config
from engine.llm_providers.json_utils import parse_llm_json

logger = logging.getLogger(__name__)


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or config.LLM_MODEL
        self.api_key = (api_key or config.OPENAI_API_KEY or "").strip()
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        url = f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": config.LLM_JUDGE_TEMPERATURE,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=config.LLM_JUDGE_TIMEOUT_SEC) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return parse_llm_json(content)
