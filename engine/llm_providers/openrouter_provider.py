"""OpenRouter provider (OpenAI-compatible chat completions API)."""

from __future__ import annotations

from typing import Any

import httpx

import config
from engine.llm_providers.json_utils import parse_llm_json


class OpenRouterProvider:
    name = "openrouter"

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or config.LLM_MODEL
        self.api_key = (api_key or config.OPENROUTER_API_KEY or "").strip()
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for LLM_PROVIDER=openrouter")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": config.LLM_JUDGE_TEMPERATURE,
            "max_tokens": config.LLM_JUDGE_MAX_TOKENS,
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
        referer = (config.OPENROUTER_HTTP_REFERER or "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        app_name = (config.OPENROUTER_APP_NAME or "").strip()
        if app_name:
            headers["X-Title"] = app_name

        with httpx.Client(timeout=config.LLM_JUDGE_TIMEOUT_SEC) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return parse_llm_json(content)
