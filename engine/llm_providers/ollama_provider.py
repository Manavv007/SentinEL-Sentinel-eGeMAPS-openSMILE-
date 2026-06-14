"""Local Ollama chat provider."""

from __future__ import annotations

from typing import Any

import httpx

import config
from engine.llm_providers.json_utils import parse_llm_json


class OllamaProvider:
    name = "ollama"

    def __init__(self, *, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or config.LLM_MODEL
        self.base_url = (base_url or config.OLLAMA_BASE_URL or "").rstrip("/")
        if not self.base_url:
            raise ValueError("OLLAMA_BASE_URL is required for LLM_PROVIDER=ollama")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": config.LLM_JUDGE_TEMPERATURE},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        with httpx.Client(timeout=config.LLM_JUDGE_TIMEOUT_SEC) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "{}")
        return parse_llm_json(content)
