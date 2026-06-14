"""Instantiate configured LLM provider."""

from __future__ import annotations

import config
from engine.llm_providers.anthropic_provider import AnthropicProvider
from engine.llm_providers.base import LLMProvider
from engine.llm_providers.ollama_provider import OllamaProvider
from engine.llm_providers.openai_provider import OpenAIProvider
from engine.llm_providers.openrouter_provider import OpenRouterProvider

_PROVIDERS = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def get_llm_provider() -> LLMProvider:
    name = (config.LLM_PROVIDER or "openai").lower().strip()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER={name!r}. Use one of: {', '.join(_PROVIDERS)}"
        )
    return cls()
