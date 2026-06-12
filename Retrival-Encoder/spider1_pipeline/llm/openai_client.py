"""
OpenAI-compatible LLM client.

Works with:
  - OpenAI API  (gpt-4o, gpt-4, gpt-3.5-turbo, …)
  - DeepSeek API (deepseek-chat, deepseek-coder)
  - Any OpenAI-compatible endpoint (set base_url accordingly)
"""
from __future__ import annotations

from .base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """
    Thin wrapper around the openai Python SDK.

    Parameters
    ----------
    api_key  : your API key (or set OPENAI_API_KEY env var)
    model    : model name, e.g. "gpt-4o", "gpt-3.5-turbo", "deepseek-chat"
    base_url : override for non-OpenAI endpoints
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ):
        import os
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai>=1.0"
            )
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
        )

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        candidates = [c.message.content or "" for c in resp.choices]
        return LLMResponse(
            text=candidates[0],
            model=resp.model,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            candidates=candidates,
        )
