"""
Ollama local LLM client.

Connects to a running Ollama server (default: http://localhost:11434).

Recommended models for Text-to-SQL:
  - deepseek-coder:33b      (~84% EX on Spider, best local open-source)
  - qwen2.5-coder:32b       (~83% EX, very competitive)
  - codestral:22b           (~81% EX)
  - llama3.3:70b            (~80% EX, general-purpose)
  - deepseek-coder:6.7b     (~76% EX, fast on CPU)

Install Ollama: https://ollama.ai
Pull a model:   ollama pull deepseek-coder:33b
"""
from __future__ import annotations

from .base import LLMClient, LLMResponse


class OllamaClient(LLMClient):
    """
    Ollama-backed LLM using its OpenAI-compatible /v1 endpoint.

    Parameters
    ----------
    model    : Ollama model tag, e.g. "deepseek-coder:33b"
    base_url : Ollama server URL (default: http://localhost:11434/v1)
    """

    def __init__(
        self,
        model: str = "deepseek-coder:33b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",     # Ollama ignores API key but SDK requires it
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai>=1.0"
            )
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

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
            model=self.model,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
            candidates=candidates,
        )
