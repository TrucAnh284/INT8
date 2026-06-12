"""
Ollama native Python client.

Uses the official `ollama` library (pip install ollama) instead of the
OpenAI-compatible shim. Recommended for local models like qwen3.5.

Qwen3 thinking mode:
  Qwen3 models produce <think>...</think> reasoning blocks by default.
  We pass think=False to disable it for cleaner SQL output.

Usage:
  ollama serve          # start server (or run Ollama app)
  ollama pull qwen3.5   # pull the model
"""
from __future__ import annotations

from .base import LLMClient, LLMResponse


class OllamaNativeClient(LLMClient):
    """
    Ollama native client via the `ollama` Python package.

    Parameters
    ----------
    model    : Ollama model tag, e.g. "qwen3.5", "qwen3:8b", "deepseek-coder:33b"
    host     : Ollama server URL (default: http://localhost:11434)
    think    : enable Qwen3 <think> reasoning mode (default: False for SQL tasks)
    """

    def __init__(
        self,
        model: str = "qwen3.5",
        host: str = "http://localhost:11434",
        think: bool = False,
        **_kwargs,           # absorb unused kwargs (api_key, base_url) gracefully
    ):
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError(
                "ollama package not found. Install with: pip install ollama"
            )
        self.model  = model
        self.think  = think
        self._ollama = _ollama
        self._client = _ollama.Client(host=host)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
    ) -> LLMResponse:
        """
        Send messages to Ollama and return an LLMResponse.

        n>1 is achieved by making n sequential calls (Ollama doesn't support
        batch generation natively).
        """
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        def _call_once() -> str:
            kwargs = dict(
                model=self.model,
                messages=messages,
                options=options,
            )
            if self.think is False:
                kwargs["think"] = False
            resp = self._client.chat(**kwargs)
            return resp.message.content or ""

        if n == 1:
            text = _call_once()
            return LLMResponse(
                text=text,
                model=self.model,
                candidates=[text],
            )

        candidates = [_call_once() for _ in range(n)]
        return LLMResponse(
            text=candidates[0],
            model=self.model,
            candidates=candidates,
        )
