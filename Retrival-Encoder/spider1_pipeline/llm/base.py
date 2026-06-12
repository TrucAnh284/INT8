"""Abstract LLM client interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    text: str                       # raw response text
    sql: str = ""                   # extracted SQL (set by postprocess)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    candidates: list[str] = field(default_factory=list)  # for self-consistency


class LLMClient(ABC):
    """Base class for all LLM backends."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
    ) -> LLMResponse:
        """
        Send chat messages to the LLM and return a response.

        messages: OpenAI-style [{"role": "system"|"user"|"assistant", "content": str}]
        n:        number of completions to generate (for self-consistency)
        """
        ...

    def complete_with_retry(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> LLMResponse:
        """Wrapper with exponential-backoff retry on transient errors."""
        import time
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self.complete(messages, temperature, max_tokens, n)
            except Exception as exc:
                last_err = exc
                wait = retry_delay * (2 ** attempt)
                print(f"[LLM] Attempt {attempt + 1} failed: {exc}. Retrying in {wait:.1f}s ...")
                time.sleep(wait)
        raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_err}") from last_err
