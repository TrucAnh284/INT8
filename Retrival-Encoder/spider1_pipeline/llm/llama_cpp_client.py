"""
llama-cpp-python LLM client with Metal GPU offload (macOS).

Uses GGUF quantized models downloaded from HuggingFace.
Recommended model: byteshape/Qwen3.5-9B-GGUF  (IQ3_S, ~3.2 GB)

Metal offload: all transformer layers are sent to the GPU via
n_gpu_layers=-1, giving 2-4× speedup over Ollama on Apple Silicon.

Thinking mode is disabled by appending /no_think to the system
prompt so Qwen3.5 never emits <think> blocks.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Optional

from llm.base import LLMClient, LLMResponse

_LOAD_LOCK = threading.Lock()   # one load at a time — llama.cpp not thread-safe during init

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LlamaCppClient(LLMClient):
    """
    LLM client backed by llama-cpp-python with Metal GPU offload.

    Parameters
    ----------
    repo_id     : HuggingFace repo, e.g. "byteshape/Qwen3.5-9B-GGUF"
    filename    : GGUF filename,    e.g. "Qwen3.5-9B-IQ3_S-2.81bpw.gguf"
    local_path  : if given, load from disk instead of HF (skip download)
    n_gpu_layers: -1 = all layers on Metal GPU (recommended)
    n_ctx       : context window (tokens)
    verbose     : show llama.cpp progress/stats
    """

    def __init__(
        self,
        repo_id: str = "byteshape/Qwen3.5-9B-GGUF",
        filename: str = "Qwen3.5-9B-IQ3_S-2.81bpw.gguf",
        local_path: Optional[Path] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        verbose: bool = False,
    ):
        self.repo_id      = repo_id
        self.filename     = filename
        self.local_path   = local_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx        = n_ctx
        self.verbose      = verbose
        self._llm         = None   # lazy load

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self):
        if self._llm is not None:
            return
        with _LOAD_LOCK:
            if self._llm is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "llama-cpp-python not installed. Install with Metal support:\n"
                    "  CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python"
                ) from e

            if self.local_path and Path(self.local_path).exists():
                print(f"[llama-cpp] Loading {self.local_path} "
                      f"(n_gpu_layers={self.n_gpu_layers}) ...")
                self._llm = Llama(
                    model_path=str(self.local_path),
                    n_gpu_layers=self.n_gpu_layers,
                    n_ctx=self.n_ctx,
                    verbose=self.verbose,
                )
            else:
                print(f"[llama-cpp] Downloading {self.repo_id}/{self.filename} "
                      f"(n_gpu_layers={self.n_gpu_layers}) ...")
                self._llm = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    n_gpu_layers=self.n_gpu_layers,
                    n_ctx=self.n_ctx,
                    verbose=self.verbose,
                )
            print("[llama-cpp] Model ready.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
    ) -> LLMResponse:
        self._load()

        # Disable Qwen3.5 thinking mode by appending /no_think to system prompt
        patched = []
        for m in messages:
            if m["role"] == "system":
                content = m["content"]
                if "/no_think" not in content:
                    content = content + " /no_think"
                patched.append({"role": "system", "content": content})
            else:
                patched.append(m)

        candidates = []
        for _ in range(n):
            resp = self._llm.create_chat_completion(
                messages=patched,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp["choices"][0]["message"]["content"] or ""
            # Strip any residual <think>…</think> blocks
            raw = _THINK_RE.sub("", raw).strip()
            candidates.append(raw)

        text = candidates[0] if candidates else ""
        return LLMResponse(text=text, candidates=candidates)

    def close(self):
        self._llm = None
