from .base import LLMClient, LLMResponse
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from .ollama_native_client import OllamaNativeClient
from .llama_cpp_client import LlamaCppClient


def get_client(backend: str, **kwargs) -> LLMClient:
    """Factory: return the appropriate LLM client by backend name."""
    if backend == "openai":
        return OpenAIClient(**kwargs)
    elif backend == "ollama":
        return OllamaClient(**kwargs)
    elif backend == "ollama_native":
        return OllamaNativeClient(**kwargs)
    elif backend == "llama_cpp":
        # Strip unknown kwargs (api_key, base_url) that don't apply to llama.cpp
        cpp_keys = {"repo_id", "filename", "local_path", "n_gpu_layers", "n_ctx", "verbose"}
        cpp_kwargs = {k: v for k, v in kwargs.items() if k in cpp_keys}
        return LlamaCppClient(**cpp_kwargs)
    else:
        raise ValueError(
            f"Unknown LLM backend '{backend}'. "
            f"Choose: openai | ollama | ollama_native | llama_cpp"
        )


__all__ = ["LLMClient", "LLMResponse", "OpenAIClient", "OllamaClient",
           "OllamaNativeClient", "LlamaCppClient", "get_client"]
