"""
Snowflake Arctic Embed M — optimized for asymmetric retrieval (query → document).

Model: Snowflake/snowflake-arctic-embed-m  (110M params, dim=768)

Key difference from MiniLM:
  - Documents (schema texts): encoded WITHOUT prefix
  - Queries  (NL questions):  encoded WITH instruction prefix
  This asymmetric encoding dramatically improves retrieval precision for
  structured metadata like SQL schemas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PIPELINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT))

from sentence_transformers import SentenceTransformer

ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ARCTIC_EMBED_DIM    = 768
ARCTIC_HF_ID        = "Snowflake/snowflake-arctic-embed-m"


class ArcticEmbedModel:
    """
    Thin SentenceTransformer wrapper for arctic-embed-m with correct
    asymmetric query/document encoding.

    Parameters
    ----------
    model_dir : local path to downloaded weights (or HuggingFace model id)
    device    : "mps" | "cpu" | "cuda"
    """

    def __init__(self, model_dir: Path | str, device: str = "mps", quantize_int8: bool = False):
        self.dim = ARCTIC_EMBED_DIM
        if quantize_int8:
            import platform
            engine = "qnnpack" if platform.machine() == "arm64" else "fbgemm"
            torch.backends.quantized.engine = engine
            print(f"[arctic-embed] Loading (INT8/{engine}) from {model_dir} ...")
            model = SentenceTransformer(str(model_dir), device="cpu")
            self.model  = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.device = "cpu"
        else:
            self.device = device
            print(f"[arctic-embed] Loading from {model_dir} on {device} ...")
            self.model = SentenceTransformer(str(model_dir), device=device)
        print(f"[arctic-embed] Ready  (dim={self.dim}, int8={quantize_int8})")

    # ── Document encoding (schema descriptions, no prefix) ────────────────────

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode schema/passage texts — no prefix."""
        with torch.no_grad():
            embs = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                batch_size=batch_size,
            )
        return np.array(embs, dtype=np.float32)

    # ── Query encoding (NL questions, with instruction prefix) ────────────────

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode NL questions — prepends instruction prefix."""
        prefixed = [ARCTIC_QUERY_PREFIX + t for t in texts]
        with torch.no_grad():
            embs = self.model.encode(
                prefixed,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
            )
        return np.array(embs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single NL question."""
        return self.encode_queries([text])[0]
