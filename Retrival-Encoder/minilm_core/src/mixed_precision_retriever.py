"""
Mixed-Precision Schema Retriever — Phase 4 Core Contribution.

Applies a configurable mixed-precision quantization strategy to all-MiniLM-L6-v2:
  - "sensitive" layers stay at FP16  (embeddings, pooler, optionally attention Q/K)
  - "bulk" layers are quantized to INT8_dynamic (FFN, V, attn_out)

Pre-defined profiles (from sensitivity_analysis results):
  MP-Conservative : FP16 for embed+pooler+Q+K, INT8 for V+attn_out+FFN
  MP-Balanced     : FP16 for embed+pooler, INT8 for all transformer linears
  MP-Aggressive   : FP16 for embed only, INT8 for pooler+all transformer linears
  INT8-Full       : FP16 embed, INT8 for everything else (no FP16 transformer)
  FP16-Full       : All FP16 (baseline comparison)

Usage:
    from mixed_precision_retriever import MixedPrecisionRetriever, MP_PROFILES

    retriever = MixedPrecisionRetriever(profile="MP-Balanced")
    results = retriever.retrieve("How many singers?", db_id="concert_singer", k=5)
"""

import copy
import sys
import time
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
try:
    import sqlite_vec as sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except Exception:
    sqlite_vec = None          # type: ignore[assignment]
    _SQLITE_VEC_AVAILABLE = False
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

DEFAULT_MINILM_DIR = REPO_ROOT / "models" / "minilm"
DEFAULT_INDEX_DIR  = REPO_ROOT / "dataset" / "spider" / "vec_index"

EMBEDDING_DIM = 384
N_LAYERS = 6


# ── Mixed-precision profile definitions ───────────────────────────────────────

@dataclass
class MPProfile:
    """
    Defines which parts of the model stay at FP16 vs get INT8 quantization.

    fp16_components: set of component name patterns to KEEP at FP16.
    Everything else (all nn.Linear layers) gets INT8_dynamic.

    Component patterns (matched via 'in name'):
      "embeddings"         -> BertEmbeddings (word + pos + type embed)
      "pooler"             -> BertPooler.dense
      "attention.self.query" -> Q projection in all layers
      "attention.self.key"   -> K projection in all layers
      "attention.self.value" -> V projection in all layers
      "attention.output"     -> attention output dense
      "intermediate"         -> FFN intermediate dense (expand)
      "output.dense"         -> FFN output dense (contract) + attn output
    """
    name: str
    description: str
    fp16_components: set[str] = field(default_factory=set)


MP_PROFILES: dict[str, MPProfile] = {
    "FP16-Full": MPProfile(
        name="FP16-Full",
        description="All parameters at FP16 — baseline comparison",
        fp16_components={"embeddings", "encoder", "pooler"},  # everything
    ),
    "MP-Conservative": MPProfile(
        name="MP-Conservative",
        description="FP16: embed+pooler+Q+K | INT8: V+attn_out+FFN",
        fp16_components={"embeddings", "pooler", "attention.self.query", "attention.self.key"},
    ),
    "MP-Balanced": MPProfile(
        name="MP-Balanced",
        description="FP16: embed+pooler | INT8: all transformer linear layers",
        fp16_components={"embeddings", "pooler"},
    ),
    "MP-Aggressive": MPProfile(
        name="MP-Aggressive",
        description="FP16: embed only | INT8: pooler + all transformer linears",
        fp16_components={"embeddings"},
    ),
    "INT8-Full": MPProfile(
        name="INT8-Full",
        description="INT8 for all linear layers including pooler (embed stays FP32)",
        fp16_components=set(),
    ),
}


# ── Mixed-precision model builder ─────────────────────────────────────────────

def _set_quant_engine():
    available = torch.backends.quantized.supported_engines
    if "qnnpack" in available:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in available:
        torch.backends.quantized.engine = "fbgemm"


def _should_keep_fp16(name: str, fp16_components: set[str]) -> bool:
    """Return True if this module name should stay at FP16."""
    return any(pattern in name for pattern in fp16_components)


def build_mixed_precision_model(
    base_model: SentenceTransformer,
    profile: MPProfile,
) -> SentenceTransformer:
    """
    Apply mixed-precision quantization to a copy of the base model.

    Design:
      - All activations stay at FP32 to avoid dtype mismatch.
      - "Protected" layers (in fp16_components): kept at FP32 (not quantized).
      - "Bulk" layers (not in fp16_components): INT8_dynamic quantization
        (weights stored as INT8, computation at FP32 via qnnpack).

    Naming convention: "FP16" in profile names means "high precision / not quantized".
    On CPU, this translates to: protected=FP32 weights, bulk=INT8 weights.

    Memory savings come from INT8 weight storage (4x vs FP32).
    """
    _set_quant_engine()
    m = copy.deepcopy(base_model)

    # FP16-Full: keep everything unquantized (FP32)
    if "encoder" in profile.fp16_components and "pooler" in profile.fp16_components:
        return m

    # INT8-Full: quantize every Linear layer
    if not profile.fp16_components:
        hf_model = m[0].auto_model
        m[0].auto_model = torch.quantization.quantize_dynamic(
            hf_model, {nn.Linear}, dtype=torch.qint8
        )
        return m

    # Mixed: selectively quantize only non-protected Linear layers
    # Approach: quantize layer-by-layer using the encoder structure
    hf_model = m[0].auto_model

    def _quantize_if_unprotected(module: nn.Module, module_name: str):
        """Quantize a submodule in-place if it's not in the protected set."""
        if _should_keep_fp16(module_name, profile.fp16_components):
            return  # keep at FP32

        # Check if this submodule contains any Linear layers
        has_linear = any(isinstance(c, nn.Linear) for c in module.modules())
        if not has_linear:
            return

        # Replace the submodule with its dynamically-quantized version
        quantized = torch.quantization.quantize_dynamic(
            copy.deepcopy(module), {nn.Linear}, dtype=torch.qint8
        )
        # Navigate to parent and replace
        parts = module_name.split(".")
        parent = hf_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], quantized)

    # Process each transformer layer's sub-components individually
    for layer_idx in range(N_LAYERS):
        prefix = f"encoder.layer.{layer_idx}"
        sublayer_paths = {
            f"{prefix}.attention.self":    f"{prefix}.attention.self",
            f"{prefix}.attention.output":  f"{prefix}.attention.output",
            f"{prefix}.intermediate":      f"{prefix}.intermediate",
            f"{prefix}.output":            f"{prefix}.output",
        }
        for path, full_name in sublayer_paths.items():
            try:
                submod = hf_model
                for part in path.split("."):
                    submod = getattr(submod, part)
                _quantize_if_unprotected(submod, full_name)
            except AttributeError:
                pass

    # Handle pooler separately
    if not _should_keep_fp16("pooler", profile.fp16_components):
        try:
            pooler_q = torch.quantization.quantize_dynamic(
                copy.deepcopy(hf_model.pooler), {nn.Linear}, dtype=torch.qint8
            )
            hf_model.pooler = pooler_q
        except AttributeError:
            pass

    m[0].auto_model = hf_model
    return m


def estimate_effective_size_mb(model: SentenceTransformer) -> float:
    """Estimate effective memory footprint in MB, accounting for quantized layers."""
    total = 0
    for name, param in model.named_parameters():
        if hasattr(param, "element_size"):
            total += param.nelement() * param.element_size()
        else:
            total += param.nelement() * 4
    # Also account for quantized buffers
    for name, buf in model.named_buffers():
        if buf is not None:
            total += buf.nelement() * buf.element_size()
    return total / (1024 ** 2)


# ── Mixed-Precision Retriever class ───────────────────────────────────────────

class MixedPrecisionRetriever:
    """
    Schema retriever using mixed-precision all-MiniLM-L6-v2.

    Supports both:
      1. sqlite-vec index lookup (production mode)
      2. In-memory cosine similarity (experiment mode)
    """

    def __init__(
        self,
        profile: str | MPProfile = "MP-Balanced",
        minilm_dir: Path = DEFAULT_MINILM_DIR,
        index_dir: Path = DEFAULT_INDEX_DIR,
        device: str = "cpu",
    ):
        if isinstance(profile, str):
            if profile not in MP_PROFILES:
                raise ValueError(f"Unknown profile '{profile}'. Choose from: {list(MP_PROFILES)}")
            self.profile = MP_PROFILES[profile]
        else:
            self.profile = profile

        self.device = device
        self.index_dir = Path(index_dir)
        self._conn_cache: dict[str, sqlite3.Connection] = {}

        print(f"[INFO] Building {self.profile.name} retriever on {device} ...")
        print(f"       {self.profile.description}")

        base = SentenceTransformer(str(minilm_dir), device=device)
        if device != "cpu":
            # INT8 dynamic quantization only works on CPU; use unquantized model on MPS/CUDA
            self.model = base
            print(f"       Quantization skipped (device={device}); using FP32")
        else:
            self.model = build_mixed_precision_model(base, self.profile)
        self.model_size_mb = estimate_effective_size_mb(self.model)
        print(f"       Effective size: {self.model_size_mb:.1f} MB")

        self._encode_times: list[float] = []
        self._search_times: list[float] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to normalized 384-dim float32 embeddings."""
        t0 = time.perf_counter()
        with torch.no_grad():
            embs = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        self._encode_times.append((time.perf_counter() - t0) * 1000)
        return embs.astype(np.float32)

    def _get_conn(self, db_id: str) -> sqlite3.Connection:
        if db_id not in self._conn_cache:
            db_path = self.index_dir / f"{db_id}.db"
            if not db_path.exists():
                raise FileNotFoundError(
                    f"Index not found for '{db_id}'. Run: python src/schema_indexer.py"
                )
            conn = sqlite3.connect(str(db_path))
            if not _SQLITE_VEC_AVAILABLE:
                raise RuntimeError(
                    "sqlite_vec is not available on this platform. "
                    "Use Spider1Retriever (spider1_pipeline) which uses plain SQLite BLOB indexes."
                )
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._conn_cache[db_id] = conn
        return self._conn_cache[db_id]

    def retrieve(
        self,
        query: str,
        db_id: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Retrieve Top-K tables using sqlite-vec index.
        Returns list of {table_name, score, code_repr} dicts.
        """
        q_emb = self.encode([query])[0]

        t0 = time.perf_counter()
        conn = self._get_conn(db_id)
        rows = conn.execute(
            """
            SELECT m.table_name, v.distance, m.code_repr
            FROM schema_vectors v
            JOIN schema_metadata m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            [q_emb.tobytes(), k],
        ).fetchall()
        self._search_times.append((time.perf_counter() - t0) * 1000)

        return [
            {"table_name": r[0], "score": max(0.0, 1.0 - r[1] / 2.0), "code_repr": r[2]}
            for r in rows
        ]

    def retrieve_inmemory(
        self,
        query: str,
        schema_embs: dict[str, np.ndarray],
        k: int = 5,
    ) -> list[str]:
        """
        In-memory retrieval for experiments (no sqlite-vec needed).
        schema_embs: {table_name: normalized (384,) array}
        """
        q_emb = self.encode([query])[0]
        table_names = list(schema_embs.keys())
        emb_matrix  = np.stack([schema_embs[t] for t in table_names])
        scores = emb_matrix @ q_emb
        top_k = np.argsort(scores)[::-1][:k]
        return [table_names[i] for i in top_k]

    def latency_stats(self) -> dict:
        def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0
        return {
            "encode_avg_ms": avg(self._encode_times),
            "search_avg_ms": avg(self._search_times),
            "n_queries": len(self._encode_times),
        }

    def close(self):
        for conn in self._conn_cache.values():
            conn.close()
        self._conn_cache.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


if __name__ == "__main__":
    print("Available profiles:")
    for name, p in MP_PROFILES.items():
        print(f"  {name:<20} {p.description}")

    print("\nBuilding MP-Balanced retriever ...")
    r = MixedPrecisionRetriever("MP-Balanced")
    emb = r.encode(["How many singers are there?"])
    print(f"  Encode test: shape={emb.shape}, dtype={emb.dtype}")
    print(f"  Model size: {r.model_size_mb:.1f} MB")
    r.close()
