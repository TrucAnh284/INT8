"""
Spider 1.0 Schema Retriever.

Encodes NL questions with snowflake-arctic-embed-m (query-instruction prefix),
loads per-database BLOB embeddings from plain SQLite, and retrieves top-K tables
via numpy cosine similarity — no sqlite-vec / enable_load_extension needed.
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_PIPELINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT))

from embed.arctic_embed import ArcticEmbedModel


@dataclass
class TableMatch:
    db_id: str
    table_name: str
    score: float          # cosine similarity (dot product of normalized vectors)
    retrieval_text: str


class Spider1Retriever:
    """
    Top-K table retriever for Spider 1.0.

    Uses asymmetric arctic-embed-m: queries encoded with instruction prefix,
    documents (schema descriptions) encoded without prefix at index time.
    Retrieval is in-memory numpy cosine similarity over per-DB BLOB indexes.
    """

    def __init__(
        self,
        model_dir: Path,
        index_dir: Path,
        device: str = "mps",
        quantize_int8: bool = False,
        # legacy compat params — ignored
        profile: str = "MP-Balanced",
        minilm_dir: Optional[Path] = None,
    ):
        self._embed = ArcticEmbedModel(model_dir=model_dir, device=device, quantize_int8=quantize_int8)
        self.index_dir = Path(index_dir)
        # Cache: db_id → (table_names, retrieval_texts, embeddings_matrix)
        self._cache: dict[str, tuple[list[str], list[str], np.ndarray]] = {}

    def _load_db(self, db_id: str) -> tuple[list[str], list[str], np.ndarray]:
        """Load embeddings from the SQLite BLOB index into memory (cached)."""
        if db_id not in self._cache:
            db_path = self.index_dir / f"{db_id}.db"
            if not db_path.exists():
                raise FileNotFoundError(
                    f"No index found for '{db_id}'. "
                    f"Run: python3 run.py index  (or python3 run.py index --db_id {db_id})"
                )
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT db_id, table_name, retrieval_text, embedding "
                "FROM schema_index ORDER BY rowid"
            ).fetchall()
            conn.close()

            table_names = [r[1] for r in rows]
            ret_texts   = [r[2] for r in rows]
            emb_matrix  = np.stack(
                [np.frombuffer(r[3], dtype=np.float32) for r in rows]
            )  # shape: (N, 384)
            self._cache[db_id] = (table_names, ret_texts, emb_matrix)
        return self._cache[db_id]

    def retrieve(
        self,
        question: str,
        db_id: str,
        k: int = 10,
    ) -> list[TableMatch]:
        """
        Retrieve the Top-K most relevant tables for a NL question.

        Returns TableMatch list sorted by cosine-similarity score descending.
        """
        q_emb = self._embed.encode_query(question)      # (768,) normalized
        table_names, ret_texts, emb_matrix = self._load_db(db_id)

        scores = emb_matrix @ q_emb   # cosine sim (vectors already normalized)
        k_eff  = min(k, len(table_names))
        top_idx = np.argsort(scores)[::-1][:k_eff]

        return [
            TableMatch(
                db_id=db_id,
                table_name=table_names[i],
                score=float(scores[i]),
                retrieval_text=ret_texts[i],
            )
            for i in top_idx
        ]

    def retrieve_table_names(self, question: str, db_id: str, k: int = 10) -> list[str]:
        """Convenience: return just the top-K table name strings."""
        return [m.table_name for m in self.retrieve(question, db_id, k)]

    def retrieve_all_tables(self, db_id: str) -> list[str]:
        """Return all indexed table names for a given database."""
        table_names, _, _ = self._load_db(db_id)
        return table_names

    def close(self):
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
