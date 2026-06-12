"""
Schema Retriever — Top-K table retrieval via cosine similarity on sqlite-vec.

Given a natural language query, returns the Top-K most relevant tables
from the target database, ranked by embedding cosine similarity.

Usage:
    from schema_retriever import SchemaRetriever
    retriever = SchemaRetriever()
    results = retriever.retrieve("How many singers are there?", db_id="concert_singer", k=5)
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MINILM_DIR = REPO_ROOT / "models" / "minilm"
DEFAULT_INDEX_DIR  = REPO_ROOT / "dataset" / "spider" / "vec_index"

EMBEDDING_DIM = 384


@dataclass
class RetrievalResult:
    db_id: str
    table_name: str
    distance: float       # L2 distance (lower = more similar)
    score: float          # 1 - distance/2 (approx cosine similarity)
    code_repr: str
    retrieval_text: str


class SchemaRetriever:
    """
    FP16/FP32 baseline retriever.
    Wraps all-MiniLM-L6-v2 + sqlite-vec for schema linking.
    """

    def __init__(
        self,
        minilm_dir: Path = DEFAULT_MINILM_DIR,
        index_dir: Path = DEFAULT_INDEX_DIR,
        device: str = "cpu",
        precision: str = "fp32",  # "fp32" | "fp16"
    ):
        self.index_dir = Path(index_dir)
        self.precision = precision
        self._conn_cache: dict[str, sqlite3.Connection] = {}

        import torch
        dtype_map = {"fp32": None, "fp16": "float16"}
        torch_dtype = dtype_map.get(precision)

        print(f"[INFO] Loading all-MiniLM-L6-v2 ({precision}) from {minilm_dir} ...")
        self.model = SentenceTransformer(str(minilm_dir), device=device)

        if precision == "fp16":
            import torch
            self.model = self.model.half()

        self._encode_time_ms: list[float] = []
        self._search_time_ms: list[float] = []

    def _get_conn(self, db_id: str) -> sqlite3.Connection:
        if db_id not in self._conn_cache:
            db_path = self.index_dir / f"{db_id}.db"
            if not db_path.exists():
                raise FileNotFoundError(
                    f"Index not found for db_id='{db_id}'. "
                    f"Run: python src/schema_indexer.py --db_id {db_id}"
                )
            conn = sqlite3.connect(str(db_path))
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._conn_cache[db_id] = conn
        return self._conn_cache[db_id]

    def _encode_query(self, query: str) -> bytes:
        t0 = time.perf_counter()
        emb = self.model.encode([query], normalize_embeddings=True)[0]
        self._encode_time_ms.append((time.perf_counter() - t0) * 1000)
        return emb.astype(np.float32).tobytes()

    def retrieve(
        self,
        query: str,
        db_id: str,
        k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve Top-K tables for a NL query from the given database.

        Returns list of RetrievalResult sorted by distance (ascending).
        """
        query_vec = self._encode_query(query)
        conn = self._get_conn(db_id)

        t0 = time.perf_counter()
        rows = conn.execute(
            """
            SELECT
                m.db_id,
                m.table_name,
                v.distance,
                m.code_repr,
                m.retrieval_text
            FROM schema_vectors v
            JOIN schema_metadata m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            [query_vec, k],
        ).fetchall()
        self._search_time_ms.append((time.perf_counter() - t0) * 1000)

        results = []
        for db_id_row, table_name, distance, code_repr, retrieval_text in rows:
            # sqlite-vec returns squared L2 distance for normalized vectors
            # For unit vectors: L2² = 2(1 - cosine_sim)  =>  cosine_sim = 1 - L2²/2
            score = max(0.0, 1.0 - distance / 2.0)
            results.append(RetrievalResult(
                db_id=db_id_row,
                table_name=table_name,
                distance=distance,
                score=score,
                code_repr=code_repr,
                retrieval_text=retrieval_text,
            ))
        return results

    def retrieve_tables_names(self, query: str, db_id: str, k: int = 5) -> list[str]:
        """Convenience: return just the table names."""
        return [r.table_name for r in self.retrieve(query, db_id, k)]

    def latency_stats(self) -> dict:
        """Return average encode + search latency in ms."""
        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "encode_avg_ms": _avg(self._encode_time_ms),
            "search_avg_ms": _avg(self._search_time_ms),
            "total_avg_ms":  _avg(self._encode_time_ms) + _avg(self._search_time_ms),
            "n_queries": len(self._encode_time_ms),
        }

    def close(self):
        for conn in self._conn_cache.values():
            conn.close()
        self._conn_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    retriever = SchemaRetriever()

    test_cases = [
        ("How many singers do we have?", "concert_singer"),
        ("Find the name of the stadium with the highest capacity.", "concert_singer"),
        ("List all employees and their departments.", "department_management"),
    ]

    for query, db_id in test_cases:
        idx_path = DEFAULT_INDEX_DIR / f"{db_id}.db"
        if not idx_path.exists():
            print(f"[SKIP] Index not found for {db_id} — run schema_indexer.py first")
            continue

        print(f"\n── Query: '{query}' (db={db_id}) ──")
        results = retriever.retrieve(query, db_id, k=5)
        for i, r in enumerate(results):
            print(f"  #{i+1} {r.table_name:<30} score={r.score:.4f}  dist={r.distance:.4f}")

    print("\n── Latency ──")
    stats = retriever.latency_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" + (" ms" if "ms" in k else ""))

    retriever.close()
