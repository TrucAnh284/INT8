"""
Schema Indexer — Build sqlite-vec index from serialized Spider schemas.

For each database, creates a sqlite-vec virtual table storing
384-dim embeddings of each table's retrieval text.

Usage:
    python src/schema_indexer.py                          # index all DBs
    python src/schema_indexer.py --db_id concert_singer  # single DB
    python src/schema_indexer.py --force                  # rebuild
"""

import json
import sqlite3
import struct
import argparse
import time
from pathlib import Path

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from schema_serializer import load_tables_json, TableSchema

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_TABLES_JSON = REPO_ROOT / "dataset" / "spider" / "tables.json"
DEFAULT_MINILM_DIR  = REPO_ROOT / "models" / "minilm"
DEFAULT_INDEX_DIR   = REPO_ROOT / "dataset" / "spider" / "vec_index"

EMBEDDING_DIM = 384


def _encode_vector(vec: np.ndarray) -> bytes:
    """Serialize float32 numpy array to bytes for sqlite-vec."""
    return vec.astype(np.float32).tobytes()


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def build_index_for_database(
    db_id: str,
    tables: list[TableSchema],
    model: SentenceTransformer,
    index_dir: Path,
    force: bool = False,
) -> Path:
    """
    Build sqlite-vec index for one Spider database.
    Each row = one table's embedding.

    Returns path to the created .db file.
    """
    index_path = index_dir / f"{db_id}.db"

    if index_path.exists() and not force:
        return index_path

    # Collect retrieval texts
    texts = [t.to_retrieval_text() for t in tables]
    table_names = [t.table_name for t in tables]

    # Encode in one batch
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Write to sqlite-vec DB
    conn = _open_db(index_path)
    conn.execute("DROP TABLE IF EXISTS schema_metadata")
    conn.execute("DROP TABLE IF EXISTS schema_vectors")

    # Metadata table: rowid → table_name, retrieval_text, code_repr
    conn.execute("""
        CREATE TABLE schema_metadata (
            rowid      INTEGER PRIMARY KEY,
            db_id      TEXT NOT NULL,
            table_name TEXT NOT NULL,
            retrieval_text TEXT,
            code_repr  TEXT
        )
    """)

    # Vector table
    conn.execute(f"""
        CREATE VIRTUAL TABLE schema_vectors
        USING vec0(embedding float[{EMBEDDING_DIM}])
    """)

    rows_meta = []
    rows_vec  = []
    for i, (table, text, emb) in enumerate(zip(tables, texts, embeddings)):
        rowid = i + 1  # sqlite rowid is 1-indexed
        rows_meta.append((rowid, db_id, table.table_name, text, table.to_code_repr()))
        rows_vec.append((rowid, _encode_vector(emb)))

    conn.executemany(
        "INSERT INTO schema_metadata VALUES (?, ?, ?, ?, ?)", rows_meta
    )
    conn.executemany(
        "INSERT INTO schema_vectors(rowid, embedding) VALUES (?, ?)", rows_vec
    )
    conn.commit()
    conn.close()

    return index_path


def build_all_indexes(
    tables_json: Path = DEFAULT_TABLES_JSON,
    minilm_dir: Path = DEFAULT_MINILM_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    force: bool = False,
    db_id_filter: str | None = None,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model from {minilm_dir} ...")
    model = SentenceTransformer(str(minilm_dir))

    print(f"[INFO] Loading tables from {tables_json} ...")
    db_schemas = load_tables_json(tables_json)

    if db_id_filter:
        if db_id_filter not in db_schemas:
            raise ValueError(f"db_id '{db_id_filter}' not found in tables.json")
        db_schemas = {db_id_filter: db_schemas[db_id_filter]}

    print(f"[INFO] Building indexes for {len(db_schemas)} databases → {index_dir}")
    t0 = time.perf_counter()

    for db_id, tables in tqdm(db_schemas.items(), desc="Indexing"):
        build_index_for_database(db_id, tables, model, index_dir, force=force)

    elapsed = time.perf_counter() - t0
    print(f"[OK] Indexed {len(db_schemas)} databases in {elapsed:.1f}s")
    print(f"     Index directory: {index_dir}")

    # Quick stats
    db_files = list(index_dir.glob("*.db"))
    total_size_mb = sum(f.stat().st_size for f in db_files) / (1024 ** 2)
    print(f"     {len(db_files)} .db files, total {total_size_mb:.1f} MB")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=str(DEFAULT_TABLES_JSON))
    parser.add_argument("--minilm_dir", default=str(DEFAULT_MINILM_DIR))
    parser.add_argument("--index_dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--db_id", default=None, help="Index a single database only")
    parser.add_argument("--force", action="store_true", help="Rebuild existing indexes")
    args = parser.parse_args()

    build_all_indexes(
        tables_json=Path(args.tables_json),
        minilm_dir=Path(args.minilm_dir),
        index_dir=Path(args.index_dir),
        force=args.force,
        db_id_filter=args.db_id,
    )
