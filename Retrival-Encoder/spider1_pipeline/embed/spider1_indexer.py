"""
Spider 1.0 Schema Indexer.

Builds per-database plain-SQLite BLOB indexes using ArcticEmbedModel
(snowflake-arctic-embed-m, 768-dim), with optional column sample-value
metadata injection for richer semantic embeddings.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Resolve minilm_core + spider1_pipeline on sys.path
_PROJECT_ROOT    = Path(__file__).parent.parent.parent
_PIPELINE_ROOT   = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "minilm_core" / "src"))
sys.path.insert(0, str(_PIPELINE_ROOT))

from embed.arctic_embed import ArcticEmbedModel
from schema.loader import SpiderSchema, load_tables_json


def build_index_for_schema(
    schema: SpiderSchema,
    embed_model: ArcticEmbedModel,
    index_dir: Path,
    force: bool = False,
    database_dir: Optional[Path] = None,
) -> Path:
    """
    Build a plain-SQLite BLOB index for one Spider 1.0 database.

    Each row stores one table's enriched retrieval text + its 768-dim float32
    embedding.  No sqlite-vec / enable_load_extension needed.
    Retrieval is numpy cosine similarity in Spider1Retriever.

    Parameters
    ----------
    database_dir : if provided, each table's retrieval text is enriched with
                   column types and up to 3 sample values from the live SQLite DB.
    """
    index_path = index_dir / f"{schema.db_id}.db"
    if index_path.exists() and not force:
        return index_path

    db_path = None
    if database_dir:
        candidate = database_dir / schema.db_id / f"{schema.db_id}.sqlite"
        if candidate.exists():
            db_path = candidate

    retrieval_pairs = schema.to_retrieval_texts(db_path=db_path)  # [(tname, text)]
    if not retrieval_pairs:
        return index_path

    table_names = [tname for tname, _ in retrieval_pairs]
    texts       = [text  for _, text  in retrieval_pairs]

    embeddings = embed_model.encode_documents(texts, show_progress=False)

    if index_path.exists():
        index_path.unlink()

    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_path))
    conn.execute(
        "CREATE TABLE schema_index ("
        "rowid INTEGER PRIMARY KEY, db_id TEXT NOT NULL, "
        "table_name TEXT NOT NULL, retrieval_text TEXT NOT NULL, "
        "embedding BLOB NOT NULL)"
    )
    for i, (tname, text, emb) in enumerate(zip(table_names, texts, embeddings), start=1):
        conn.execute(
            "INSERT INTO schema_index VALUES (?, ?, ?, ?, ?)",
            (i, schema.db_id, tname, text, emb.tobytes()),
        )
    conn.commit()
    conn.close()
    return index_path


def build_spider1_indexes(
    tables_json: Path,
    model_dir: Path,
    index_dir: Path,
    force: bool = False,
    db_id_filter: Optional[str] = None,
    device: str = "mps",
    database_dir: Optional[Path] = None,
) -> None:
    """
    Build BLOB-based embedding indexes for all Spider 1.0 databases.

    Parameters
    ----------
    model_dir    : path to snowflake-arctic-embed-m weights
    database_dir : Spider database root; when provided, sample values are
                   injected into each table's retrieval text for richer embeddings.
    """
    from tqdm import tqdm

    embed_model = ArcticEmbedModel(model_dir=model_dir, device=device)

    print(f"[spider1-indexer] Loading tables from {tables_json} ...")
    schemas = load_tables_json(tables_json)

    if db_id_filter:
        if db_id_filter not in schemas:
            raise ValueError(f"db_id '{db_id_filter}' not found in tables.json")
        schemas = {db_id_filter: schemas[db_id_filter]}

    sample_note = f" (+ sample values from {database_dir})" if database_dir else ""
    print(f"[spider1-indexer] Indexing {len(schemas)} databases{sample_note} ...")

    index_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for db_id, schema in tqdm(schemas.items(), desc="Indexing"):
        build_index_for_schema(schema, embed_model, index_dir,
                               force=force, database_dir=database_dir)

    elapsed  = time.perf_counter() - t0
    db_files = list(index_dir.glob("*.db"))
    total_mb = sum(f.stat().st_size for f in db_files) / (1024 ** 2)
    print(f"[spider1-indexer] Done: {len(schemas)} DBs in {elapsed:.1f}s")
    print(f"                  {len(db_files)} index files, {total_mb:.1f} MB total")
