"""
Schema Serializer for minilm_core — Spider 1.0 tables.json adapter.

Provides:
  load_tables_json(path) → dict[db_id, list[TableSchema]]
  TableSchema             — single-table container with retrieval_text + code_repr

Used by src/schema_indexer.py to build sqlite-vec embeddings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── TableSchema dataclass ─────────────────────────────────────────────────────

@dataclass
class TableSchema:
    """One table inside a Spider 1.0 database."""

    table_name: str                          # original-casing table name
    column_names: list[str]                  # original-casing column names (no *)
    column_types: list[str]                  # matching types: text|number|time|boolean|others
    primary_keys: list[str] = field(default_factory=list)   # pk column names
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)
    # foreign_keys: list of (src_col, tgt_table, tgt_col)

    def to_retrieval_text(self) -> str:
        """
        Compact retrieval text for MiniLM embedding.

        Format: "{table_name}: col1, col2, ... | FK: col→tgt_table.tgt_col, ..."
        """
        col_part = ", ".join(self.column_names)
        text = f"{self.table_name}: {col_part}"
        if self.foreign_keys:
            fk_parts = [f"{src}→{tgt_tbl}.{tgt_col}"
                        for src, tgt_tbl, tgt_col in self.foreign_keys]
            text += " | FK: " + ", ".join(fk_parts)
        return text

    def to_code_repr(self) -> str:
        """
        CREATE TABLE SQL representation used in code-repr prompts.

        Format:
            CREATE TABLE "table_name" (
              "col1" TEXT PRIMARY KEY,
              "col2" INT,
              FOREIGN KEY ("src") REFERENCES "tgt_table" ("tgt_col")
            )
        """
        type_map = {
            "text":    "TEXT",
            "number":  "INT",
            "time":    "DATETIME",
            "boolean": "BOOLEAN",
            "others":  "TEXT",
        }

        lines: list[str] = []
        for col, ctype in zip(self.column_names, self.column_types):
            sql_type = type_map.get(ctype.lower(), "TEXT")
            pk_suffix = " PRIMARY KEY" if col in self.primary_keys else ""
            lines.append(f'  "{col}" {sql_type}{pk_suffix}')

        for src, tgt_tbl, tgt_col in self.foreign_keys:
            lines.append(
                f'  FOREIGN KEY ("{src}") REFERENCES "{tgt_tbl}" ("{tgt_col}")'
            )

        body = ",\n".join(lines)
        return f'CREATE TABLE "{self.table_name}" (\n{body}\n)'


# ── Loader ────────────────────────────────────────────────────────────────────

def load_tables_json(
    tables_json_path: Path,
) -> dict[str, list[TableSchema]]:
    """
    Parse Spider 1.0 tables.json.

    Returns dict[db_id → list[TableSchema]], one TableSchema per table.
    """
    raw: list[dict] = json.loads(Path(tables_json_path).read_text(encoding="utf-8"))
    result: dict[str, list[TableSchema]] = {}

    for db in raw:
        db_id: str               = db["db_id"]
        table_names_orig         = db["table_names_original"]
        col_names_orig_raw       = db["column_names_original"]  # [[tbl_idx, col], ...]
        col_types: list[str]     = db["column_types"]
        primary_keys: list[int]  = db["primary_keys"]
        foreign_keys_raw         = db["foreign_keys"]           # [[src_idx, tgt_idx], ...]

        pk_set: set[int] = set(primary_keys)

        # Build FK lookup: col_idx → (tgt_table_name, tgt_col_name)
        fk_map: dict[int, tuple[str, str]] = {}
        for src_idx, tgt_idx in foreign_keys_raw:
            tgt_tbl_idx = col_names_orig_raw[tgt_idx][0]
            tgt_col     = col_names_orig_raw[tgt_idx][1]
            fk_map[src_idx] = (table_names_orig[tgt_tbl_idx], tgt_col)

        # Group columns by table
        table_cols: dict[int, list[tuple[int, str, str]]] = {
            i: [] for i in range(len(table_names_orig))
        }
        for col_idx, (tbl_idx, col_name) in enumerate(col_names_orig_raw):
            if tbl_idx == -1:
                continue  # skip the * wildcard column
            table_cols[tbl_idx].append((col_idx, col_name, col_types[col_idx]))

        schemas: list[TableSchema] = []
        for t_idx, t_name in enumerate(table_names_orig):
            cols_info  = table_cols.get(t_idx, [])
            col_names  = [c[1] for c in cols_info]
            ctypes     = [c[2] for c in cols_info]
            pks        = [c[1] for c in cols_info if c[0] in pk_set]
            fks: list[tuple[str, str, str]] = []
            for col_idx, col_name, _ in cols_info:
                if col_idx in fk_map:
                    tgt_tbl, tgt_col = fk_map[col_idx]
                    fks.append((col_name, tgt_tbl, tgt_col))

            schemas.append(TableSchema(
                table_name=t_name,
                column_names=col_names,
                column_types=ctypes,
                primary_keys=pks,
                foreign_keys=fks,
            ))

        result[db_id] = schemas

    return result
