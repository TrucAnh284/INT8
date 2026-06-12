"""
Spider 1.0 Schema Loader.

Parses tables.json into typed dataclasses that the rest of the pipeline uses.
Also reads column sample values directly from the SQLite databases when needed.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SpiderColumn:
    col_idx: int               # global column index in tables.json
    table_idx: int             # which table this column belongs to (-1 = *)
    name: str                  # cleaned name
    name_original: str         # original casing
    col_type: str              # text | number | time | boolean | others
    is_primary_key: bool = False
    foreign_key_to: Optional[tuple[str, str]] = None  # (table_name, col_name)


@dataclass
class SpiderTable:
    table_idx: int
    name: str                  # cleaned name (lower)
    name_original: str         # original casing
    columns: list[SpiderColumn] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name_original for c in self.columns if c.name_original != "*"]

    @property
    def primary_keys(self) -> list[SpiderColumn]:
        return [c for c in self.columns if c.is_primary_key]


@dataclass
class SpiderSchema:
    db_id: str
    tables: list[SpiderTable] = field(default_factory=list)
    # table_name_original → SpiderTable
    _table_map: dict[str, SpiderTable] = field(default_factory=dict, repr=False)

    def get_table(self, name_original: str) -> Optional[SpiderTable]:
        return self._table_map.get(name_original.lower())

    def to_retrieval_texts(
        self,
        db_path: Optional[Path] = None,
        max_samples: int = 3,
    ) -> list[tuple[str, str]]:
        """
        Return list of (table_name_original, retrieval_text) for indexing.

        If db_path is provided, column descriptions include the SQL type and
        up to max_samples distinct sample values — this enriches the embedding
        so the model understands column semantics (e.g. knowing a "country"
        column holds values like "USA", "UK" rather than numbers).
        """
        type_map = {
            "text":    "TEXT",
            "number":  "INT",
            "time":    "DATETIME",
            "boolean": "BOOL",
            "others":  "TEXT",
        }
        results = []
        for tbl in self.tables:
            col_parts = []
            fk_parts  = []
            for c in tbl.columns:
                if c.name_original == "*":
                    continue
                sql_t = type_map.get(c.col_type.lower(), "TEXT")
                desc  = f"{c.name_original} ({sql_t})"
                if c.is_primary_key:
                    desc += " PK"
                if db_path:
                    vals = get_sample_values(db_path, tbl.name_original, c.name_original,
                                             max_vals=max_samples)
                    if vals:
                        desc += " e.g. " + ", ".join(f'"{v}"' for v in vals)
                col_parts.append(desc)
                if c.foreign_key_to:
                    fk_parts.append(
                        f"{c.name_original} → {c.foreign_key_to[0]}.{c.foreign_key_to[1]}"
                    )

            text = f"Table: {tbl.name_original} | Columns: " + "; ".join(col_parts)
            if fk_parts:
                text += " | FK: " + ", ".join(fk_parts)
            results.append((tbl.name_original, text))
        return results


def load_tables_json(tables_json_path: Path) -> dict[str, SpiderSchema]:
    """
    Load Spider 1.0 tables.json.

    Returns dict[db_id → SpiderSchema].
    """
    raw: list[dict] = json.loads(Path(tables_json_path).read_text(encoding="utf-8"))
    schemas: dict[str, SpiderSchema] = {}

    for db in raw:
        db_id: str = db["db_id"]
        table_names_orig: list[str] = db["table_names_original"]
        table_names: list[str]      = db["table_names"]
        col_names_raw: list[list]   = db["column_names"]
        col_names_orig_raw: list[list] = db["column_names_original"]
        col_types: list[str]        = db["column_types"]
        primary_keys: list[int]     = db["primary_keys"]
        foreign_keys: list[list]    = db["foreign_keys"]

        pk_set: set[int] = set(primary_keys)

        # Build FK lookup: col_idx → (target_table_name_orig, target_col_name_orig)
        fk_map: dict[int, tuple[str, str]] = {}
        for src_idx, tgt_idx in foreign_keys:
            tgt_tbl_idx = col_names_orig_raw[tgt_idx][0]
            tgt_col     = col_names_orig_raw[tgt_idx][1]
            tgt_tbl     = table_names_orig[tgt_tbl_idx]
            fk_map[src_idx] = (tgt_tbl, tgt_col)

        # Build table list
        tables: list[SpiderTable] = []
        for t_idx, (t_name, t_orig) in enumerate(zip(table_names, table_names_orig)):
            tables.append(SpiderTable(
                table_idx=t_idx,
                name=t_name,
                name_original=t_orig,
            ))

        # Attach columns to their tables
        for col_idx, ((tbl_idx, col_name), (_, col_name_orig), col_type) in enumerate(
            zip(col_names_raw, col_names_orig_raw, col_types)
        ):
            if tbl_idx == -1:
                continue  # skip the * column

            col = SpiderColumn(
                col_idx=col_idx,
                table_idx=tbl_idx,
                name=col_name,
                name_original=col_name_orig,
                col_type=col_type,
                is_primary_key=(col_idx in pk_set),
                foreign_key_to=fk_map.get(col_idx),
            )
            tables[tbl_idx].columns.append(col)

        table_map = {t.name_original.lower(): t for t in tables}
        schemas[db_id] = SpiderSchema(
            db_id=db_id,
            tables=tables,
            _table_map=table_map,
        )

    return schemas


def get_sample_values(
    db_path: Path,
    table_name: str,
    col_name: str,
    max_vals: int = 3,
) -> list[str]:
    """Read up to max_vals distinct non-null values from a column."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
            f'WHERE "{col_name}" IS NOT NULL LIMIT {max_vals}'
        )
        rows = cur.fetchall()
        conn.close()
        return [str(r[0]).strip() for r in rows if str(r[0]).strip()]
    except Exception:
        return []
