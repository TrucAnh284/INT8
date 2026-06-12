"""
Schema Serializers for Spider 1.0.

Provides two formats used in DAIL-SQL experiments:
  - Code Representation (CREATE TABLE SQL) — best performing on Spider
  - Text Representation (natural language table/column list)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .loader import SpiderSchema, SpiderTable, SpiderColumn, get_sample_values


# ── Code Representation ───────────────────────────────────────────────────────

def _col_ddl(col: SpiderColumn, include_samples: bool = False,
             db_path: Optional[Path] = None) -> str:
    """Format a single column definition for CREATE TABLE."""
    type_map = {
        "text":    "TEXT",
        "number":  "INT",
        "time":    "DATETIME",
        "boolean": "BOOLEAN",
        "others":  "TEXT",
    }
    sql_type = type_map.get(col.col_type.lower(), "TEXT")

    parts = [f'  "{col.name_original}" {sql_type}']
    if col.is_primary_key:
        parts.append("PRIMARY KEY")
    ddl = " ".join(parts)

    if include_samples and db_path and col.col_type.lower() in ("text", "others"):
        vals = get_sample_values(db_path, _get_table_name(col), col.name_original)
        if vals:
            sample_str = ", ".join(f'"{v}"' for v in vals[:3])
            ddl += f"  -- e.g. {sample_str}"
    return ddl


def _get_table_name(col: SpiderColumn) -> str:
    return ""  # placeholder; actual name fetched at table level


def serialize_table_create(
    table: SpiderTable,
    all_fks: list[tuple[str, str, str, str]],
    include_samples: bool = False,
    db_path: Optional[Path] = None,
    selected_columns: Optional[list[str]] = None,
) -> str:
    """
    Serialize one table as a CREATE TABLE statement.

    all_fks: list of (src_table, src_col, tgt_table, tgt_col)
    selected_columns: if given, only include those column names (after LLM pruning)
    """
    sel_lower = {c.lower() for c in selected_columns} if selected_columns else None

    col_lines = []
    for col in table.columns:
        if col.name_original == "*":
            continue
        if sel_lower is not None and col.name_original.lower() not in sel_lower:
            if not col.is_primary_key:   # always keep PKs for JOIN correctness
                continue
        type_map = {
            "text":    "TEXT",
            "number":  "INT",
            "time":    "DATETIME",
            "boolean": "BOOLEAN",
            "others":  "TEXT",
        }
        sql_type = type_map.get(col.col_type.lower(), "TEXT")
        pk_suffix = " PRIMARY KEY" if col.is_primary_key else ""

        line = f'  "{col.name_original}" {sql_type}{pk_suffix}'

        if include_samples and db_path:
            vals = get_sample_values(db_path, table.name_original, col.name_original)
            if vals:
                sample_str = ", ".join(f'"{v}"' for v in vals[:3])
                line += f"  -- e.g. {sample_str}"

        col_lines.append(line)

    # Foreign key constraints (only for included columns)
    included = {col.name_original.lower() for col in table.columns
                if not (sel_lower and col.name_original.lower() not in sel_lower and not col.is_primary_key)}
    fk_constraints = []
    for src_tbl, src_col, tgt_tbl, tgt_col in all_fks:
        if src_tbl.lower() == table.name_original.lower() and src_col.lower() in included:
            fk_constraints.append(
                f'  FOREIGN KEY ("{src_col}") REFERENCES "{tgt_tbl}" ("{tgt_col}")'
            )

    all_lines = col_lines + fk_constraints
    body = ",\n".join(all_lines)
    return f'CREATE TABLE "{table.name_original}" (\n{body}\n)'


def serialize_schema_code_repr(
    schema: SpiderSchema,
    selected_tables: Optional[list[str]] = None,
    selected_columns: Optional[dict[str, list[str]]] = None,
    include_samples: bool = False,
    db_path: Optional[Path] = None,
) -> str:
    """
    Serialize entire schema as CREATE TABLE SQL statements.

    selected_tables:  if given, only include those table names.
    selected_columns: dict[table_name → [col_name, ...]] from LLM pruning;
                      columns not listed are dropped (PKs always kept).
    Returns CREATE TABLE statements joined by blank lines.
    """
    all_fks: list[tuple[str, str, str, str]] = []
    for tbl in schema.tables:
        for col in tbl.columns:
            if col.foreign_key_to:
                tgt_tbl, tgt_col = col.foreign_key_to
                all_fks.append((tbl.name_original, col.name_original, tgt_tbl, tgt_col))

    tables_to_use = schema.tables
    if selected_tables:
        sel_lower = {t.lower() for t in selected_tables}
        tables_to_use = [t for t in schema.tables
                         if t.name_original.lower() in sel_lower]

    parts = []
    for tbl in tables_to_use:
        tbl_cols = None
        if selected_columns:
            # look up by original name or lower-cased key
            tbl_cols = (selected_columns.get(tbl.name_original)
                        or selected_columns.get(tbl.name_original.lower()))
        parts.append(
            serialize_table_create(tbl, all_fks, include_samples, db_path, tbl_cols)
        )
    return "\n\n".join(parts)


# ── Text Representation ───────────────────────────────────────────────────────

def serialize_schema_text_repr(
    schema: SpiderSchema,
    selected_tables: Optional[list[str]] = None,
) -> str:
    """
    Serialize schema as natural-language table/column list.

    Format:
      Table singer: Singer_ID (INT, PK), Name (TEXT), Country (TEXT), ...
      ...
      Foreign keys: singer.People_ID = people.People_ID
    """
    tables_to_use = schema.tables
    if selected_tables:
        sel_lower = {t.lower() for t in selected_tables}
        tables_to_use = [t for t in schema.tables
                         if t.name_original.lower() in sel_lower]

    lines = []
    fk_lines = []

    for tbl in tables_to_use:
        type_map = {
            "text":    "TEXT",
            "number":  "INT",
            "time":    "DATETIME",
            "boolean": "BOOL",
            "others":  "TEXT",
        }
        col_descs = []
        for col in tbl.columns:
            t = type_map.get(col.col_type.lower(), "TEXT")
            suffix = ", PK" if col.is_primary_key else ""
            col_descs.append(f"{col.name_original} ({t}{suffix})")
            if col.foreign_key_to:
                tgt_tbl, tgt_col = col.foreign_key_to
                fk_lines.append(
                    f"  {tbl.name_original}.{col.name_original} = "
                    f"{tgt_tbl}.{tgt_col}"
                )

        lines.append(f"Table {tbl.name_original}: " + ", ".join(col_descs))

    if fk_lines:
        lines.append("Foreign keys:")
        lines.extend(fk_lines)

    return "\n".join(lines)
