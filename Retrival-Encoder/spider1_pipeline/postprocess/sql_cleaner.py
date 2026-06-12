"""
SQL Post-processor for Spider 1.0.

Handles:
  1. Extracting SQL from raw LLM output (which may include prose)
  2. Normalising whitespace, casing, and common syntax issues
  3. Self-consistency voting (majority vote by execution result)
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional


# ── SQL extraction ────────────────────────────────────────────────────────────

_SQL_START   = re.compile(
    r"(?:^|\n)\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|DROP|ALTER)",
    re.IGNORECASE,
)
_CODE_BLOCK  = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Full <think>…</think> pair (standard)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Orphan </think> — Ollama strips the opening tag but keeps </think>;
# everything before it (including any preceding newlines) is thinking content
_ORPHAN_THINK_END = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove Qwen3/Qwen3.5 thinking blocks in all observed Ollama formats."""
    # 1. Strip complete <think>…</think> pairs
    text = _THINK_BLOCK.sub("", text)
    # 2. If an orphan </think> still exists, everything before it is thinking
    if "</think>" in text.lower():
        text = _ORPHAN_THINK_END.sub("", text)
    return text.strip()


def extract_sql(text: str) -> str:
    """
    Extract the SQL query from a potentially noisy LLM response.

    Strategy:
      0. Strip Qwen3/Qwen3.5 thinking blocks (<think>…</think> or orphan </think>)
      1. Look for fenced code block  ```sql ... ```
      2. Pick the LAST SQL statement in the text (Qwen3.5 puts the answer last)
      3. Fall back to returning the full text stripped
    """
    text = _strip_thinking(text)

    # Try fenced code block first
    match = _CODE_BLOCK.search(text)
    if match:
        return match.group(1).strip()

    # Find ALL SQL start positions; take the last one (final answer, not thinking examples)
    matches = list(_SQL_START.finditer(text))
    if matches:
        last = matches[-1]
        return text[last.start():].strip()

    return text.strip()


# ── SQL cleaning ──────────────────────────────────────────────────────────────

def clean_sql(sql: str) -> str:
    """
    Normalize a SQL query:
      - Collapse whitespace
      - Remove trailing semicolons (evaluation scripts add them)
      - Normalise comparison operators with extra spaces
    """
    sql = sql.strip()
    # Collapse newlines and multiple spaces
    sql = re.sub(r"\s+", " ", sql)
    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()
    # Fix spaced operators (common LLM artifact)
    sql = re.sub(r">\s+=", ">=", sql)
    sql = re.sub(r"<\s+=", "<=", sql)
    sql = re.sub(r"!\s+=", "!=", sql)
    return sql


# ── Common error fixes ────────────────────────────────────────────────────────

# Matches double-quoted string literals in value positions (after = != <> LIKE IN ,).
# LLMs frequently emit  WHERE col = "value"  instead of  WHERE col = 'value'.
# This converts them to single-quoted form so SQLite treats them as strings,
# not as (potentially missing) column identifiers.
_DQ_VALUE_RE = re.compile(
    r'((?:=|!=|<>|\bLIKE|\bNOT\s+LIKE|\bIN\s*\(|,)\s*)"([^"]+?)"',
    re.IGNORECASE,
)


def fix_common_errors(sql: str) -> str:
    """
    Apply rule-based fixes for frequent LLM SQL mistakes documented in C3SQL.

    Rules:
      - Remove double SELECT at the start (model sometimes echoes the prompt)
      - Strip orphan leading/trailing parentheses from thinking leakage
      - Ensure the query starts with SELECT if it doesn't start with WITH
      - Convert double-quoted string literals in value positions to single-quoted
    """
    sql = clean_sql(sql)

    # Fix double SELECT
    sql = re.sub(r"^SELECT\s+SELECT\s+", "SELECT ", sql, flags=re.IGNORECASE)

    # Fix double-quoted string literals in value positions → single-quoted
    sql = _DQ_VALUE_RE.sub(lambda m: m.group(1) + "'" + m.group(2) + "'", sql)

    # Remove orphan ')' anywhere in the query (depth-tracking, respects strings)
    if sql.count("(") != sql.count(")"):
        result, depth, in_str = [], 0, False
        for ch in sql:
            if ch == "'" and not in_str:
                in_str = True;  result.append(ch)
            elif ch == "'" and in_str:
                in_str = False; result.append(ch)
            elif in_str:
                result.append(ch)
            elif ch == "(":
                depth += 1; result.append(ch)
            elif ch == ")":
                if depth > 0:
                    depth -= 1; result.append(ch)
                # else orphan ')' — silently drop
            else:
                result.append(ch)
        sql = "".join(result).strip()

    # If response is just the non-SELECT part, prepend SELECT
    if not re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)", sql, re.IGNORECASE):
        sql = "SELECT " + sql

    return sql


# ── SQL validation ────────────────────────────────────────────────────────────

def is_executable(sql: str, db_path: Path) -> bool:
    """Return True if the SQL executes without error on the target SQLite DB."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        # Replace CURDATE() with a literal year (C3SQL convention)
        sql_adj = re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020",
                         sql, flags=re.IGNORECASE)
        cur.execute(sql_adj)
        cur.fetchall()
        conn.close()
        return True
    except Exception:
        return False


# ── Self-consistency voting ───────────────────────────────────────────────────

def _exec_result(sql: str, db_path: Path) -> Optional[frozenset]:
    """
    Execute SQL and return a canonical, column-order-insensitive frozenset of
    result rows, or None on error.

    Uses the same canonical form as the evaluator (_rows_canonical) so that
    SC voting groups candidates by the same equivalence relation as EX scoring.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        sql_adj = re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020",
                         sql, flags=re.IGNORECASE)
        cur.execute(sql_adj)
        rows = frozenset(
            tuple(sorted(
                str(v).strip().lower() if v is not None else "null"
                for v in row
            ))
            for row in cur.fetchall()
        )
        conn.close()
        return rows
    except Exception:
        return None


def self_consistency_vote(
    candidates: list[str],
    db_path: Path,
) -> str:
    """
    Majority-vote self-consistency (C3SQL / DAIL-SQL strategy).

    Execute all candidates on the target DB, group by result set,
    and return the SQL from the largest group.
    If all fail, return the first candidate.
    """
    if len(candidates) == 1:
        return candidates[0]

    groups: dict[frozenset, list[str]] = {}
    for sql in candidates:
        result = _exec_result(sql, db_path)
        if result is None:
            continue
        groups.setdefault(result, []).append(sql)

    if not groups:
        return candidates[0]

    # Return a representative SQL from the most common result set
    best_group = max(groups.values(), key=len)
    return best_group[0]
