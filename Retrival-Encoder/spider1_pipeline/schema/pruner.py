"""
Two-Stage Schema Pruner (Stage 2).

Stage 1: arctic-embed-m retrieves top-K most relevant TABLES (coarse).
Stage 2: Qwen 3.5 reads those tables' schemas and outputs only the
         COLUMNS needed to answer the question (fine-grained).

Output: dict[table_name → list[column_name]] — or None on parse failure
        (pipeline falls back to all columns when None is returned).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from schema.loader import SpiderSchema
from schema.serializer import serialize_schema_code_repr


_SYSTEM_PROMPT = (
    "You are an expert SQL schema analyst. "
    "Given a natural language question and a set of database table schemas, "
    "identify exactly which columns from each table are needed to answer the question. "
    "Be concise — only include columns that will appear in SELECT, WHERE, JOIN, "
    "GROUP BY, ORDER BY, or HAVING clauses."
)

_USER_TEMPLATE = """\
Question: {question}

Schemas:
{schema_sql}

Return ONLY a valid JSON object mapping each table name to a list of its needed column names.
Example: {{"table_a": ["col1", "col2"], "table_b": ["col3"]}}
If a table is not needed at all, omit it or set its list to [].
JSON:"""

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ   = re.compile(r"\{.*\}", re.DOTALL)


def _parse_pruned(text: str) -> Optional[dict[str, list[str]]]:
    """Try to extract a JSON dict from LLM output."""
    # strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # try fenced code block first
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()

    # fall back to first {...} in the response
    m = _JSON_OBJ.search(text)
    if not m:
        return None

    try:
        parsed = json.loads(m.group())
        if isinstance(parsed, dict):
            return {str(k): list(v) for k, v in parsed.items() if isinstance(v, list)}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


class SchemaPruner:
    """
    Uses the LLM to identify which columns in the retrieved tables
    are actually needed to answer a question.

    Parameters
    ----------
    llm_client  : any LLMClient with a .complete() method
    max_tokens  : token budget for the pruning response
    fallback_on_error : if True (default), return None on failure so the
                        pipeline can use all columns as a safe fallback
    """

    def __init__(
        self,
        llm_client,
        max_tokens: int = 256,
        fallback_on_error: bool = True,
    ):
        self.llm   = llm_client
        self.max_tokens = max_tokens
        self.fallback   = fallback_on_error

    def prune(
        self,
        question: str,
        schema: SpiderSchema,
        selected_tables: list[str],
    ) -> Optional[dict[str, list[str]]]:
        """
        Ask the LLM which columns are needed.

        Returns dict[table_name → [col_name, ...]] or None on failure.
        """
        if not selected_tables:
            return None

        schema_sql = serialize_schema_code_repr(schema, selected_tables=selected_tables)
        user_msg = _USER_TEMPLATE.format(
            question=question,
            schema_sql=schema_sql,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        try:
            resp = self.llm.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
                n=1,
            )
            result = _parse_pruned(resp.text)
            if result is None and not self.fallback:
                raise ValueError(f"Could not parse pruned columns from: {resp.text!r}")
            return result
        except Exception as e:
            if not self.fallback:
                raise
            print(f"[pruner] WARNING: pruning failed ({e}); using full schema")
            return None
