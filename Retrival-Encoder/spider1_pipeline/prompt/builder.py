"""
Prompt Builder for Spider 1.0 Text-to-SQL.

Assembles the final prompt from:
  1. Schema section  — Code Representation (CREATE TABLE SQL)
  2. Few-shot section — Question/SQL pairs (DAIL organization)
  3. Question section — target NL question

Follows the DAIL-SQL "Code Representation Prompt" which achieves the
highest accuracy on Spider (86.2% EX with GPT-4).

Prompt structure:
  /* Given the following database schema: */
  CREATE TABLE ...

  /* Q: <example_q1> */ <example_sql1>
  /* Q: <example_q2> */ <example_sql2>
  ...

  /* Answer the following: <target_question> */
  SELECT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from schema.loader import SpiderSchema
from schema.serializer import serialize_schema_code_repr, serialize_schema_text_repr
from examples.selector import ExampleItem


@dataclass
class PromptResult:
    prompt: str
    schema_repr: str
    few_shot_block: str
    question_block: str
    estimated_tokens: int


class PromptBuilder:
    """
    Builds LLM-ready prompts for Spider 1.0 Text-to-SQL.

    Parameters
    ----------
    repr_type:          "code" (CREATE TABLE) | "text" (natural language)
    include_fk:         whether to include FOREIGN KEY constraints in schema
    include_samples:    whether to append sample column values as SQL comments
    max_schema_tables:  if given, truncate schema to first N tables (after retrieval)
    """

    SYSTEM_PROMPT = (
        "You are an expert SQLite SQL assistant. "
        "Given a database schema and a natural language question, "
        "output ONLY a valid SQLite SQL query — no explanation, no markdown, no commentary. "
        "The query must end with a semicolon only if it is part of the SQL syntax. "
        "Do not output anything other than the SQL query itself."
    )

    def __init__(
        self,
        repr_type: str = "code",
        include_samples: bool = False,
        max_tokens: int = 4096,
        max_ans_tokens: int = 512,
    ):
        self.repr_type      = repr_type
        self.include_samples = include_samples
        self.max_tokens     = max_tokens
        self.max_ans_tokens = max_ans_tokens

    # ── Schema serialization ──────────────────────────────────────────────────

    def _build_schema_section(
        self,
        schema: SpiderSchema,
        selected_tables: Optional[list[str]],
        db_path: Optional[Path],
        pruned_columns: Optional[dict[str, list[str]]] = None,
    ) -> str:
        if self.repr_type == "code":
            body = serialize_schema_code_repr(
                schema,
                selected_tables=selected_tables,
                selected_columns=pruned_columns,
                include_samples=self.include_samples,
                db_path=db_path,
            )
            return f"/* Given the following database schema: */\n{body}"
        else:
            body = serialize_schema_text_repr(schema, selected_tables=selected_tables)
            return f"Given the following database schema:\n{body}"

    EXAMPLE_PREFIX = "/* Some SQL examples are provided based on similar problems: */"

    # ── Few-shot section ──────────────────────────────────────────────────────

    def _build_fewshot_section(
        self,
        examples: list[ExampleItem],
    ) -> str:
        if not examples:
            return ""
        blocks = []
        for ex in examples:
            q_clean   = ex.question.replace("*/", "* /")
            sql_clean = " ".join(ex.sql.split())
            blocks.append(f"/* Answer the following with no explanation: {q_clean} */\n{sql_clean}")
        return self.EXAMPLE_PREFIX + "\n" + "\n\n".join(blocks)

    # ── Question section ──────────────────────────────────────────────────────

    def _build_question_section(self, question: str) -> str:
        if self.repr_type == "code":
            return f"/* Answer the following with no explanation: {question} */\nSELECT"
        else:
            return f"Answer the following with no explanation: {question}\nSELECT"

    # ── Main build ────────────────────────────────────────────────────────────

    def build(
        self,
        question: str,
        schema: SpiderSchema,
        examples: Optional[list[ExampleItem]] = None,
        selected_tables: Optional[list[str]] = None,
        db_path: Optional[Path] = None,
        pruned_columns: Optional[dict[str, list[str]]] = None,
    ) -> PromptResult:
        """
        Assemble the complete prompt for the LLM.

        pruned_columns: dict[table_name → [col_name, ...]] from SchemaPruner;
                        when provided, only those columns appear in CREATE TABLE.
        Returns PromptResult with the full prompt string and component parts.
        """
        schema_section   = self._build_schema_section(schema, selected_tables, db_path, pruned_columns)
        fewshot_section  = self._build_fewshot_section(examples or [])
        question_section = self._build_question_section(question)

        parts = [schema_section]
        if fewshot_section:
            parts.append(fewshot_section)
        parts.append(question_section)

        prompt = "\n\n".join(parts)
        approx_tokens = len(prompt.split()) * 4 // 3   # rough estimate

        return PromptResult(
            prompt=prompt,
            schema_repr=schema_section,
            few_shot_block=fewshot_section,
            question_block=question_section,
            estimated_tokens=approx_tokens,
        )

    def build_messages(
        self,
        question: str,
        schema: SpiderSchema,
        examples: Optional[list[ExampleItem]] = None,
        selected_tables: Optional[list[str]] = None,
        db_path: Optional[Path] = None,
        pruned_columns: Optional[dict[str, list[str]]] = None,
    ) -> list[dict]:
        """
        Build OpenAI-style chat messages list.

        Returns [{"role": "system", ...}, {"role": "user", ...}]
        """
        result = self.build(
            question=question,
            schema=schema,
            examples=examples,
            selected_tables=selected_tables,
            db_path=db_path,
            pruned_columns=pruned_columns,
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": result.prompt},
        ]
