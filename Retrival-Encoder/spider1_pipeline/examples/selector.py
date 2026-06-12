"""
Few-Shot Example Selector for Spider 1.0.

Implements DAIL-SQL selection strategy:
  1. Mask schema-specific tokens from the question
  2. Embed masked questions with MiniLM
  3. Retrieve nearest neighbors by cosine similarity
  4. (Optional) Re-rank by SQL skeleton similarity

Also supports:
  - Random selection (baseline)
  - Pure question-similarity selection
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_PIPELINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "minilm_core" / "src"))
sys.path.insert(0, str(_PIPELINE_ROOT))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExampleItem:
    db_id: str
    question: str
    sql: str
    masked_question: str = ""      # question with schema tokens replaced
    sql_skeleton: str = ""         # SQL with values/names stripped


# ── SQL skeleton extractor ────────────────────────────────────────────────────

_SKEL_KEYWORDS = frozenset([
    "select", "from", "where", "group", "by", "order", "having", "limit",
    "join", "inner", "left", "right", "outer", "cross", "full", "natural",
    "on", "as", "distinct", "count", "sum", "avg", "max", "min",
    "and", "or", "not", "in", "like", "between", "exists",
    "union", "intersect", "except", "all",
    "asc", "desc",
    "case", "when", "then", "else", "end", "null", "is", "with",
])

_SKEL_PUNCT = frozenset(["(", ")", ","])

# Tokenize: quoted strings | dotted identifiers | multi-char ops | words | single-char ops
_SKEL_TOK_RE = re.compile(
    r"'[^']*'"         # single-quoted string
    r'|"[^"]*"'        # double-quoted identifier
    r"|\w+\.\w+"       # dotted identifier (T1.col or table.col)
    r"|<>|!=|<=|>="    # multi-char operators
    r"|\w+"            # plain word or number
    r"|[=<>()+\-*/,]"  # single-char ops and punct
)


def _lowercase_outside_quotes(sql: str) -> str:
    """Lowercase SQL while preserving string literal case."""
    result, in_sq = [], False
    for ch in sql:
        if ch == "'" and not in_sq:
            in_sq = True;  result.append(ch)
        elif ch == "'" and in_sq:
            in_sq = False; result.append(ch)
        elif in_sq:
            result.append(ch)
        else:
            result.append(ch.lower())
    return "".join(result)


def sql2skeleton(sql: str) -> str:
    """
    DAIL-SQL style skeleton extraction.

    Replaces ALL schema-specific tokens (table/column names, values, numbers)
    with '_', keeping only SQL structural keywords.  Handles:
      - Proper lowercase (preserving string literals)
      - Dotted identifiers (T1.col, tbl.col) → single '_'
      - JOIN chain collapse
      - WHERE / HAVING condition collapse
    """
    sql = _lowercase_outside_quotes(sql).rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)

    tokens = _SKEL_TOK_RE.findall(sql)
    masked: list[str] = []
    for tok in tokens:
        tl = tok.lower()
        if (tok.startswith("'") and tok.endswith("'")) or \
           (tok.startswith('"') and tok.endswith('"')):
            masked.append("_")
        elif re.match(r"^-?\d+(\.\d+)?$", tl):
            masked.append("_")
        elif tl in _SKEL_KEYWORDS:
            masked.append(tl)
        elif tl in _SKEL_PUNCT:
            masked.append(tl)
        elif tl in ("<>", "!=", "<=", ">=", "=", "<", ">", "+", "-", "/"):
            masked.append(tl)
        elif tl == "*":
            masked.append("*")
        elif "." in tok and re.match(r"^\w+\.\w+$", tok):
            masked.append("_")          # dotted identifier → single slot
        elif re.match(r"^\w+$", tok):
            masked.append("_")          # plain identifier (table/column name)
        else:
            masked.append(tl)

    skeleton = " ".join(masked)

    # Collapse table alias declarations: "_ as _" → "_"  (FROM tbl AS T1, JOIN tbl AS T2)
    while "_ as _" in skeleton:
        skeleton = skeleton.replace("_ as _", "_")

    # Remove JOIN ON conditions
    skeleton = re.sub(r" on _ = _ and _ = _", " on _ = _", skeleton)
    skeleton = re.sub(r" on _ = _ or _ = _",  " on _ = _", skeleton)
    skeleton = re.sub(r" on _ = _", "", skeleton)

    # Collapse JOIN chains: "_ [inner|left|...] join _ join _" → "_ join _"
    skeleton = re.sub(
        r"_(?:\s+(?:inner|left|right|outer|cross|full|natural)?\s*join\s+)+_",
        "_ join _",
        skeleton,
    )

    # "_ , _" → "_"
    while "_ , _" in skeleton:
        skeleton = skeleton.replace("_ , _", "_")

    # Collapse comparison conditions
    for op in ("=", "!=", "<>", ">", ">=", "<", "<="):
        skeleton = skeleton.replace(f"_ {op} _", "_")

    # Collapse WHERE / HAVING conjunctions
    for kw in ("where", "having"):
        while f"{kw} _ and _" in skeleton or f"{kw} _ or _" in skeleton:
            skeleton = skeleton.replace(f"{kw} _ and _", f"{kw} _")
            skeleton = skeleton.replace(f"{kw} _ or _",  f"{kw} _")

    skeleton = re.sub(r"\s+", " ", skeleton).strip()
    return skeleton


def extract_sql_skeleton(sql: str) -> str:
    """Backward-compatible alias for sql2skeleton."""
    return sql2skeleton(sql)


def mask_schema_tokens(question: str, schema_tokens: set[str]) -> str:
    """
    Replace schema-specific tokens (table/column names) with a generic
    [MASK] token so embeddings focus on question structure.
    """
    if not schema_tokens:
        return question

    pattern = r"\b(" + "|".join(re.escape(t) for t in sorted(schema_tokens,
                                                              key=len,
                                                              reverse=True)) + r")\b"
    return re.sub(pattern, "[MASK]", question, flags=re.IGNORECASE)


def _skeleton_similarity(s1: str, s2: str) -> float:
    """Jaccard similarity between skeleton token sets."""
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


# ── FewShotSelector ───────────────────────────────────────────────────────────

class FewShotSelector:
    """
    Builds a pool of candidate examples from training data and selects
    the K most relevant ones for a given test question.

    Supports three modes (set via `method` parameter):
      "dail"      — masked question embedding + optional skeleton re-rank (DAIL-SQL)
      "question"  — plain question embedding similarity
      "random"    — random K examples

    Usage:
        selector = FewShotSelector.from_file(train_json, method="dail")
        examples = selector.select(question, db_id, k=5)
    """

    def __init__(
        self,
        examples: list[ExampleItem],
        method: str = "dail",
        model=None,              # pre-built ArcticEmbedModel; loaded lazily if None
        model_dir: Optional[Path] = None,   # arctic-embed-m weights dir
        device: str = "mps",
        # legacy compat
        minilm_dir: Optional[Path] = None,
        profile: str = "MP-Balanced",
        skeleton_threshold: float = 0.0,
        cross_domain: bool = True,
    ):
        self.examples   = examples
        self.method     = method
        self.skeleton_threshold = skeleton_threshold
        self.cross_domain = cross_domain
        self._model_dir   = model_dir or minilm_dir   # accept either name
        self._device      = device
        self._embs: Optional[np.ndarray] = None
        # Accept a pre-built ArcticEmbedModel to avoid re-loading weights
        self._embed_model = model if model is not None else None

        if method in ("dail", "question") and len(examples) > 0:
            self._build_index()

    # ── Index building ────────────────────────────────────────────────────────

    def _get_embed_model(self):
        """Lazy-load ArcticEmbedModel for question encoding."""
        if self._embed_model is None:
            if self._model_dir is None:
                raise RuntimeError("FewShotSelector: model_dir is required for encoding")
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from embed.arctic_embed import ArcticEmbedModel
            self._embed_model = ArcticEmbedModel(model_dir=self._model_dir, device=self._device)
        return self._embed_model

    def _cache_path(self) -> Optional[Path]:
        """Return a .npy cache path based on model_dir, method, and example count."""
        if self._model_dir is None:
            return None
        name = f"fewshot_embs_{self.method}_{len(self.examples)}.npy"
        return Path(self._model_dir) / name

    def _build_index(self):
        """Encode all training examples once at init time (cached to disk)."""
        cache = self._cache_path()
        if cache is not None and cache.exists():
            self._embs = np.load(str(cache))
            print(f"[selector] Loaded cached embeddings ({len(self.examples)} examples) from {cache.name}")
            return

        embed = self._get_embed_model()
        texts = (
            [ex.masked_question or ex.question for ex in self.examples]
            if self.method == "dail"
            else [ex.question for ex in self.examples]
        )
        print(f"[selector] Encoding {len(texts)} training examples (runs once, then cached) ...")
        self._embs = embed.encode_queries(texts, batch_size=64, show_progress_bar=True).astype(np.float32)

        if cache is not None:
            np.save(str(cache), self._embs)
            print(f"[selector] Cached embeddings → {cache.name}")

    # ── Selection ─────────────────────────────────────────────────────────────

    def select(
        self,
        question: str,
        db_id: str,
        k: int = 5,
        schema_tokens: Optional[set[str]] = None,
        pre_gen_skeleton: Optional[str] = None,
    ) -> list[ExampleItem]:
        """
        Select K few-shot examples for a target question.

        question         : NL question to translate
        db_id            : target database id (used to optionally exclude same-DB examples)
        k                : number of examples to return
        schema_tokens    : set of table/column names in the target DB (for masking)
        pre_gen_skeleton : pre-generated SQL skeleton for DAIL skeleton re-ranking
        """
        if self.method == "random":
            pool = self.examples if self.cross_domain else [
                e for e in self.examples if e.db_id == db_id
            ]
            return random.sample(pool, min(k, len(pool)))

        # Embed the query question
        embed = self._get_embed_model()
        if self.method == "dail" and schema_tokens:
            query_text = mask_schema_tokens(question, schema_tokens)
        else:
            query_text = question

        q_emb = embed.encode_query(query_text)

        # Cosine similarity with all training embeddings
        scores = self._embs @ q_emb    # (N,)

        # Sort descending
        order = np.argsort(scores)[::-1]

        # Filter and collect candidates
        selected: list[ExampleItem] = []
        for idx in order:
            ex = self.examples[idx]
            # Optionally skip same-DB to avoid data leakage
            if not self.cross_domain and ex.db_id == db_id:
                continue
            # Optional skeleton similarity filter
            if pre_gen_skeleton and self.skeleton_threshold > 0:
                sk_sim = _skeleton_similarity(ex.sql_skeleton, pre_gen_skeleton)
                if sk_sim < self.skeleton_threshold:
                    continue
            selected.append(ex)
            if len(selected) >= k:
                break

        return selected

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        train_json_paths: list[Path],
        method: str = "dail",
        model=None,               # pre-built ArcticEmbedModel instance
        model_dir: Optional[Path] = None,
        device: str = "mps",
        cross_domain: bool = True,
        schema_tokens_per_db: Optional[dict[str, set[str]]] = None,
        skeleton_threshold: float = 0.0,
        # legacy compat
        minilm_dir: Optional[Path] = None,
        profile: str = "MP-Balanced",
    ) -> "FewShotSelector":
        """
        Build a FewShotSelector from one or more Spider train JSON files.

        schema_tokens_per_db: optional dict[db_id → set of table/col names]
        """
        examples: list[ExampleItem] = []
        for path in train_json_paths:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for item in data:
                db  = item["db_id"]
                q   = item["question"]
                sql = item.get("query", item.get("sql", ""))
                schema_toks = (schema_tokens_per_db or {}).get(db, set())
                masked_q = mask_schema_tokens(q, schema_toks) if method == "dail" else q
                skeleton = extract_sql_skeleton(sql)
                examples.append(ExampleItem(
                    db_id=db,
                    question=q,
                    sql=sql,
                    masked_question=masked_q,
                    sql_skeleton=skeleton,
                ))

        return cls(
            examples=examples,
            method=method,
            model=model,
            model_dir=model_dir or minilm_dir,
            device=device,
            cross_domain=cross_domain,
            skeleton_threshold=skeleton_threshold,
        )

    def close(self):
        self._embed_model = None
