#!/usr/bin/env python3
"""
bench_schema_linking.py — Mixed-Precision Quantization of Schema Linking Encoders

Spider 1.0 dev set (1034 queries).  Measures impact of mixed-precision
quantization on the arctic-embed-m schema linking encoder.

Metrics per profile
  Quality  : R@k (strict), SoftR@k, MRR
  Latency  : mean / p50 / p95 / p99  ms/query  (per-query encode+retrieve)
  Size     : params (M), peak RAM (MB), disk weight files (MB)
  Delta    : signed difference vs FP32 baseline for every metric
  Edge     : device-class compatibility matrix

Optional end-to-end EX measurement  (--ex-subset N):
  Runs the full Text2SQL pipeline on N dev questions for each profile
  and reports Execution Accuracy so the quantization impact on the
  downstream task is directly visible.

Usage:
  python3 bench_schema_linking.py                     # schema-linking only
  python3 bench_schema_linking.py --device cuda
  python3 bench_schema_linking.py --profiles FP32 INT8
  python3 bench_schema_linking.py --ex-subset 200     # + EX measurement
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config import (
    DEV_JSON, TABLES_JSON, ARCTIC_MODEL_DIR, VEC_INDEX_DIR, EMBED_DEVICE,
    DATABASE_DIR,
)

ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Profile definitions ───────────────────────────────────────────────────
# (display_name, torch_dtype_to_cast, int8_blocks)
#   int8_blocks = None        → quantize ALL Linear layers (full dynamic INT8)
#   int8_blocks = set()       → no INT8 (FP32 / FP16 / BF16 cast only)
#   int8_blocks = {i, j, …}   → quantize only those BertLayer blocks (0–11)
#
# Mixed-precision blocks chosen by sensitivity analysis (bench_layer_sensitivity.py):
#   Block 5 → +0.00484 ΔR@5 (strongest)  |  Blocks 8,9 → +0.00097
#   Blocks 2,10 → +0.00193               |  Blocks 0,1,4,6,7,11 → 0.0 (neutral)
#   All 12 blocks non-negative ΔR@5 → uniform INT8 is greedy-optimal
ALL_PROFILES: list[tuple[str, "Optional[torch.dtype]", "Optional[set[int]]"]] = [
    ("FP32",            torch.float32,  set()),              # baseline
    ("FP16",            torch.float16,  set()),              # float16 cast
    ("BF16",            torch.bfloat16, set()),              # bfloat16 cast
    ("MP-Conservative", torch.float32,  {8, 9}),            # 2 blocks INT8
    ("MP-Balanced",     torch.float32,  {2,3,5,8,9,10}),    # 6 blocks INT8 (positive-gain)
    ("MP-Aggressive",   torch.float32,  {0,1,2,3,4,5,6,7,8,9}),  # 10 blocks INT8
    ("INT8",            torch.float32,  None),               # all Linear → full dynamic INT8
]
PROFILE_NAMES = [p[0] for p in ALL_PROFILES]

# ── Edge-device compatibility (static, based on peak RAM requirements) ────────────
# arctic-embed-m: 110M params  → FP32≍440MB, FP16/BF16≍220MB, INT8≍110MB
# Columns: (device_label, available_ram_gb)
EDGE_DEVICES: list[tuple[str, float]] = [
    ("Pi Zero 2W",   0.35),   # 512MB total, ~350MB free after OS
    ("Jetson Nano",  1.80),   # 4GB but CUDA context ~2.2GB overhead → ~1.8GB CPU free
    ("Pi4 4G",       2.50),   # 4GB, ~2.5GB free after OS + headroom
    ("iPhone 14",    2.00),   # 6GB, iOS per-app RAM ceiling ~2GB
    ("Laptop 8G",    5.00),
    ("M-series",    16.00),
    ("RTX 3090",    24.00),
]
# Total RAM requirement per profile (GB): model params + PyTorch runtime (~350MB) + activations
# arctic-embed-m (110M params): FP32≈418MB, FP16/BF16≈209MB, INT8≈91MB
# Mixed-precision: interpolated by INT8-block fraction (each block ~30MB, 75% reduction)
# Runtime + activation overhead: +350-400MB depending on batch size
PROFILE_RAM_GB: dict[str, float] = {
    "FP32":            0.88,   # 418MB model + ~460MB overhead
    "FP16":            0.62,   # 209MB model + ~410MB overhead
    "BF16":            0.62,
    "MP-Conservative": 0.74,   # ~364MB model (2/12 blocks INT8)
    "MP-Balanced":     0.62,   # ~255MB model (6/12 blocks INT8)
    "MP-Aggressive":   0.52,   # ~147MB model (10/12 blocks INT8)
    "INT8":            0.46,   # ~91MB  model + ~370MB overhead (all Linear INT8)
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_dev_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_table_map(path: Path) -> dict[str, list[str]]:
    """db_id → list of lowercase original table names."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        e["db_id"]: [t.lower() for t in e["table_names_original"]]
        for e in data
    }


def extract_gold_tables(sql: str, db_tables: list[str]) -> frozenset[str]:
    """Return frozenset of table names (lowercase) used in the gold SQL."""
    sql_low = sql.lower()
    return frozenset(
        t for t in db_tables
        if re.search(r"\b" + re.escape(t) + r"\b", sql_low)
    )


def load_all_index_texts(
    db_ids: set[str], index_dir: Path
) -> dict[str, tuple[list[str], list[str]]]:
    """db_id → (table_names_lower, schema_texts) from pre-built SQLite indexes."""
    result: dict[str, tuple[list[str], list[str]]] = {}
    missing = []
    for db_id in sorted(db_ids):
        db_path = index_dir / f"{db_id}.db"
        if not db_path.exists():
            missing.append(db_id); continue
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT table_name, retrieval_text FROM schema_index ORDER BY rowid"
        ).fetchall()
        conn.close()
        result[db_id] = ([r[0].lower() for r in rows], [r[1] for r in rows])
    if missing:
        print(f"  [WARN] No index for {len(missing)} db(s): {missing[:5]}"
              f"{'…' if len(missing) > 5 else ''}")
    return result


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(
    model_dir: Path,
    device: str,
    dtype: Optional[torch.dtype],
    int8_blocks: Optional[set],
) -> SentenceTransformer:
    """
    Load arctic-embed-m and apply quantization.

    int8_blocks:
      None        → quantize_dynamic on ALL Linear layers (full INT8)
      set()       → no INT8; optionally cast to dtype (FP16/BF16)
      {i, j, …}  → quantize_dynamic only on the listed BertLayer blocks (0–11)
    """
    is_int8 = (int8_blocks is None) or (len(int8_blocks) > 0)
    eff_device = "cpu" if is_int8 else device
    model = SentenceTransformer(str(model_dir), device=eff_device)

    if not is_int8 and dtype not in (None, torch.float32):
        try:
            model = model.to(dtype)
        except Exception as exc:
            print(f"  [WARN] {dtype} cast failed ({exc}); staying FP32")

    if is_int8:
        import platform
        engine = "qnnpack" if platform.machine() == "arm64" else "fbgemm"
        torch.backends.quantized.engine = engine

        if int8_blocks is None:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            try:
                encoder = model[0].auto_model.encoder
                for i, layer in enumerate(encoder.layer):
                    if i in int8_blocks:
                        encoder.layer[i] = torch.quantization.quantize_dynamic(
                            layer, {torch.nn.Linear}, dtype=torch.qint8
                        )
            except AttributeError:
                print("  [WARN] Cannot access encoder.layer; falling back to full INT8")
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
    return model


def measure_model_size(model: SentenceTransformer, model_dir: Path) -> dict:
    """
    Returns:
      params_M  : total parameters in millions
      ram_mb    : estimated peak RAM = param bytes (all dtypes summed)
      disk_mb   : sum of .bin/.safetensors/.pt weight files on disk
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes  = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    disk_bytes = sum(
        f.stat().st_size
        for pat in ("*.bin", "*.safetensors", "*.pt")
        for f in model_dir.glob(pat)
    )
    return {
        "params_M": total_params / 1e6,
        "ram_mb":   total_bytes  / (1024 ** 2),
        "disk_mb":  disk_bytes   / (1024 ** 2),
    }


def _encode_single(model: SentenceTransformer, text: str, is_query: bool) -> np.ndarray:
    """Encode a single text; used for per-query latency measurement."""
    inp = (ARCTIC_QUERY_PREFIX + text) if is_query else text
    with torch.no_grad():
        emb = model.encode(
            [inp], normalize_embeddings=True, batch_size=1, show_progress_bar=False
        )
    return np.asarray(emb, dtype=np.float32)[0]


def _encode_batch(
    model: SentenceTransformer,
    texts: list[str],
    is_query: bool,
    batch_size: int = 128,
) -> np.ndarray:
    if is_query:
        texts = [ARCTIC_QUERY_PREFIX + t for t in texts]
    with torch.no_grad():
        embs = model.encode(
            texts, normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=False,
        )
    return np.asarray(embs, dtype=np.float32)


# ── Core benchmark ─────────────────────────────────────────────────────────────

def run_profile(
    questions: list[dict],
    table_map: dict[str, list[str]],
    index_texts: dict[str, tuple[list[str], list[str]]],
    model: SentenceTransformer,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    n_warmup: int = 5,
) -> dict:
    """
    Full schema-linking benchmark for one quantization profile.

    Latency is measured per-query (encode + cosine search) after n_warmup
    warm-up iterations, giving accurate mean / p50 / p95 / p99 statistics.
    """
    # Pre-encode all schema docs (not timed — same for all profiles)
    doc_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    for db_id, (tnames, texts) in index_texts.items():
        doc_cache[db_id] = (tnames, _encode_batch(model, texts, is_query=False))

    # Warm-up: encode a few queries to prime JIT / GPU pipeline
    warmup_qs = [q["question"] for q in questions[:n_warmup]]
    _encode_batch(model, warmup_qs, is_query=True, batch_size=len(warmup_qs))

    strict: dict[int, list[float]] = {k: [] for k in ks}
    soft:   dict[int, list[float]] = {k: [] for k in ks}
    mrr_buf:    list[float] = []
    per_q_ms:   list[float] = []

    for q in questions:
        db_id = q["db_id"]
        if db_id not in doc_cache:
            continue
        db_tables = table_map.get(db_id, [])
        gold = extract_gold_tables(q["query"], db_tables)
        if not gold:
            continue

        # Per-query timing: single-query encode + cosine search
        t0 = time.perf_counter()
        q_emb  = _encode_single(model, q["question"], is_query=True)
        tnames, d_embs = doc_cache[db_id]
        scores = d_embs @ q_emb
        ranked = [tnames[j] for j in np.argsort(scores)[::-1]]
        per_q_ms.append((time.perf_counter() - t0) * 1000)

        # MRR
        mrr = 0.0
        for rank, t in enumerate(ranked, 1):
            if t in gold:
                mrr = 1.0 / rank; break
        mrr_buf.append(mrr)

        # R@k / SoftR@k
        for k in ks:
            topk = set(ranked[:k])
            strict[k].append(1.0 if gold <= topk else 0.0)
            soft[k].append(len(gold & topk) / len(gold))

    n = len(mrr_buf)
    arr = np.array(per_q_ms) if per_q_ms else np.array([0.0])
    out: dict = {
        "n":      n,
        "MRR":    float(np.mean(mrr_buf)) if mrr_buf else 0.0,
        "ms_mean": float(np.mean(arr)),
        "ms_p50":  float(np.percentile(arr, 50)),
        "ms_p95":  float(np.percentile(arr, 95)),
        "ms_p99":  float(np.percentile(arr, 99)),
    }
    for k in ks:
        out[f"R@{k}"]     = float(np.mean(strict[k])) if strict[k] else 0.0
        out[f"SoftR@{k}"] = float(np.mean(soft[k]))   if soft[k]   else 0.0
    return out


# ── Optional EX measurement ────────────────────────────────────────────────────

def run_ex_subset(
    questions_sample: list[dict],
    model: SentenceTransformer,
    profile_name: str,
    device: str,
) -> float:
    """
    Run the full Text2SQL pipeline on questions_sample using the given
    quantized embedding model.  Returns Execution Accuracy (0–1).

    Requires a running Ollama / LLM backend (reads from config.py).
    """
    from schema.loader import load_tables_json
    from schema.serializer import serialize_schema_code_repr
    from embed.arctic_embed import ArcticEmbedModel
    from embed.spider1_retriever import Spider1Retriever
    from examples.selector import FewShotSelector
    from prompt.builder import PromptBuilder
    from llm import get_client
    from postprocess.sql_cleaner import extract_sql, fix_common_errors
    from config import (
        TABLES_JSON as _TJ, TRAIN_SPIDER_JSON, TRAIN_OTHERS_JSON,
        LLM_BACKEND, LLM_MODEL, LLM_BASE_URL, OPENAI_API_KEY,
        LLM_MAX_TOKENS, FEW_SHOT_K, MASK_QUESTION,
    )

    # Wrap the already-loaded SentenceTransformer in ArcticEmbedModel interface
    class _WrappedEmbed:
        def __init__(self, st_model):
            self._model = st_model
            self.dim = 768
        def encode_queries(self, texts):
            return _encode_batch(self._model, texts, is_query=True)
        def encode_docs(self, texts):
            return _encode_batch(self._model, texts, is_query=False)

    wrapped = _WrappedEmbed(model)

    schemas    = load_tables_json(_TJ)
    retriever  = Spider1Retriever(
        model_dir=ARCTIC_MODEL_DIR, index_dir=VEC_INDEX_DIR, device=device,
        _embed_override=wrapped,
    )
    schema_tokens = {
        db_id: {t.name_original.lower() for t in s.tables}
                | {c.name_original.lower() for t in s.tables for c in t.columns}
        for db_id, s in schemas.items()
    }
    selector = FewShotSelector.from_file(
        train_json_paths=[TRAIN_SPIDER_JSON, TRAIN_OTHERS_JSON],
        method="dail" if MASK_QUESTION else "question",
        model=wrapped,
        device=device,
        cross_domain=True,
        schema_tokens_per_db=schema_tokens,
    )
    prompt_builder = PromptBuilder(repr_type="code")
    llm = get_client(
        backend=LLM_BACKEND, model=LLM_MODEL,
        api_key=OPENAI_API_KEY, base_url=LLM_BASE_URL,
    )

    correct = 0
    for item in questions_sample:
        db_id    = item["db_id"]
        question = item["question"]
        gold_sql = item.get("query", "")
        toks     = schema_tokens.get(db_id, set())

        sel_tables = retriever.retrieve_table_names(question, db_id, k=5)
        examples   = selector.select(
            question=question, db_id=db_id, k=FEW_SHOT_K, schema_tokens=toks
        )
        db_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
        messages = prompt_builder.build_messages(
            question=question,
            schema=schemas[db_id],
            examples=examples,
            selected_tables=sel_tables,
            db_path=db_path if db_path.exists() else None,
        )
        resp = llm.complete_with_retry(messages=messages, temperature=0.0,
                                       max_tokens=LLM_MAX_TOKENS, n=1)
        pred_sql = fix_common_errors(extract_sql(resp.text))

        # Execute both queries and compare results
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                conn.text_factory = lambda b: b.decode(errors="ignore")
                pred_rows = set(map(tuple, conn.execute(pred_sql).fetchall()))
                gold_rows = set(map(tuple, conn.execute(gold_sql).fetchall()))
                conn.close()
                if pred_rows == gold_rows:
                    correct += 1
            except Exception:
                pass

    return correct / len(questions_sample) if questions_sample else 0.0


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _signed(v: float, decimals: int = 4) -> str:
    fmt = f"+.{decimals}f" if v >= 0 else f".{decimals}f"
    return f"{v:{fmt}}"


def _signed_ms(v: float) -> str:
    return f"{v:+.1f}" if v >= 0 else f"{v:.1f}"


def print_box(title: str, width: int = 76) -> None:
    pad = width - 2
    print(f"\n╔{'═'*pad}╗")
    print(f"║  {title:<{pad-2}}║")
    print(f"╚{'═'*pad}╝")


def print_section(title: str, width: int = 76) -> None:
    print(f"\n  ── {title} {'─'*(width - len(title) - 6)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-Precision Quantization of Schema Linking Encoders"
    )
    parser.add_argument("--device",     default=EMBED_DEVICE,
                        help="Torch device: mps | cpu | cuda")
    parser.add_argument("--profiles",   nargs="+", default=PROFILE_NAMES,
                        choices=PROFILE_NAMES,
                        help="Profiles to evaluate (default: all)")
    parser.add_argument("--ks",         nargs="+", type=int, default=[1, 3, 5, 10],
                        help="k values for R@k")
    parser.add_argument("--ex-subset",  type=int, default=0, metavar="N",
                        help="Also measure EX on N sampled questions per profile "
                             "(requires running LLM; 0 = skip)")
    parser.add_argument("--warmup",     type=int, default=5,
                        help="Warm-up queries before timing (default: 5)")
    args = parser.parse_args()

    device  = args.device
    ks      = tuple(args.ks)
    chosen  = set(args.profiles)
    ex_n    = args.ex_subset
    warmup  = args.warmup

    print_box(
        f"Mixed-Precision Quantization  │  arctic-embed-m  │  device={device}"
    )
    print(f"  Dataset : Spider 1.0 dev  │  {DEV_JSON}")
    print(f"  Model   : {ARCTIC_MODEL_DIR}")

    questions   = load_dev_questions(DEV_JSON)
    table_map   = load_table_map(TABLES_JSON)
    all_db_ids  = {q["db_id"] for q in questions}
    index_texts = load_all_index_texts(all_db_ids, VEC_INDEX_DIR)

    n_valid = sum(
        1 for q in questions
        if q["db_id"] in index_texts
        and extract_gold_tables(q["query"], table_map.get(q["db_id"], []))
    )
    print(f"  Queries : {len(questions)} total │ {len(index_texts)} DBs "
          f"│ {n_valid} with gold tables")
    if ex_n:
        print(f"  EX mode : full pipeline on {ex_n} sampled questions per profile")

    # ── Per-profile evaluation ─────────────────────────────────────────────────
    results:   dict[str, dict] = {}
    ex_sample: list[dict] = []
    if ex_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(questions), size=min(ex_n, len(questions)), replace=False)
        ex_sample = [questions[i] for i in sorted(idx)]

    for name, dtype, int8_blocks in ALL_PROFILES:
        if name not in chosen:
            continue
        is_int8 = (int8_blocks is None) or (len(int8_blocks) > 0)
        eff_dev = "cpu" if is_int8 else device
        n_int8  = "all" if int8_blocks is None else len(int8_blocks)
        tag     = f" [INT8 blocks={n_int8}]" if is_int8 else ""
        print(f"\n  ┌─ {name}{tag}  device={eff_dev} {'─'*40}")

        t0 = time.perf_counter()
        try:
            model = load_model(ARCTIC_MODEL_DIR, device, dtype, int8_blocks)
        except Exception as exc:
            print(f"  │  SKIP: {exc}"); continue
        load_s = time.perf_counter() - t0

        size   = measure_model_size(model, ARCTIC_MODEL_DIR)
        metrics = run_profile(questions, table_map, index_texts, model, ks, warmup)
        metrics.update(size)
        metrics["load_s"] = load_s

        if ex_n and ex_sample:
            print(f"  │  Running EX on {len(ex_sample)} questions …")
            metrics["EX"] = run_ex_subset(ex_sample, model, name, device)

        results[name] = metrics

        r5 = metrics.get("R@5", metrics.get(f"R@{ks[-1]}", 0))
        print(f"  │  R@5={r5:.4f}  MRR={metrics['MRR']:.4f}  "
              f"ms_mean={metrics['ms_mean']:.1f}  "
              f"RAM={metrics['ram_mb']:.0f}MB  "
              f"params={metrics['params_M']:.1f}M")
        print(f"  └{'─'*54}")

        del model
        if   device == "mps":  torch.mps.empty_cache()
        elif device == "cuda": torch.cuda.empty_cache()

    if not results:
        print("\nNo profiles completed."); return

    baseline = results.get("FP32", next(iter(results.values())))

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 1 — Quality + Latency + Size
    # ══════════════════════════════════════════════════════════════════════════
    print_section("TABLE 1 · Quality  ×  Latency  ×  Size")

    hdr_q  = "  ".join(f"{'R@'+str(k):>7}" for k in ks)
    hdr_ex = "  EX(%)" if ex_n else ""
    W = 5
    sep = "─" * 95
    print(f"\n  {'Profile':<{W}}│ {hdr_q}  {'MRR':>7}  {'SoftR@5':>8}{hdr_ex} "
          f"│ {'mean':>6} {'p50':>6} {'p95':>6} {'p99':>6}ms "
          f"│ {'RAM':>6} {'Disk':>6}MB  {'Params':>8}M")
    print(f"  {'─'*W}┼{'─'*57}{'─'*8 if ex_n else ''}┼{'─'*29}┼{'─'*24}")
    for pname, m in results.items():
        qs   = "  ".join(f"{m[f'R@{k}']:>7.4f}" for k in ks)
        sr5  = m.get("SoftR@5", m.get(f"SoftR@{ks[-1]}", 0))
        ex_s = f"  {m['EX']*100:>5.1f}%" if ex_n and "EX" in m else ("  —    " if ex_n else "")
        print(
            f"  {pname:<{W}}│ {qs}  {m['MRR']:>7.4f}  {sr5:>8.4f}{ex_s} "
            f"│ {m['ms_mean']:>6.1f} {m['ms_p50']:>6.1f} "
            f"{m['ms_p95']:>6.1f} {m['ms_p99']:>6.1f}ms "
            f"│ {m['ram_mb']:>6.0f} {m['disk_mb']:>6.0f}MB  {m['params_M']:>8.1f}M"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 2 — Delta vs FP32 baseline
    # ══════════════════════════════════════════════════════════════════════════
    print_section("TABLE 2 · Delta vs FP32 baseline (signed)")

    non_base = {k: v for k, v in results.items() if k != "FP32"}
    if not non_base:
        print("  (only one profile — nothing to compare)")
    else:
        hdr_d = "  ".join(f"{'ΔR@'+str(k):>8}" for k in ks)
        hdr_dex = "  ΔEX%" if ex_n else ""
        print(f"\n  {'Profile':<{W}}│ {hdr_d}  {'ΔMRR':>8}  {'ΔSoftR@5':>9}{hdr_dex} "
              f"│ {'Δmean':>7} {'Δp95':>7}ms │ {'ΔRAM':>7} {'ΔDisk':>7}MB")
        print(f"  {'─'*W}┼{'─'*55}{'─'*8 if ex_n else ''}┼{'─'*18}┼{'─'*18}")
        for pname, m in non_base.items():
            dqs  = "  ".join(
                _signed(m[f"R@{k}"] - baseline[f"R@{k}"], 4) for k in ks
            )
            dsr5 = baseline.get("SoftR@5", 0)
            dex_s = ""
            if ex_n and "EX" in m and "EX" in baseline:
                dex_s = f"  {_signed((m['EX']-baseline['EX'])*100, 1)}%"
            elif ex_n:
                dex_s = "  —"
            print(
                f"  {pname:<{W}}│ {dqs}  "
                f"{_signed(m['MRR']-baseline['MRR'], 4)}  "
                f"{_signed(m.get('SoftR@5',0)-dsr5, 4):>9}{dex_s} "
                f"│ {_signed_ms(m['ms_mean']-baseline['ms_mean']):>7} "
                f"{_signed_ms(m['ms_p95']-baseline['ms_p95']):>7}ms "
                f"│ {_signed_ms(m['ram_mb']-baseline['ram_mb']):>7} "
                f"{_signed_ms(m['disk_mb']-baseline['disk_mb']):>7}MB"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 3 — Edge Device Compatibility
    # ══════════════════════════════════════════════════════════════════════════
    print_section("TABLE 3 · Edge Device Compatibility  (✓ fits  ✗ too large)")

    dev_labels = [d for d, _ in EDGE_DEVICES]
    dev_ram    = {d: r for d, r in EDGE_DEVICES}
    col_w = max(len(d) for d in dev_labels) + 2

    header = "  ".join(f"{d:^{col_w}}" for d in dev_labels)
    print(f"\n  {'Profile':<{W}}│ {header}")
    print(f"  {'─'*W}┼{'─'*(col_w*len(dev_labels) + 2*len(dev_labels))}")
    for pname in results:
        req_gb = PROFILE_RAM_GB.get(pname, 0.44)
        cells  = []
        for d, avail_gb in EDGE_DEVICES:
            ok = req_gb <= avail_gb
            cells.append(f"{'✓' if ok else '✗':^{col_w}}")
        print(f"  {pname:<{W}}│ {'  '.join(cells)}")

    print(f"\n  RAM requirements (estimated):  "
          + "  ".join(f"{p}={PROFILE_RAM_GB[p]*1024:.0f}MB"
                      for p in results if p in PROFILE_RAM_GB))

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print_section("SUMMARY")
    best_r5_name = max(results, key=lambda k: results[k].get("R@5", 0))
    best_ms_name = min(results, key=lambda k: results[k]["ms_mean"])
    best_ram_name = min(results, key=lambda k: results[k]["ram_mb"])
    print(f"\n  Best R@5    : {best_r5_name}  ({results[best_r5_name].get('R@5',0):.4f})")
    print(f"  Fastest     : {best_ms_name}  ({results[best_ms_name]['ms_mean']:.1f} ms/q mean)")
    print(f"  Smallest RAM: {best_ram_name}  ({results[best_ram_name]['ram_mb']:.0f} MB)")
    if ex_n:
        best_ex = max((k for k in results if "EX" in results[k]),
                      key=lambda k: results[k]["EX"], default=None)
        if best_ex:
            print(f"  Best EX     : {best_ex}  ({results[best_ex]['EX']*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
