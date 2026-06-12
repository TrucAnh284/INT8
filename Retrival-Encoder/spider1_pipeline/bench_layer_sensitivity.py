#!/usr/bin/env python3
"""
bench_layer_sensitivity.py — Per-Layer Sensitivity Analysis for Mixed-Precision Quantization
of the arctic-embed-m Schema Linking Encoder (Spider 1.0 dev set, 1034 queries).

Analysis levels
  1. Block-level   : quantize one BertLayer at a time (layers 0–11) → 12 experiments
  2. Component-type: quantize all layers of each component type    →  7 experiments
     (attn_query, attn_key, attn_value, attn_out, ffn_gate, ffn_out, pooler)
  3. Fine-grained  : per-layer within each block (optional, --fine)   → 72 experiments

From the sensitivity profile the script:
  • Designs an optimal mixed-precision assignment (greedy INT8 first,
    then FP16 for sensitive layers, FP32 only where critical)
  • Benchmarks the optimal assignment vs uniform FP32 / FP16 / INT8

Usage:
  python3 bench_layer_sensitivity.py
  python3 bench_layer_sensitivity.py --fine          # fine-grained per-sublayer
  python3 bench_layer_sensitivity.py --device cpu
  python3 bench_layer_sensitivity.py --budget 0.001  # max allowed ΔR@5
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config import DEV_JSON, TABLES_JSON, ARCTIC_MODEL_DIR, VEC_INDEX_DIR, EMBED_DEVICE

ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Data helpers (shared with bench_schema_linking) ───────────────────────────

def load_dev_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_table_map(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {e["db_id"]: [t.lower() for t in e["table_names_original"]] for e in data}


def extract_gold_tables(sql: str, db_tables: list[str]) -> frozenset[str]:
    sql_low = sql.lower()
    return frozenset(t for t in db_tables
                     if re.search(r"\b" + re.escape(t) + r"\b", sql_low))


def load_index_texts(db_ids: set[str], index_dir: Path) -> dict:
    result = {}
    for db_id in sorted(db_ids):
        p = index_dir / f"{db_id}.db"
        if not p.exists():
            continue
        conn = sqlite3.connect(str(p))
        rows = conn.execute(
            "SELECT table_name, retrieval_text FROM schema_index ORDER BY rowid"
        ).fetchall()
        conn.close()
        result[db_id] = ([r[0].lower() for r in rows], [r[1] for r in rows])
    return result


# ── Encoding helpers ───────────────────────────────────────────────────────────

def encode_batch(model: SentenceTransformer, texts: list[str],
                 is_query: bool, batch_size: int = 128) -> np.ndarray:
    if is_query:
        texts = [ARCTIC_QUERY_PREFIX + t for t in texts]
    with torch.no_grad():
        embs = model.encode(texts, normalize_embeddings=True,
                            batch_size=batch_size, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)


# ── Quality metric (R@k, MRR) — batch mode, no per-query timing ──────────────

def compute_quality(
    questions: list[dict],
    table_map: dict,
    index_texts: dict,
    model: SentenceTransformer,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Batch-encode everything for speed; returns R@k and MRR."""
    doc_cache = {}
    for db_id, (tnames, texts) in index_texts.items():
        doc_cache[db_id] = (tnames, encode_batch(model, texts, is_query=False))

    q_embs = encode_batch(model, [q["question"] for q in questions], is_query=True)

    strict = {k: [] for k in ks}
    soft   = {k: [] for k in ks}
    mrr_buf: list[float] = []

    for i, q in enumerate(questions):
        db_id = q["db_id"]
        if db_id not in doc_cache:
            continue
        gold = extract_gold_tables(q["query"], table_map.get(db_id, []))
        if not gold:
            continue
        tnames, d_embs = doc_cache[db_id]
        ranked = [tnames[j] for j in np.argsort(d_embs @ q_embs[i])[::-1]]
        mrr = 0.0
        for rank, t in enumerate(ranked, 1):
            if t in gold:
                mrr = 1.0 / rank; break
        mrr_buf.append(mrr)
        for k in ks:
            topk = set(ranked[:k])
            strict[k].append(float(gold <= topk))
            soft[k].append(len(gold & topk) / len(gold))

    out = {"MRR": float(np.mean(mrr_buf)) if mrr_buf else 0.0}
    for k in ks:
        out[f"R@{k}"]     = float(np.mean(strict[k])) if strict[k] else 0.0
        out[f"SoftR@{k}"] = float(np.mean(soft[k]))   if soft[k]   else 0.0
    return out


# ── Quantization helpers ───────────────────────────────────────────────────────

def _get_quant_engine() -> str:
    return "qnnpack" if platform.machine() == "arm64" else "fbgemm"


def quantize_module_inplace(module: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization to all Linear layers within module.
    Uses inplace=True so the module is modified in-place and parent references
    to it remain valid without needing to update the parent explicitly.
    """
    torch.backends.quantized.engine = _get_quant_engine()
    return torch.quantization.quantize_dynamic(
        module, {nn.Linear}, dtype=torch.qint8, inplace=True
    )


def get_bert_model(st_model: SentenceTransformer) -> nn.Module:
    """Extract the underlying BertModel from a SentenceTransformer wrapper."""
    return st_model[0].auto_model


def count_linear_params(module: nn.Module) -> int:
    return sum(p.numel() for m in module.modules()
               if isinstance(m, nn.Linear) for p in m.parameters())


# ── Sensitivity experiment ─────────────────────────────────────────────────────

def run_sensitivity_experiment(
    fp32_model: SentenceTransformer,
    target_path: str,          # dotted path into BertModel, e.g. "encoder.layer.3"
    questions: list[dict],
    table_map: dict,
    index_texts: dict,
    ks: tuple[int, ...],
    device: str,
) -> dict:
    """
    Clone the FP32 model, quantize only the submodule at target_path to INT8,
    then measure quality.  Returns quality dict + size savings info.
    """
    import warnings
    # INT8 quantization requires CPU; move first, then quantize
    m = copy.deepcopy(fp32_model).to("cpu")
    bert = get_bert_model(m)

    # Navigate to the target submodule
    node = bert
    parts = target_path.split(".")
    for part in parts:
        if part.isdigit():
            node = node[int(part)]
        else:
            node = getattr(node, part)

    # Count FP32 Linear params BEFORE quantizing (after quantization they become
    # DynamicQuantizedLinear which is not nn.Linear, so count returns 0)
    params_quantized = count_linear_params(node)

    # Quantize in-place (inplace=True modifies node directly inside m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_module_inplace(node)

    quality = compute_quality(questions, table_map, index_texts, m, ks)
    del m
    return {"quality": quality, "params_quantized_M": params_quantized / 1e6}


# ── Mixed-precision assignment design ─────────────────────────────────────────

def design_mixed_precision(
    block_sensitivities: list[dict],  # [{block: int, delta_r5: float}, ...]
    baseline_r5: float,
    budget: float = 0.002,
    fp16_budget: float = 0.0005,
) -> dict:
    """
    Greedy mixed-precision assignment for transformer blocks.

    Strategy:
      - Sort blocks by |ΔR@5| ascending (least sensitive first)
      - Assign INT8 greedily while cumulative ΔR@5 stays within `budget`
      - Assign FP16 to blocks where ΔR@5 < fp16_budget (very low sensitivity)
        but can't fit INT8 budget
      - Keep remaining blocks at FP32

    Returns dict: block_id → precision
    """
    # Positive ΔR@5 = INT8 improves quality → assign for free (no budget cost)
    # Negative ΔR@5 = INT8 degrades quality → charge |ΔR@5| to budget
    assignment = {}
    cumulative_degradation = 0.0

    free   = [e for e in block_sensitivities if e["delta_r5"] >= 0]
    costly = sorted([e for e in block_sensitivities if e["delta_r5"] < 0],
                    key=lambda x: x["delta_r5"], reverse=True)  # least bad first

    for entry in free:
        assignment[entry["block"]] = "INT8"

    for entry in costly:
        bid  = entry["block"]
        cost = abs(entry["delta_r5"])
        if cumulative_degradation + cost <= budget:
            assignment[bid] = "INT8"
            cumulative_degradation += cost
        elif cost <= fp16_budget:
            assignment[bid] = "FP16"
        else:
            assignment[bid] = "FP32"

    return assignment


def build_mixed_precision_model(
    fp32_model: SentenceTransformer,
    block_assignment: dict,  # block_id → "INT8" | "FP16" | "FP32"
    device: str,
) -> SentenceTransformer:
    """Clone FP32 model, move to CPU, apply per-block precision assignments."""
    import warnings
    # Must be on CPU for INT8 quantization; FP16 layers also run on CPU in mixed mode
    m    = copy.deepcopy(fp32_model).to("cpu")
    bert = get_bert_model(m)

    for bid, precision in block_assignment.items():
        layer = bert.encoder.layer[bid]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if precision == "INT8":
                quantize_module_inplace(layer)   # inplace=True modifies layer inside m
            elif precision == "FP16":
                layer.to(torch.float16)          # .to() is always in-place on the module
            # FP32: leave as-is
    return m


# ── Formatting helpers ─────────────────────────────────────────────────────────

def bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(abs(value) / max(abs(max_val), 1e-9) * width)
    return "█" * filled + "░" * (width - filled)


def print_section(title: str, width: int = 78) -> None:
    print(f"\n  ── {title} {'─' * max(0, width - len(title) - 6)}")


def print_box(title: str, width: int = 78) -> None:
    pad = width - 2
    print(f"\n╔{'═' * pad}╗")
    print(f"║  {title:<{pad - 2}}║")
    print(f"╚{'═' * pad}╝")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer sensitivity analysis for mixed-precision quantization"
    )
    parser.add_argument("--device",  default=EMBED_DEVICE)
    parser.add_argument("--ks",      nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--budget",  type=float, default=0.002,
                        help="Max cumulative ΔR@5 budget for INT8 assignment")
    parser.add_argument("--fine",    action="store_true",
                        help="Also run fine-grained per-sublayer experiments (slower)")
    args = parser.parse_args()

    ks     = tuple(args.ks)
    device = args.device
    budget = args.budget

    print_box(f"Layer Sensitivity Analysis  │  arctic-embed-m  │  device={device}")
    print(f"  Model  : {ARCTIC_MODEL_DIR}")
    print(f"  Budget : ΔR@5 ≤ {budget} for INT8  │  ks={list(ks)}")

    # Load data
    questions  = load_dev_questions(DEV_JSON)
    table_map  = load_table_map(TABLES_JSON)
    index_texts = load_index_texts({q["db_id"] for q in questions}, VEC_INDEX_DIR)
    print(f"  Data   : {len(questions)} queries  │  {len(index_texts)} DBs")

    # Load FP32 baseline
    print("\n  Loading FP32 baseline …")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp32_model = SentenceTransformer(str(ARCTIC_MODEL_DIR), device=device)

    t0       = time.perf_counter()
    baseline = compute_quality(questions, table_map, index_texts, fp32_model, ks)
    enc_s    = time.perf_counter() - t0

    print(f"  Baseline: R@5={baseline['R@5']:.4f}  MRR={baseline['MRR']:.4f}  "
          f"({enc_s:.1f}s)")

    bert      = get_bert_model(fp32_model)
    n_blocks  = len(bert.encoder.layer)
    total_params = sum(p.numel() for p in fp32_model.parameters()) / 1e6

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1 — Block-level sensitivity (quantize one BertLayer at a time)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("EXPERIMENT 1 · Block-level Sensitivity (INT8 one block at a time)")
    print(f"\n  {'Block':<7} {'R@5':>7} {'ΔR@5':>8} {'ΔMRR':>8} {'Params(M)':>10}  Sensitivity bar")
    print(f"  {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*10}  {'─'*22}")

    block_results: list[dict] = []

    for i in range(n_blocks):
        path = f"encoder.layer.{i}"
        t0   = time.perf_counter()
        res  = run_sensitivity_experiment(
            fp32_model, path, questions, table_map, index_texts, ks, device
        )
        elapsed = time.perf_counter() - t0

        q       = res["quality"]
        dr5     = q["R@5"]  - baseline["R@5"]
        dmrr    = q["MRR"] - baseline["MRR"]
        block_results.append({"block": i, "r5": q["R@5"], "delta_r5": dr5,
                               "delta_mrr": dmrr, "params_M": res["params_quantized_M"]})

        sign   = "+" if dr5 >= 0 else ""
        b      = bar(dr5, -0.01)
        print(f"  Block {i:2d}  {q['R@5']:>7.4f}  {sign}{dr5:>7.4f}  "
              f"{sign}{dmrr:>7.4f}  {res['params_quantized_M']:>9.1f}M  {b}  ({elapsed:.1f}s)")

    # Rank by sensitivity
    ranked_blocks = sorted(block_results, key=lambda x: x["delta_r5"])
    max_sens      = max(abs(r["delta_r5"]) for r in block_results) or 1e-9

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2 — Component-type sensitivity
    # ══════════════════════════════════════════════════════════════════════════
    print_section("EXPERIMENT 2 · Component-Type Sensitivity (INT8 one type across all blocks)")

    component_specs = {
        "attn_query":  [f"encoder.layer.{i}.attention.self.query"  for i in range(n_blocks)],
        "attn_key":    [f"encoder.layer.{i}.attention.self.key"    for i in range(n_blocks)],
        "attn_value":  [f"encoder.layer.{i}.attention.self.value"  for i in range(n_blocks)],
        "attn_out":    [f"encoder.layer.{i}.attention.output.dense" for i in range(n_blocks)],
        "ffn_gate":    [f"encoder.layer.{i}.intermediate.dense"    for i in range(n_blocks)],
        "ffn_out":     [f"encoder.layer.{i}.output.dense"          for i in range(n_blocks)],
        "pooler":      ["pooler"],   # quantize whole BertPooler (contains .dense)
    }

    print(f"\n  {'Component':<14} {'R@5':>7} {'ΔR@5':>8} {'ΔMRR':>8} {'Params(M)':>10}  Sensitivity bar")
    print(f"  {'─'*14} {'─'*7} {'─'*8} {'─'*8} {'─'*10}  {'─'*22}")

    component_results: list[dict] = []

    for comp_name, paths in component_specs.items():
        # INT8 requires CPU — move first, then quantize
        m    = copy.deepcopy(fp32_model).to("cpu")
        bert = get_bert_model(m)
        total_q_params = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for path in paths:
                node = bert
                for part in path.split("."):
                    node = node[int(part)] if part.isdigit() else getattr(node, part)
                total_q_params += count_linear_params(node)  # count BEFORE quantizing
                quantize_module_inplace(node)                 # inplace=True modifies node in m
        t0  = time.perf_counter()
        q   = compute_quality(questions, table_map, index_texts, m, ks)
        elapsed = time.perf_counter() - t0
        del m

        dr5  = q["R@5"]  - baseline["R@5"]
        dmrr = q["MRR"] - baseline["MRR"]
        component_results.append({"component": comp_name, "r5": q["R@5"],
                                   "delta_r5": dr5, "delta_mrr": dmrr,
                                   "params_M": total_q_params / 1e6})

        sign = "+" if dr5 >= 0 else ""
        b    = bar(dr5, -0.01)
        print(f"  {comp_name:<14} {q['R@5']:>7.4f}  {sign}{dr5:>7.4f}  "
              f"{sign}{dmrr:>7.4f}  {total_q_params/1e6:>9.1f}M  {b}  ({elapsed:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3 — Fine-grained (per-sublayer within each block)
    # ══════════════════════════════════════════════════════════════════════════
    fine_results: list[dict] = []

    if args.fine:
        print_section("EXPERIMENT 3 · Fine-Grained Per-Sublayer Sensitivity")
        sublayer_specs = [
            ("Q",    "attention.self.query"),
            ("K",    "attention.self.key"),
            ("V",    "attention.self.value"),
            ("Aout", "attention.output.dense"),
            ("FFN1", "intermediate.dense"),
            ("FFN2", "output.dense"),
        ]
        print(f"\n  {'Block':<7} {'Sub':<6} {'R@5':>7} {'ΔR@5':>8}  bar")
        print(f"  {'─'*7} {'─'*6} {'─'*7} {'─'*8}  {'─'*20}")

        for i in range(n_blocks):
            for sub_name, sub_path in sublayer_specs:
                path = f"encoder.layer.{i}.{sub_path}"
                res  = run_sensitivity_experiment(
                    fp32_model, path, questions, table_map, index_texts, ks, device
                )
                q   = res["quality"]
                dr5 = q["R@5"] - baseline["R@5"]
                fine_results.append({"block": i, "sub": sub_name,
                                      "r5": q["R@5"], "delta_r5": dr5})
                sign = "+" if dr5 >= 0 else ""
                b    = bar(dr5, -0.005)
                print(f"  Block {i:2d}  {sub_name:<6} {q['R@5']:>7.4f}  {sign}{dr5:>7.4f}  {b}")

    # ══════════════════════════════════════════════════════════════════════════
    # SENSITIVITY HEATMAP (ASCII)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("SENSITIVITY HEATMAP  (block × component, ΔR@5 darker = more sensitive)")

    comp_names = [c["component"] for c in component_results]
    comp_dr5   = {c["component"]: c["delta_r5"] for c in component_results}
    block_dr5  = {r["block"]: r["delta_r5"] for r in block_results}

    # Compact heatmap: show per-block sensitivity as colored bar
    abs_all   = [abs(r["delta_r5"]) for r in block_results if r["delta_r5"] != 0]
    scale     = max(abs_all) if abs_all else 1e-9

    print(f"\n  Block  ΔR@5       Sensitivity           Recommended precision")
    print(f"  {'─'*6} {'─'*9} {'─'*22} {'─'*22}")
    for r in range(n_blocks):
        entry = block_dr5[r]
        filled = int(min(abs(entry) / scale, 1.0) * 20)
        bbar   = "█" * filled + "░" * (20 - filled)
        if   entry > 0:                              rec = "INT8  ← improves quality"
        elif entry == 0:                             rec = "INT8  ← neutral (safe)"
        elif abs(entry) <= budget:                   rec = "INT8  ← marginal loss"
        elif abs(entry) < 0.005:                     rec = "FP16  ← moderate loss"
        else:                                        rec = "FP32  ← sensitive"
        sign = "+" if entry >= 0 else ""
        print(f"  [{r:2d}]   {sign}{entry:.5f}  {bbar}  {rec}")

    print(f"\n  Component sensitivity (quantizing all blocks of that type):")
    for c in sorted(component_results, key=lambda x: x["delta_r5"]):
        sign   = "+" if c["delta_r5"] >= 0 else ""
        filled = int(min(abs(c["delta_r5"]) / scale, 1.0) * 15)
        bbar   = "█" * filled + "░" * (15 - filled)
        print(f"    {c['component']:<14} {sign}{c['delta_r5']:.5f}  {bbar}")

    # ══════════════════════════════════════════════════════════════════════════
    # MIXED-PRECISION DESIGN
    # ══════════════════════════════════════════════════════════════════════════
    print_section(f"MIXED-PRECISION DESIGN  (budget ΔR@5 ≤ {budget})")

    assignment = design_mixed_precision(block_results, baseline["R@5"], budget=budget)

    int8_blocks  = [b for b, p in assignment.items() if p == "INT8"]
    fp16_blocks  = [b for b, p in assignment.items() if p == "FP16"]
    fp32_blocks  = [b for b, p in assignment.items() if p == "FP32"]

    print(f"\n  INT8  blocks  ({len(int8_blocks):2d}): {int8_blocks}")
    print(f"  FP16  blocks  ({len(fp16_blocks):2d}): {fp16_blocks}")
    print(f"  FP32  blocks  ({len(fp32_blocks):2d}): {fp32_blocks}")

    # Estimate RAM savings from mixed assignment
    params_per_block = sum(count_linear_params(bert.encoder.layer[i]) for i in range(n_blocks)) / n_blocks
    mixed_bytes = 0
    for i in range(n_blocks):
        prec  = assignment.get(i, "FP32")
        bpp   = 1 if prec == "INT8" else (2 if prec in ("FP16", "BF16") else 4)
        mixed_bytes += params_per_block * bpp
    mixed_ram_mb = mixed_bytes / (1024 ** 2)

    fp32_block_ram = params_per_block * n_blocks * 4 / (1024 ** 2)
    savings_pct    = (1 - mixed_bytes / (params_per_block * n_blocks * 4)) * 100

    print(f"\n  Encoder block RAM:  FP32={fp32_block_ram:.0f}MB  "
          f"Mixed={mixed_ram_mb:.0f}MB  "
          f"Savings={savings_pct:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # BENCHMARK: Optimal Mixed-Precision vs Uniform Profiles
    # ══════════════════════════════════════════════════════════════════════════
    print_section("BENCHMARK · Optimal Mixed-Precision vs Uniform Profiles")

    print(f"\n  Building mixed-precision model …")
    mixed_model = build_mixed_precision_model(fp32_model, assignment, device)

    t0      = time.perf_counter()
    mixed_q = compute_quality(questions, table_map, index_texts, mixed_model, ks)
    enc_s   = time.perf_counter() - t0
    del mixed_model

    # Count actual RAM of mixed model
    mixed_actual_ram = 0.0
    _tmp = build_mixed_precision_model(fp32_model, assignment, device)
    for p in _tmp.parameters():
        mixed_actual_ram += p.numel() * p.element_size()
    mixed_actual_ram /= (1024 ** 2)
    del _tmp

    # Compare
    profiles_ref = {
        "FP32 (baseline)": {"r5": baseline["R@5"], "mrr": baseline["MRR"],
                             "ram_mb": sum(p.numel() * p.element_size()
                                          for p in fp32_model.parameters()) / (1024 ** 2)},
        "Mixed (optimal)": {"r5": mixed_q["R@5"],  "mrr": mixed_q["MRR"],
                             "ram_mb": mixed_actual_ram},
    }

    # Also run uniform FP16 and INT8 for comparison
    for pname, dtype, q_int8 in [("FP16", torch.float16, False),
                                  ("INT8 (uniform)", torch.float32, True)]:
        # INT8 quantization requires CPU; run FP16 on CPU too for fair latency comparison
        _m = copy.deepcopy(fp32_model).to("cpu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if q_int8:
                torch.backends.quantized.engine = _get_quant_engine()
                _m = torch.quantization.quantize_dynamic(_m, {nn.Linear}, dtype=torch.qint8)
            else:
                _m = _m.to(dtype)
        _q  = compute_quality(questions, table_map, index_texts, _m, ks)
        _ram = sum(p.numel() * p.element_size() for p in _m.parameters()) / (1024 ** 2)
        profiles_ref[pname] = {"r5": _q["R@5"], "mrr": _q["MRR"], "ram_mb": _ram}
        del _m

    base_r5 = profiles_ref["FP32 (baseline)"]["r5"]
    print(f"\n  {'Profile':<20} {'R@5':>8} {'ΔR@5':>8} {'MRR':>8} {'RAM MB':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for pname, vals in profiles_ref.items():
        dr5  = vals["r5"] - base_r5
        sign = "+" if dr5 >= 0 else ""
        mark = " ◄ optimal" if pname == "Mixed (optimal)" else ""
        print(f"  {pname:<20} {vals['r5']:>8.4f}  {sign}{dr5:.4f}  "
              f"{vals['mrr']:>8.4f}  {vals['ram_mb']:>7.0f}MB{mark}")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print_section("SUMMARY")

    most_degrading = min(block_results,     key=lambda x: x["delta_r5"])
    most_improving = max(block_results,     key=lambda x: x["delta_r5"])
    worst_comp     = min(component_results, key=lambda x: x["delta_r5"])
    best_comp      = max(component_results, key=lambda x: x["delta_r5"])
    n_neg          = sum(1 for r in block_results if r["delta_r5"] < 0)

    print(f"\n  Most degrading block  : Block {most_degrading['block']:2d}  "
          f"(ΔR@5={most_degrading['delta_r5']:+.5f})  "
          f"{'→ INT8 safe (no degradation!)' if most_degrading['delta_r5'] >= 0 else '→ keep FP16/FP32'}")
    print(f"  Most improving block  : Block {most_improving['block']:2d}  "
          f"(ΔR@5={most_improving['delta_r5']:+.5f})  → INT8 strongest regularization gain")
    print(f"  Most degrading component: {worst_comp['component']}  "
          f"(ΔR@5={worst_comp['delta_r5']:+.5f})")
    print(f"  Most improving component: {best_comp['component']}  "
          f"(ΔR@5={best_comp['delta_r5']:+.5f})")
    print(f"\n  INT8-safety verdict   : "
          f"{'ALL 12 blocks safe — uniform INT8 is optimal!' if n_neg == 0 else f'{n_neg}/12 blocks degrade — use mixed-precision'}")
    print(f"\n  Mixed-precision result:")
    print(f"    R@5  = {mixed_q['R@5']:.4f}  (ΔR@5 = {mixed_q['R@5']-base_r5:+.5f} vs FP32)")
    print(f"    MRR  = {mixed_q['MRR']:.4f}")
    print(f"    RAM  = {mixed_actual_ram:.0f}MB  (savings={savings_pct:.1f}% vs FP32 encoder)")
    print(f"    INT8 blocks: {len(int8_blocks)}/{n_blocks}  "
          f"FP16: {len(fp16_blocks)}/{n_blocks}  "
          f"FP32: {len(fp32_blocks)}/{n_blocks}")
    print()


if __name__ == "__main__":
    main()
