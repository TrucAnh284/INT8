"""
Latency and memory benchmark: INT8 vs FP32 arctic-embed-m.

Measures:
  - Encode latency (ms/query) for 100 Spider 1.0 dev questions
  - Peak RAM usage (process RSS)
  - Model parameter memory (bytes)

Run:
  python3 bench_latency.py
"""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import ARCTIC_MODEL_DIR, DEV_JSON
from embed.arctic_embed import ArcticEmbedModel

N_WARMUP  = 5
N_MEASURE = 100


def load_questions(n: int) -> list[str]:
    with open(DEV_JSON) as f:
        data = json.load(f)
    return [d["question"] for d in data[:n]]


def measure(model: ArcticEmbedModel, questions: list[str]) -> dict:
    for q in questions[:N_WARMUP]:
        model.encode_queries([q])

    tracemalloc.start()
    t0 = time.perf_counter()
    for q in questions:
        model.encode_queries([q])
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latency_ms = (elapsed / len(questions)) * 1000
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.model.parameters()
    )
    return {
        "latency_ms_per_query": round(latency_ms, 3),
        "peak_ram_mb":          round(peak_bytes / 1024**2, 1),
        "param_ram_mb":         round(param_bytes / 1024**2, 1),
        "n_questions":          len(questions),
    }


def main():
    questions = load_questions(N_MEASURE)
    print(f"Loaded {len(questions)} questions for benchmark\n")

    configs = [
        ("FP32", False),
        ("INT8", True),
    ]
    results = {}
    for label, int8 in configs:
        print(f"── {label} ──────────────────────────")
        model = ArcticEmbedModel(
            model_dir=ARCTIC_MODEL_DIR,
            device="cpu",          # fair comparison: both on CPU
            quantize_int8=int8,
        )
        r = measure(model, questions)
        results[label] = r
        print(f"  Latency  : {r['latency_ms_per_query']:.2f} ms/query")
        print(f"  Param RAM: {r['param_ram_mb']:.0f} MB")
        print(f"  Peak RAM : {r['peak_ram_mb']:.0f} MB\n")
        del model

    fp32 = results["FP32"]
    int8 = results["INT8"]
    speedup  = fp32["latency_ms_per_query"] / int8["latency_ms_per_query"]
    ram_save = 1 - int8["param_ram_mb"] / fp32["param_ram_mb"]
    print("── Summary ──────────────────────────────")
    print(f"  Speedup (INT8/FP32)  : {speedup:.2f}×")
    print(f"  RAM reduction        : {ram_save*100:.1f}%")
    print(f"  FP32 latency         : {fp32['latency_ms_per_query']:.2f} ms/query")
    print(f"  INT8 latency         : {int8['latency_ms_per_query']:.2f} ms/query")

    out = Path("output/bench_latency.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"fp32": fp32, "int8": int8,
                               "speedup": round(speedup, 3),
                               "ram_reduction_pct": round(ram_save*100, 1)},
                              indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
