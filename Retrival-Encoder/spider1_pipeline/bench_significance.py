"""
Statistical significance tests for Spider 1.0 pipeline results.

Tests:
  1. McNemar's test  — pairwise per-question win/loss between two systems
  2. Bootstrap CI    — 95% confidence interval on EX accuracy via resampling

Usage:
  python3 bench_significance.py --a output/results_sc3_k5_2pass.json \
                                 --b output/results_ablation_k5_sc1.json
  python3 bench_significance.py --all   # compare all result files
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


# ── Core stats ────────────────────────────────────────────────────────────────

def bootstrap_ci(scores: list[int], n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) via percentile bootstrap."""
    n = len(scores)
    means = sorted(
        sum(random.choices(scores, k=n)) / n
        for _ in range(n_boot)
    )
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(scores) / n, lo, hi


def mcnemar(a_scores: list[int], b_scores: list[int]) -> tuple[float, float]:
    """
    McNemar's test (mid-p variant) for two paired binary classifiers.
    Returns (chi2_statistic, p_value).
    b01 = A wrong, B correct; b10 = A correct, B wrong.
    """
    b01 = sum(1 for a, b in zip(a_scores, b_scores) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_scores, b_scores) if a == 1 and b == 0)
    n   = b01 + b10
    if n == 0:
        return 0.0, 1.0

    # Mid-p McNemar: chi2 = (|b01-b10| - 1)^2 / (b01+b10)
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    # Chi2 CDF approximation (df=1, Wilson-Hilferty)
    x = chi2
    p = 1 - _chi2_cdf(x, df=1)
    return round(chi2, 4), round(p, 6)


def _chi2_cdf(x: float, df: int) -> float:
    """Regularized incomplete gamma function via series expansion (chi2 CDF)."""
    if x <= 0:
        return 0.0
    k = df / 2
    return _reg_inc_gamma(k, x / 2)


def _reg_inc_gamma(a: float, x: float, iters: int = 200) -> float:
    """Regularized lower incomplete gamma P(a, x) via series."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    log_gamma_a = math.lgamma(a)
    term = math.exp(a * math.log(x) - x - log_gamma_a) / a
    s = term
    for n in range(1, iters):
        term *= x / (a + n)
        s += term
        if abs(term) < 1e-10 * abs(s):
            break
    return min(s, 1.0)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_scores(path: Path) -> tuple[list[str], list[int]]:
    data = json.loads(path.read_text())
    results = data["results"]
    ids    = [str(r.get("id", r.get("instance_id", i))) for i, r in enumerate(results)]
    scores = [int(bool(r.get("exec_match", r.get("score", 0)))) for r in results]
    return ids, scores


def all_result_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("results_*.json"))


# ── Report ────────────────────────────────────────────────────────────────────

def report_pair(name_a: str, scores_a: list[int], name_b: str, scores_b: list[int]):
    assert len(scores_a) == len(scores_b), "Score lists must be same length"
    mean_a, lo_a, hi_a = bootstrap_ci(scores_a)
    mean_b, lo_b, hi_b = bootstrap_ci(scores_b)
    chi2, p = mcnemar(scores_a, scores_b)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    print(f"\n{'─'*60}")
    print(f"  A: {name_a}")
    print(f"     EX = {mean_a:.4f}  95%CI [{lo_a:.4f}, {hi_a:.4f}]")
    print(f"  B: {name_b}")
    print(f"     EX = {mean_b:.4f}  95%CI [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"  McNemar χ²={chi2:.3f}  p={p:.4f}  {sig}")
    print(f"  ΔEX = {mean_b - mean_a:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a",   type=Path, default=None)
    parser.add_argument("--b",   type=Path, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "output")
    args = parser.parse_args()

    random.seed(42)

    if args.all:
        files = all_result_files(args.output_dir)
        if len(files) < 2:
            print("Need at least 2 results_*.json files in output/")
            return
        print(f"Found {len(files)} result files:")
        loaded = []
        for f in files:
            _, scores = load_scores(f)
            acc = sum(scores) / len(scores)
            print(f"  {f.stem:40s}  EX={acc:.4f}  n={len(scores)}")
            loaded.append((f.stem, scores))

        print("\n== Bootstrap 95% CI ==")
        all_ci = {}
        for name, scores in loaded:
            mean, lo, hi = bootstrap_ci(scores)
            all_ci[name] = (mean, lo, hi)
            print(f"  {name:40s}  {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

        print("\n== McNemar pairwise (baseline vs rest) ==")
        base_name, base_scores = loaded[0]
        for name, scores in loaded[1:]:
            report_pair(base_name, base_scores, name, scores)

        out = args.output_dir / "significance.json"
        out.write_text(json.dumps({
            "ci": {n: {"mean": m, "lo": l, "hi": h} for n, (m, l, h) in all_ci.items()},
        }, indent=2))
        print(f"\nSaved CI to {out}")

    elif args.a and args.b:
        _, sa = load_scores(args.a)
        _, sb = load_scores(args.b)
        report_pair(args.a.stem, sa, args.b.stem, sb)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
