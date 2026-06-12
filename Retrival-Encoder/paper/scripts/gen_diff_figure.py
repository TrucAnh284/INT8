#!/usr/bin/env python3
"""
gen_diff_figure.py — Generate fig16_difficulty_breakdown.pdf
Run: python3 paper/scripts/gen_diff_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "pdf")
os.makedirs(OUT, exist_ok=True)

RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
}
plt.rcParams.update(RC)

# ── Data ──────────────────────────────────────────────────────────────────────
DIFFS   = ["Easy", "Medium", "Hard", "Extra Hard"]
COUNTS  = [248, 446, 174, 166]   # Spider 1.0 dev set distribution

DATA = {
    "Baseline":       [0.8727, 0.8533, 0.6235, 0.6260],
    "+sql2skeleton":  [0.8743, 0.8647, 0.7546, 0.5743],
    "+$k$=5":         [0.8802, 0.8698, 0.7730, 0.5850],
    "+SC=3":          [0.8862, 0.8667, 0.7725, 0.5762],
    "Best (SC3+$k$5+2pass)": [0.8929, 0.8836, 0.7952, 0.5871],
}

COLORS = ["#9E9E9E", "#4DB6AC", "#26A69A", "#0288D1", "#E65100"]
LABELS = list(DATA.keys())
VALS   = np.array(list(DATA.values()))   # (5, 4)

# Δ EX relative to Baseline
DELTA = VALS[1:] - VALS[0]              # (4, 4)
DELTA_LABELS = ["+sql2skeleton", "+$k$=5", "+SC=3", "Best"]
DELTA_COLORS = ["#4DB6AC", "#26A69A", "#0288D1", "#E65100"]

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3),
                          gridspec_kw={"width_ratios": [0.9, 2.8, 2.4],
                                       "wspace": 0.38})

# ── Panel (a): difficulty distribution ───────────────────────────────────────
ax = axes[0]
bar_colors = ["#81C784", "#FFF176", "#FFB74D", "#E57373"]
bars = ax.barh(DIFFS, COUNTS, color=bar_colors, edgecolor="#aaa", height=0.55)
ax.set_xlabel("Question count")
ax.set_title("(a) Dev Set Distribution", fontsize=9, pad=6)
ax.set_xlim(0, 520)
for bar, c in zip(bars, COUNTS):
    ax.text(c + 8, bar.get_y() + bar.get_height()/2,
            str(c), va="center", fontsize=8, color="#444")

# ── Panel (b): grouped bar chart EX by config × difficulty ───────────────────
ax = axes[1]
x = np.arange(len(DIFFS))
n = len(LABELS)
width = 0.14
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

for i, (label, vals, col) in enumerate(zip(LABELS, VALS, COLORS)):
    bars = ax.bar(x + offsets[i], vals, width=width*0.92, color=col,
                  label=label, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(DIFFS)
ax.set_ylabel("Execution Accuracy (EX)")
ax.set_title("(b) EX by Difficulty and Configuration", fontsize=9, pad=6)
ax.set_ylim(0.45, 0.97)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.legend(loc="lower left", ncol=1, fontsize=7, framealpha=0.85,
          handlelength=1.2, handletextpad=0.5)

# annotate Baseline and Best on Hard column
for config_idx, ann_label in [(0, "Baseline"), (4, "Best")]:
    v = VALS[config_idx][2]  # Hard = index 2
    off = offsets[config_idx]
    ax.text(x[2] + off, v + 0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=6.5, color="#333")

# ── Panel (c): Δ EX heatmap ───────────────────────────────────────────────────
ax = axes[2]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rg", ["#EF9A9A", "#FFFFFF", "#A5D6A7"])
vmax = 0.20

im = ax.imshow(DELTA, cmap=cmap, aspect="auto",
               vmin=-vmax, vmax=vmax)

ax.set_xticks(range(4))
ax.set_xticklabels(DIFFS, fontsize=8)
ax.set_yticks(range(4))
ax.set_yticklabels(DELTA_LABELS, fontsize=8)
ax.set_title("(c) $\\Delta$ EX vs. Baseline (pp)", fontsize=9, pad=6)

for i in range(4):
    for j in range(4):
        v = DELTA[i, j]
        color = "black" if abs(v) < 0.12 else "white"
        ax.text(j, i, f"{v*100:+.1f}", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("$\\Delta$ EX", fontsize=7)

plt.tight_layout()
fig.savefig(f"{OUT}/fig16_difficulty_breakdown.pdf", bbox_inches="tight")
plt.close()
print(f"fig16_difficulty_breakdown.pdf  ✓  → {OUT}/")
