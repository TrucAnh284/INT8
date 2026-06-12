#!/usr/bin/env python3
"""
gen_arch_figures.py
  fig13_encoder_arch.pdf     — arctic-embed-m architecture with INT8 targets (like reference image)
  fig14_mp_heatmap.pdf       — mixed-precision profile heatmap (which layer → which dtype)
  fig15_retrieval_overview.pdf — schema retrieval end-to-end overview
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "pdf")
os.makedirs(OUT, exist_ok=True)

# ── Colour palette matching reference image ───────────────────────────────────
INT8_C  = "#DC6B4E"   # orange-red for INT8 layers
FP32_C  = "#4A90D9"   # blue for FP32 layers
NORM_C  = "#888888"   # grey for LayerNorm
GREEN_C = "#4CAF50"   # green for output embedding
BLOCK_C = "#F5F5F5"   # light grey for block background
WHITE   = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def rounded_box(ax, x, y, w, h, text, fc, ec="none", fontsize=8,
                fontcolor="white", bold=False, radius=0.03, lw=1.2,
                ha="center", va="center", fontfamily="DejaVu Sans"):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad={radius}",
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           transform=ax.transData, clip_on=False, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha=ha, va=va, fontsize=fontsize,
            color=fontcolor, fontweight="bold" if bold else "normal",
            fontfamily=fontfamily, zorder=4)


def down_arrow(ax, x, y_from, y_to, color="#555", lw=1.5):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10), zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — arctic-embed-m Encoder Architecture with INT8 Targets
# ─────────────────────────────────────────────────────────────────────────────
def fig_encoder_arch():
    """
    Horizontal 2-panel landscape layout (fits as figure* ~3.5in tall):
    (a) left  — overall encoder flow (bottom→top, compact)
    (b) right — expanded L0 block sublayers
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.5, 3.4),
                                      gridspec_kw={"width_ratios": [1, 1.7],
                                                   "wspace": 0.10})
    for ax in (ax_a, ax_b):
        ax.axis("off")

    # ── Panel (a): compact overview flow ────────────────────────────────────
    ax = ax_a
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)

    def rb(ax, x, y, w, h, txt, fc, fs=8, bold=False):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=fc, edgecolor="none", linewidth=1.2,
                            transform=ax.transData, clip_on=False, zorder=3)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold" if bold else "normal",
                multialignment="center", zorder=4, clip_on=False)

    def arr_a(ax, x, y0, y1):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                    lw=1.3, mutation_scale=9), zorder=5)

    cx, bw, bx = 5, 8.5, 0.75
    items = [
        (0.4,  "Natural Language Query",          "#888888"),
        (2.0,  "Token Emb. + Pos. Enc.  [FP32]",   FP32_C),
        (3.6,  "Encoder Blocks ×12  [INT8]",        INT8_C),
        (6.4,  "Pooler Linear  [FP32]",             FP32_C),
        (8.0,  "768-dim Schema Embedding",          GREEN_C),
    ]
    for y_base, label, fc in items:
        h = 1.1 if y_base not in (0.4, 8.0) else 0.9
        rb(ax, bx, y_base, bw, h, label, fc, fs=8 if y_base != 0.4 else 7.5,
           bold=(y_base not in (0.4,)))
    for y0, y1 in [(1.3,1.9),(3.0,3.5),(5.45,6.3),(7.4,7.9)]:
        arr_a(ax, cx, y0, y1)

    ax.text(0.5, 13.5, "(a)  Overall Architecture", ha="left",
            fontsize=8.5, fontweight="bold", color="#333")

    # ── Panel (b): expanded L0 sublayers ────────────────────────────────────
    ax = ax_b
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)

    bx2, bw2 = 0.5, 9.0
    inner_x, inner_w = 0.9, 8.2
    row_h, gap = 1.45, 0.22

    # background block
    bg_h = 6 * row_h + 5 * gap + 0.5
    bg = FancyBboxPatch((bx2 - 0.1, 0.6), bw2 + 0.2, bg_h + 0.2,
                         boxstyle="round,pad=0.07",
                         facecolor=BLOCK_C, edgecolor="#CCC", linewidth=1.3,
                         transform=ax.transData, clip_on=False, zorder=1)
    ax.add_patch(bg)
    ax.text(cx, bg_h + 1.0, "(b)  Encoder Block L0  (all 12 blocks identical)",
            ha="center", fontsize=8.5, fontweight="bold", color="#333")
    ax.text(0.02, (bg_h + 0.6)/2 + 0.6, "×12", ha="left", va="center",
            fontsize=8.5, color="#888", rotation=90, style="italic")

    def rb_b(y, txt, fc, split=None):
        if split:
            third = (inner_w - 0.16) / 3
            for j, t in enumerate(split):
                rb(ax, inner_x + j*(third+0.08), y, third, row_h,
                   t, INT8_C, fs=8)
            ax.text(inner_x + inner_w + 0.15, y + row_h/2,
                    "INT8", ha="left", va="center",
                    fontsize=7.5, color=INT8_C, fontweight="bold")
        else:
            rb(ax, inner_x, y, inner_w, row_h, txt, fc, fs=8.5)

    y = 0.8
    sublayers = [
        ("Layer Norm   [FP32]", NORM_C, None),
        (None, None, ["Q proj", "K proj", "V proj"]),
        ("Attn Output Projection   [INT8]", INT8_C, None),
        ("Layer Norm   [FP32]", NORM_C, None),
        (None, INT8_C, None),   # FFN row — handled manually
        (None, INT8_C, None),   # FFN row 2
    ]
    for k, (txt, fc, split) in enumerate(sublayers):
        if k == 4:   # FFN1 + FFN2 side by side
            half = (inner_w - 0.1) / 2
            rb(ax, inner_x, y, half, row_h, "FFN Dense 1   [INT8]", INT8_C, fs=8)
            rb(ax, inner_x + half + 0.1, y, half, row_h, "FFN Dense 2   [INT8]", INT8_C, fs=8)
        elif k == 5:
            pass  # already drew FFN in k==4 as one row, skip
        else:
            rb_b(y, txt, fc, split)
        if k < 4:
            y += row_h + gap

    # legend
    handles = [
        mpatches.Patch(color=INT8_C, label="INT8 dynamic (Linear)"),
        mpatches.Patch(color=FP32_C, label="FP32 (embedding, pooler)"),
        mpatches.Patch(color=NORM_C, label="FP32 (LayerNorm)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=7.5, framealpha=0.9, edgecolor="#CCC",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{OUT}/fig13_encoder_arch.pdf")
    plt.close()
    print("fig13_encoder_arch.pdf ✓")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 — Mixed-Precision Profile Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_mp_heatmap():
    """
    Rows = 6 quantizable components (per-block) + embedding + pooler
    Cols = 6 profiles
    Cell colour: orange=INT8, blue=FP32
    """
    profiles = ["FP32 Full\n418 MB", "FP16 Full\n209 MB", "MP-Conserv.\n364 MB",
                "MP-Balanced\n255 MB", "MP-Aggress.\n147 MB", "INT8 Full\n91 MB"]
    components = [
        "Token Embedding",
        "Pooler Linear",
        "Q / K / V proj  (×12)",
        "Attn Output proj  (×12)",
        "FFN Dense 1  (×12)",
        "FFN Dense 2  (×12)",
    ]

    # 1=FP32, 0=INT8, 0.5=FP16
    # profiles: FP32, FP16, MP-Con, MP-Bal, MP-Agg, INT8
    data = np.array([
        # Emb  Pool  Q/K/V  AttnO  FFN1  FFN2
        [1,    1,    1,     1,     1,    1],   # FP32-Full
        [0.5,  0.5,  0.5,   0.5,   0.5,  0.5], # FP16-Full
        [1,    1,    1,     1,     0,    0],    # MP-Conservative
        [1,    1,    0,     0,     0,    0],    # MP-Balanced
        [1,    0,    0,     0,     0,    0],    # MP-Aggressive
        [1,    0,    0,     0,     0,    0],    # INT8-Full (pooler FP32? no, INT8-Full=all linear→INT8, embedding FP32)
    ]).T   # shape (components, profiles)

    # INT8-Full: embed=FP32, pooler=INT8, all linear=INT8
    data[1, 5] = 0    # pooler → INT8

    # custom colormap: 0=INT8(orange), 0.5=FP16(grey), 1=FP32(blue)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "q", [(0.0, "#DC6B4E"), (0.5, "#AAAAAA"), (1.0, "#4A90D9")]
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels(profiles, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(components)))
    component_labels = [f"† {components[0]}"] + components[1:]
    ax.set_yticklabels(component_labels, fontsize=9)
    ax.set_xlabel("Quantization Profile  (RAM = parameter storage)",
                 fontsize=9, labelpad=8)
    ax.set_title("Mixed-Precision Profile Map — arctic-embed-m\n"
                 "(orange = INT8,  grey = FP16,  blue = FP32)",
                 fontsize=10, pad=10)

    # cell text labels
    dtype_label = {1: "FP32", 0.5: "FP16", 0: "INT8"}
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            txt = dtype_label[round(v * 2) / 2]
            fc = "white"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=8.5, color=fc, fontweight="bold")


    # Highlight INT8-Full column
    rect = FancyBboxPatch((-0.5 + 5, -0.5), 1.0, len(components),
                           boxstyle="square,pad=0", linewidth=2.5,
                           facecolor="none", edgecolor="#DC6B4E",
                           transform=ax.transData)
    ax.add_patch(rect)
    ax.text(5, -0.85, "★ Pareto\noptimal", ha="center", fontsize=8,
            color="#DC6B4E", fontweight="bold")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.text(0.02, 0.01,
             "† Token Embedding (nn.Embedding) is not an nn.Linear layer"
             " — quantize_dynamic leaves it in FP32 for all INT8 profiles.",
             fontsize=7.0, color="#666", style="italic", va="bottom")
    fig.savefig(f"{OUT}/fig14_mp_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("fig14_mp_heatmap.pdf ✓")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 15 — Schema Retrieval Overview (end-to-end, inspired by DIN-SQL / DAIL-SQL diagrams)
# ─────────────────────────────────────────────────────────────────────────────
def fig_retrieval_overview():
    """
    Horizontal pipeline showing:
    [NL Question] → [arctic-embed-m INT8] → [sqlite-vec Index] → [Top-k Tables]
                                               ↑ pre-computed
    [DB Schema]   → [arctic-embed-m INT8] ───→ [Embedding Store]
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 22); ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="none", fs=8.5, fc_txt="white", bold=False, multi=False):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=fc, edgecolor=ec, linewidth=1.3,
                            transform=ax.transData, clip_on=False, zorder=3)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=fc_txt, fontweight="bold" if bold else "normal",
                multialignment="center", zorder=4)

    def arr(x0, y0, x1, y1, label="", color="#555"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.4, mutation_scale=12), zorder=5)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2 + 0.25
            ax.text(mx, my, label, ha="center", fontsize=7.5, color="#666", style="italic")

    # ── Top row: query path ───────────────────────────────────────────────
    box(0.3, 5.8, 2.8, 1.4, "Natural\nLanguage\nQuery", FP32_C, fs=8.5, bold=True)
    arr(3.1, 6.5, 4.0, 6.5)

    box(4.0, 5.5, 3.2, 2.0,
        "arctic-embed-m\n(INT8 dynamic)\n109.5M params\n91 MB",
        INT8_C, fs=8, bold=False)
    arr(7.2, 6.5, 8.2, 6.5)

    box(8.2, 5.5, 3.2, 2.0,
        "sqlite-vec\nIndex\n(per-database\nBLOB store)",
        "#5B6A7F", fs=8)
    arr(11.4, 6.5, 12.4, 6.5)

    box(12.4, 5.5, 3.2, 2.0,
        "Top-k Tables\n(k = 5)\nDDL + 3 sample\nrows each",
        GREEN_C, fs=8)
    arr(15.6, 6.5, 16.6, 6.5)

    box(16.6, 5.5, 3.2, 2.0,
        "LLM Prompt\n(DAIL-SQL\nformat)\n~1,400 tokens",
        "#7D3C98", fc_txt="white", fs=8)
    arr(19.8, 6.5, 20.8, 6.5)
    box(20.8, 5.5, 1.0, 2.0, "SQL", GREEN_C, fs=8, bold=True)

    # ── Bottom row: offline indexing ──────────────────────────────────────
    ax.text(11.0, 4.75, "─── Offline (index construction time) ───",
            ha="center", fontsize=8, color="#888", style="italic")

    box(0.3, 1.8, 2.8, 1.4, "Database\nSchema\n(tables)", FP32_C, fs=8, bold=True)
    arr(3.1, 2.5, 4.0, 2.5)

    box(4.0, 1.5, 3.2, 2.0,
        "arctic-embed-m\n(INT8 dynamic)\n(same weights,\ncached)",
        INT8_C, fs=8)
    arr(7.2, 2.5, 8.2, 2.5)

    box(8.2, 1.5, 3.2, 2.0,
        "Embedding\nStore\n(sqlite-vec\nBLOB)",
        "#5B6A7F", fs=8)

    # vertical arrow from embedding store to index
    arr(9.8, 3.5, 9.8, 5.4, color="#5B6A7F")
    ax.text(11.5, 4.3, "pre-computed", ha="left", fontsize=7.5,
            color="#5B6A7F", style="italic")

    # Latency badge
    ax.text(11.0, 0.6,
            "Retrieval latency: 18.4 ms/query  (FP32: 21.7 ms,  INT8: 18.4 ms,  −15.2%)",
            ha="center", fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9E6", ec="#F0C040", lw=1.2))

    ax.set_title("Schema Retrieval Pipeline — arctic-embed-m with INT8 Encoder",
                 fontsize=10, pad=6)
    fig.savefig(f"{OUT}/fig15_retrieval_overview.pdf")
    plt.close()
    print("fig15_retrieval_overview.pdf ✓")


if __name__ == "__main__":
    print("Generating architecture figures...")
    fig_encoder_arch()
    fig_mp_heatmap()
    fig_retrieval_overview()
    print(f"\nAll written to {OUT}/")
