#!/usr/bin/env python3
"""
gen_explain_figures.py — Generate 4 explanatory figures.
  fig9_sql2skeleton.pdf     — sql2skeleton transformation + few-shot matching
  fig10_selfconsistency.pdf — self-consistency + 2-pass correction mechanics
  fig11_int8_mechanism.pdf  — INT8 dynamic quantization mechanism
  fig12_prompt_structure.pdf — DAIL-SQL prompt structure
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "pdf")
os.makedirs(OUT, exist_ok=True)

RC = {
    "font.family": "monospace",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
}
plt.rcParams.update(RC)

C = dict(
    sql="#2C3E50", skel="#1A6B3C", q="#1F4E79",
    hit="#27AE60", miss="#C0392B", arrow="#555",
    box1="#EBF5FB", box2="#E9F7EF", box3="#FEF9E7",
    box4="#FDEDEC", int8="#DC5032", fp32="#3478C5",
    lm="#7D3C98", exec_ok="#27AE60", exec_err="#E74C3C",
)


def draw_box(ax, x, y, w, h, text, fc, ec="gray", fontsize=8,
             bold=False, mono=False, valign="center"):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02",
                           facecolor=fc, edgecolor=ec, linewidth=1.2,
                           transform=ax.transData, clip_on=False)
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ff = "monospace" if mono else plt.rcParams["font.family"]
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va=valign, fontsize=fontsize,
            fontweight=fw, family=ff, wrap=True,
            transform=ax.transData)


def arr(ax, x0, y0, x1, y1, color="gray", style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=1.3, mutation_scale=12))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — sql2skeleton + Few-Shot Matching
# ─────────────────────────────────────────────────────────────────────────────
def _sql_card(ax, x0, y_top, w, diff, col, question, sql_lines, skel_lines):
    """Draw one difficulty card at (x0, y_top) with stacked SQL→skeleton."""
    fs_mono = 6.4

    # header badge
    r = matplotlib.patches.FancyBboxPatch(
        (x0, y_top - 0.52), w, 0.50,
        boxstyle="round,pad=0.04", facecolor=col, edgecolor="#aaa",
        linewidth=0.8, transform=ax.transData, clip_on=False)
    ax.add_patch(r)
    ax.text(x0 + w/2, y_top - 0.27, diff,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#333", clip_on=False)

    # question (italic, truncate to 55 chars)
    q_short = question[:55] + ("…" if len(question) > 55 else "")
    ax.text(x0 + 0.1, y_top - 0.72, q_short,
            fontsize=6.8, style="italic", color="#444",
            va="top", clip_on=False)

    # SQL box
    sql_y0 = y_top - 1.0 - len(sql_lines) * 0.34 - 0.2
    sql_h  = len(sql_lines) * 0.34 + 0.3
    r2 = matplotlib.patches.FancyBboxPatch(
        (x0, sql_y0), w, sql_h,
        boxstyle="round,pad=0.04", facecolor="#F0F8FF", edgecolor=C["sql"],
        linewidth=0.7, transform=ax.transData, clip_on=False)
    ax.add_patch(r2)
    ax.text(x0 + 0.12, y_top - 1.0, "SQL",
            fontsize=6.2, color=C["sql"], fontweight="bold",
            va="top", clip_on=False)
    for i, line in enumerate(sql_lines):
        ax.text(x0 + 0.12, y_top - 1.0 - (i+1)*0.34,
                line, fontsize=fs_mono, family="monospace",
                color=C["sql"], va="top", clip_on=False)

    # strip-values arrow
    arrow_y = sql_y0 - 0.18
    ax.annotate("", xy=(x0 + w/2 + 0.1, arrow_y),
                xytext=(x0 + w/2 - 0.1, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="#999",
                                lw=1.1, mutation_scale=8))
    ax.text(x0 + w/2, arrow_y + 0.16, "strip values",
            ha="center", fontsize=6.0, color="#999", style="italic",
            clip_on=False)

    # skeleton box
    skel_y0 = arrow_y - 0.36 - len(skel_lines) * 0.34 - 0.2
    skel_h  = len(skel_lines) * 0.34 + 0.3
    r3 = matplotlib.patches.FancyBboxPatch(
        (x0, skel_y0), w, skel_h,
        boxstyle="round,pad=0.04", facecolor="#F0FFF4", edgecolor=C["hit"],
        linewidth=0.7, transform=ax.transData, clip_on=False)
    ax.add_patch(r3)
    ax.text(x0 + 0.12, arrow_y - 0.36, "Skeleton",
            fontsize=6.2, color=C["hit"], fontweight="bold",
            va="top", clip_on=False)
    for i, line in enumerate(skel_lines):
        ax.text(x0 + 0.12, arrow_y - 0.36 - (i+1)*0.34,
                line, fontsize=fs_mono, family="monospace",
                color=C["skel"], va="top", clip_on=False)


def fig_sql2skeleton():
    """2-panel: (a) 2×2 grid of difficulty cards, (b) retrieval matching."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 5.2),
                                      gridspec_kw={"width_ratios": [1.7, 1],
                                                   "wspace": 0.06})
    for ax in (ax_a, ax_b):
        ax.axis("off")

    # ── Panel (a): 2×2 difficulty grid ────────────────────────────────────
    ax = ax_a
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)
    ax.set_title("(a)  sql2skeleton Transformation by SQL Complexity",
                 fontsize=9, pad=6, loc="left")

    CARD_W = 4.65   # card width in axis units
    GAP    = 0.35   # horizontal gap between cards

    cards = [
        # (col_idx, row_idx, diff, badge_color, question, sql_lines, skel_lines)
        (0, 0, "Easy", "#81C784",
         "How many cars have more than 4 cylinders?",
         ["SELECT COUNT(*)",
          "FROM cars_data",
          "WHERE cylinders > 4"],
         ["SELECT count(*)",
          "FROM cars_data",
          "WHERE cylinders > VALUE"]),
        (1, 0, "Medium", "#E6C84A",
         "For each stadium, how many concerts?",
         ["SELECT T2.name, COUNT(*)",
          "FROM concert T1 JOIN stadium T2",
          "  ON T1.stadium_id = T2.stadium_id",
          "GROUP BY T1.stadium_id"],
         ["SELECT T2.name, count(*)",
          "FROM concert T1 JOIN stadium T2",
          "  ON T1.stadium_id = T2.stadium_id",
          "GROUP BY T1.stadium_id"]),
        (0, 1, "Hard", "#FFB74D",
         "Countries in Europe with ≥3 car makers?",
         ["SELECT T1.country_name FROM countries T1",
          "JOIN continents T2 ON T1.continent=T2.cont_id",
          "JOIN car_makers T3 ON T1.country_id=T3.country",
          "WHERE T2.continent = 'Europe'",
          "GROUP BY T1.country_name",
          "HAVING COUNT(*) >= 3"],
         ["SELECT T1.country_name FROM countries T1",
          "JOIN continents T2 ON T1.continent=T2.cont_id",
          "JOIN car_makers T3 ON T1.country_id=T3.country",
          "WHERE T2.continent = VALUE",
          "GROUP BY T1.country_name",
          "HAVING count(*) >= VALUE"]),
        (1, 1, "Extra Hard", "#E57373",
         "Avg life expectancy, English not official?",
         ["SELECT AVG(life_expectancy) FROM country",
          "WHERE name NOT IN (",
          "  SELECT T1.name FROM country T1",
          "  JOIN country_language T2",
          "    ON T1.code = T2.country_code",
          "  WHERE T2.language = 'English'",
          "    AND T2.is_official = 'T')"],
         ["SELECT avg(life_expectancy) FROM country",
          "WHERE name NOT IN (",
          "  SELECT T1.name FROM country T1",
          "  JOIN country_language T2",
          "    ON T1.code = T2.country_code",
          "  WHERE T2.language = VALUE",
          "    AND T2.is_official = VALUE)"]),
    ]

    for (ci, ri, diff, col, q, sql_lines, skel_lines) in cards:
        x0    = 0.15 + ci * (CARD_W + GAP)
        y_top = 13.5 - ri * 7.2
        _sql_card(ax, x0, y_top, CARD_W, diff, col, q, sql_lines, skel_lines)

    # ── Panel (b): retrieval matching ─────────────────────────────────────
    ax2 = ax_b
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 14)
    ax2.set_title("(b)  Few-Shot Retrieval\nby Skeleton Similarity",
                  fontsize=9, pad=6, loc="left")

    draw_box(ax2, 0.3, 12.1, 9.4, 1.55, "", C["box1"], C["fp32"])
    ax2.text(0.55, 13.48, "Test question skeleton", fontsize=7.8,
             color=C["fp32"], fontweight="bold")
    ax2.text(0.55, 13.0,
             "SELECT count(*) FROM _ T1\nJOIN _ T2 ON ... GROUP BY _",
             fontsize=7.5, family="monospace", color=C["skel"],
             va="top", linespacing=1.45)

    arr(ax2, 5, 12.1, 5, 11.5, color=C["arrow"])
    ax2.text(5, 11.72, "cosine sim(skeletons)",
             fontsize=7.0, style="italic", color="#888", ha="center")

    pool = [
        ("sim = 0.96  ✓  TOP-1",
         "SELECT count(*) FROM _ T1\nJOIN _ T2 ON ... GROUP BY _",
         C["hit"], C["box2"]),
        ("sim = 0.88  ✓  TOP-2",
         "SELECT count(*) FROM _\nJOIN _ ON ... GROUP BY _ ORDER BY _",
         C["hit"], C["box2"]),
        ("sim = 0.41  ✗  skipped",
         "SELECT avg(_) FROM _\nWHERE _ NOT IN (SELECT ...)",
         C["miss"], C["box4"]),
    ]
    y2 = 11.3
    for lbl, skel_ex, ec, fc in pool:
        draw_box(ax2, 0.3, y2 - 1.72, 9.4, 1.62, "", fc, ec)
        ax2.text(0.55, y2 - 0.22, lbl, fontsize=7.8, color=ec, fontweight="bold")
        ax2.text(0.55, y2 - 0.62, skel_ex, fontsize=7.3,
                 family="monospace", color=C["skel"], va="top", linespacing=1.4)
        y2 -= 2.02

    benefits = [
        "✓  Hides column names → fewer no_such_column errors",
        "✓  Structural match → identical JOIN/HAVING pattern",
        "✓  Value-agnostic → works across databases",
    ]
    ax2.text(0.4, 4.7, "Why it helps:", fontsize=8, fontweight="bold", color="#333")
    for i, b in enumerate(benefits):
        ax2.text(0.4, 4.1 - i * 0.7, b, fontsize=7.3, color="#444")

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig9_sql2skeleton.pdf", bbox_inches="tight")
    plt.close()
    print("fig9_sql2skeleton.pdf ✓")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Self-Consistency + 2-Pass Correction
# ─────────────────────────────────────────────────────────────────────────────
def fig_self_consistency():
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    ax.set_xlim(0, 20); ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Self-Consistency (SC = 3) + Two-Pass Correction", fontsize=10, pad=8)
    plt.rcParams["font.family"] = "sans-serif"

    # Prompt box
    draw_box(ax, 0.2, 8.5, 3.6, 2.0,
             "Prompt\n(schema + few-shot\n+ question)",
             "#EBF5FB", C["fp32"], fontsize=8.5)

    # LLM x3
    for i, t in enumerate(["T=0.7\nCandidate 1", "T=0.7\nCandidate 2", "T=0.7\nCandidate 3"]):
        yx = 9.5 - i * 3.0
        arr(ax, 3.8, 9.5, 5.2, yx, color=C["fp32"])
        draw_box(ax, 5.2, yx - 0.8, 2.8, 1.6,
                 f"LLM\n{t}", "#F5EEF8", "#7D3C98", fontsize=8)

    # SQLs generated
    sqls = [
        ("SELECT a FROM t WHERE b>1", True,  "exec OK"),
        ("SELECT a FROM t WHER b>1",  False, "SyntaxError"),
        ("SELECT a FROM t WHERE x>1", False, "no_such_column: x"),
    ]
    for i, (sql, ok, err) in enumerate(sqls):
        yx = 9.5 - i * 3.0
        arr(ax, 8.0, yx, 9.2, yx, color="#888")
        ec = C["exec_ok"] if ok else C["exec_err"]
        fc = C["box2"] if ok else C["box4"]
        draw_box(ax, 9.2, yx - 0.7, 4.2, 1.4, f"{sql}\n→ {err}",
                 fc, ec, fontsize=7.8, mono=True)

    # Executor selector
    draw_box(ax, 9.4, 3.8, 3.8, 1.5,
             "Executor Selector\npick first SQL that runs\nfallback: Candidate 1",
             "#FEF9E7", "#E67E22", fontsize=8.2)
    for i in range(3):
        yx = 9.5 - i * 3.0
        arr(ax, 11.3, yx - 0.7, 11.3, 5.3, color="#AAA")

    # 2-pass
    arr(ax, 13.2, 4.5, 14.4, 4.5, color=C["arrow"])
    draw_box(ax, 14.4, 3.8, 3.6, 1.5,
             "2-Pass Correction\nif exec error:\nre-prompt + error msg",
             "#FDEDEC", C["int8"], fontsize=8.2)

    arr(ax, 18.0, 4.5, 19.0, 4.5, color=C["hit"])
    draw_box(ax, 19.0, 3.8, 0.8, 1.5, "SQL\nout", C["box2"], C["hit"], fontsize=8)

    # Stat annotations
    ax.text(10.5, 7.8,
            "SC=3 reduces exec errors\nby 57.6% vs single-pass",
            fontsize=8, ha="center", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#CCC"))
    ax.text(16.2, 6.5,
            "2-pass fixes\nremaining 18%",
            fontsize=8, ha="center", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec=C["int8"]))

    ax.annotate("", xy=(11.3, 7.3), xytext=(11.3, 7.8),
                arrowprops=dict(arrowstyle="-|>", color="#AAA", lw=1.0))
    ax.annotate("", xy=(16.2, 5.9), xytext=(16.2, 6.2),
                arrowprops=dict(arrowstyle="-|>", color=C["int8"], lw=1.0))

    fig.savefig(f"{OUT}/fig10_selfconsistency.pdf")
    plt.close()
    print("fig10_selfconsistency.pdf ✓")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — INT8 Dynamic Quantization Mechanism
# ─────────────────────────────────────────────────────────────────────────────
def fig_int8_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2),
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    plt.rcParams["font.family"] = "sans-serif"

    # ── Panel A: weight matrix ──────────────────────────────────────────
    ax = axes[0]
    ax.set_title("(a)  Weight Quantization\n(at model load time)", fontsize=9)
    rng = np.random.RandomState(7)
    W_fp32 = rng.randn(6, 6).astype(np.float32)
    W_int8 = np.clip(np.round(W_fp32 / np.abs(W_fp32).max() * 127), -128, 127).astype(int)

    im = ax.imshow(W_fp32, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("FP32 weights  →  stored as INT8\n"
                  r"$w_{int8} = \mathrm{round}(w_{fp32} / s)$"
                  "\nscale  $s$ = max(|W|) / 127",
                  fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("FP32 value", fontsize=7)
    # Overlay INT8 values
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(W_int8[i, j]), ha="center", va="center",
                    fontsize=6.5, color="black")

    # ── Panel B: activation quantization ───────────────────────────────
    ax2 = axes[1]
    ax2.set_title("(b)  Dynamic Activation Quant.\n(per-token, at inference)", fontsize=9)
    tokens = ["[CLS]", "What", "singers", "from", "France", "[SEP]"]
    act_fp32 = rng.randn(6, 4).astype(np.float32)
    scales = np.abs(act_fp32).max(axis=1)
    act_int8 = np.clip(np.round(act_fp32 / scales[:, None] * 127), -128, 127)

    im2 = ax2.imshow(act_fp32, cmap="PuOr", vmin=-2, vmax=2, aspect="auto")
    ax2.set_xticks(range(4)); ax2.set_xticklabels([f"d{i}" for i in range(4)], fontsize=7)
    ax2.set_yticks(range(6)); ax2.set_yticklabels(tokens, fontsize=8)
    ax2.set_xlabel("Activation matrix (6 tokens × 4 dims)\n"
                   r"$a_{int8}[t] = \mathrm{round}(a_{fp32}[t] / s_t)$"
                   "\n$s_t$ computed fresh for each token $t$", fontsize=8)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=7)

    # Per-row scale annotations
    for i, s in enumerate(scales):
        ax2.text(4.15, i, f"s={s:.2f}", fontsize=6.5, va="center", color="#555")

    # ── Panel C: memory + latency ───────────────────────────────────────
    ax3 = axes[2]
    ax3.set_title("(c)  Effect on Memory & Latency\n(arctic-embed-m, CPU ARM64)", fontsize=9)
    ax3.axis("off")
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)

    rows = [
        ("",         "FP32",  "INT8",  "Δ"),
        ("RAM (MB)",  "418",   "91",   "−78%"),
        ("ms/query",  "64.1",  "54.4", "−15%"),
        ("R@5",       "0.9952","0.9952","0.000"),
        ("MRR",       "0.9513","0.9513","0.000"),
        ("Bits/param","32",    "8",    "−75%"),
    ]
    col_x = [0.3, 2.8, 5.0, 7.3]
    col_colors = ["#555", C["fp32"], C["int8"], C["hit"]]
    y0 = 9.5
    for r, row in enumerate(rows):
        for c, (val, cx, cc) in enumerate(zip(row, col_x, col_colors)):
            fw = "bold" if r == 0 else "normal"
            fc_bg = "#F0F0F0" if r == 0 else ("white" if r % 2 == 0 else "#F9F9F9")
            ax3.text(cx, y0 - r * 1.4, val, fontsize=9,
                     fontweight=fw, color=cc, va="center")
        if r > 0:
            ax3.axhline(y0 - r * 1.4 + 0.7, xmin=0.02, xmax=0.98,
                        color="#DDD", lw=0.7)

    ax3.text(5.0, 0.6,
             "No quality degradation\n"
             "at 4× smaller model",
             ha="center", fontsize=8.5, color=C["hit"],
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#E9F7EF", ec=C["hit"]))

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig11_int8_mechanism.pdf")
    plt.close()
    print("fig11_int8_mechanism.pdf ✓")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — DAIL-SQL Prompt Structure
# ─────────────────────────────────────────────────────────────────────────────
def fig_prompt_structure():
    """
    Compact horizontal prompt structure diagram — 5 stacked bars,
    no code dump. figsize=(9, 2.8) to avoid dominating the page.
    """
    import matplotlib.patches as mpatches2
    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_title("DAIL-SQL Prompt Structure (annotated)", fontsize=9.5, pad=5)

    sections = [
        # (label, token_count, bg_color, border_color, note)
        ("① DDL schema  (CREATE TABLE × k tables)",
         "~480 tok", "#EBF5FB", C["fp32"],
         "Top-k=5 retrieved tables + primary/foreign keys"),
        ("② Sample rows  (3 rows × k tables)",
         "~180 tok", "#F8F9FA", "#999",
         "Representative data values for schema grounding"),
        ("③–④  Few-shot examples  (k question–SQL pairs, ordered by sim)",
         "~630 tok", C["box2"], C["hit"],
         "Structural demonstrations selected by sql2skeleton cosine similarity"),
        ("⑤ Target question  +  SELECT  trigger",
         "~40 tok", "#F5EEF8", "#7D3C98",
         "LLM completes from SELECT onward"),
    ]

    bar_h = 0.82
    gap   = 0.18
    y = 5.6 - 0.45
    for label, tokens, bg, bc, note in sections:
        y -= bar_h
        rect = FancyBboxPatch((0.1, y), 9.0, bar_h - 0.05,
                               boxstyle="round,pad=0.04",
                               facecolor=bg, edgecolor=bc, linewidth=1.3,
                               transform=ax.transData)
        ax.add_patch(rect)
        ax.text(0.28, y + bar_h*0.68, label, fontsize=8.5, color=bc,
                fontweight="bold", va="center")
        ax.text(0.28, y + bar_h*0.28, note, fontsize=7.5, color="#555",
                va="center", style="italic")
        ax.text(9.2, y + bar_h*0.5, tokens, fontsize=8, color="#888",
                ha="right", va="center")
        y -= gap

    # Total token count brace
    ax.text(9.4, 5.6/2, "Total\n~1,400\ntokens",
            fontsize=7.5, color="#888", ha="center", va="center",
            rotation=90)

    ax.annotate("LLM completes from here →",
                xy=(3.5, 1.33), xytext=(3.5, 0.55),
                fontsize=8, color="#7D3C98", ha="center",
                arrowprops=dict(arrowstyle="->", color="#7D3C98", lw=1.2))

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig12_prompt_structure.pdf", bbox_inches="tight")
    plt.close()
    print("fig12_prompt_structure.pdf ✓")


if __name__ == "__main__":
    print("Generating explanatory figures...")
    fig_sql2skeleton()
    fig_self_consistency()
    fig_int8_mechanism()
    fig_prompt_structure()
    print(f"\nAll written to {OUT}/")
