"""Figures for the memo, from committed artifacts only (plan §7).

Every number is read from results/events/*_report.json,
results/exactness/*_k5.json and results/metrics.jsonl — nothing is
hand-typed, so the figures cannot drift from the README tables.
results/plots/table.md is the table-view twin of every figure.

Five figures into results/plots/:
  fig1_stage_wise_tau.png    grouped bars, xLAM-500 + TB-500 per draft,
                             CI whiskers; the transfer story
  fig2_wall_clock.png        three panels (b1 / b32 / TB per-turn),
                             gen tok/s from metrics.jsonl + speedup labels
  fig3_region_alpha.png      two panels (xLAM / TB), per-region alpha for
                             Untuned / Stage-1 / TB-KD — where the gain lives
                             (the TB panel is the asymmetric-transfer proof:
                             xLAM-only training lifts nothing on TB)
  fig4_per_position.png      per-position alpha_n small multiples,
                             untuned vs TB-KD on both evals

Conventions follow the repo's analyzer: flat.tau is the pooled mean, the
bootstrap CI is over per-prompt means (sits slightly above it; both
orderings agree — stated in the README footnote). Region codes:
0=context / 1=assistant prose / 2=tool-call JSON / 3=wrapper tag /
4=final answer; subcuts 10=call name / 11=call args. TB context-region
(region 0) positions are turn-overrun proposals the analyzer counts
faithfully; they carry ~14% of TB verified positions and are reported in
table.md but not drawn (matching the README region table).

Because image reads are unavailable in this workspace, the final
"render and inspect" pass is programmatic: after every figure is drawn,
check_fig() verifies no two text artists overlap and the patch count
matches expectation; after every save, check_png() verifies the PNG
header and dimensions. Failures print a report and exit nonzero.

Palette: 3-slot categorical (validated with the dataviz six-checks
script: adjacent CVD dE 9.2/9.4 light/dark, in band; green's 2.74:1
contrast vs surface obligates the value labels + table view, present).
Global color semantics across figures: gray = untrained/AR baseline,
green = n-gram, blue = xLAM-trained draft (Stage-1), orange = TB-trained
draft (TB-KD / Stage-2 TB-only); fig1 is the exception — its two colors
encode the EVAL DATASET (its legend says so), every draft wears both.
Mark spec: thin marks, hairline solid grid, no borders, 2px gaps,
selective direct labels (bar-tip values on bar charts, story labels only
elsewhere); rounded bar ends are omitted — matplotlib's data-coord
rounding distorts at this aspect, and the anti-pattern list does not
require them.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.patches import Rectangle

# --- palette (validated: adjacent CVD dE 9.2/9.4 light/dark, in band) ------
C_XLAM, C_TB, C_NGRAM = "#2a78d6", "#eb6834", "#1baf7a"  # light mode
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, BASE = "#e1e0d9", "#fcfcfb", "#c3c2b7"
GRAY = "#a7a59e"  # de-emphasis (AR / untuned)

EVENTS, EXACT, METRICS = "results/events", "results/exactness", "results/metrics.jsonl"

DRAFTS = [  # (table name, tick name, xLAM report, TB report)
    ("Untuned Qwen2.5-Coder-0.5B", "Untuned\n0.5B", "draft05", "tb_draft05"),
    ("n-gram (no model)", "n-gram\n(no model)", "ngram", "tb_ngram"),
    ("Stage-1 KD (xLAM ctxs)", "Stage-1 KD\n(xLAM ctxs)", "stage1_draft05", "tb_stage1_draft05"),
    ("TB-KD (TB ctxs)", "TB-KD\n(TB ctxs)", "tbkd_draft05_on_xlam", "tbkd_draft05_on_tb"),
    ("Stage-2 GKD 1:1", "GKD 1:1\n(TB:xLAM)", "stage2_final_on_xlam", "stage2_final_on_tb"),
    ("Stage-2 GKD TB-only", "GKD TB-only\n(final)", "s2tb_final_on_xlam", "s2tb_final_on_tb"),
]

REGIONS = [  # (code, short label) — codes 10/11 read from subcut_alpha
    ("1", "assistant prose"),
    ("2", "tool-call JSON"),
    ("3", "wrapper tags"),
    ("10", "call name"),
    ("11", "call args"),
]
REGIONS_TB = REGIONS + [("4", "final answer")]


def report(tag):
    with open(os.path.join(EVENTS, f"{tag}_k5_report.json")) as f:
        return json.load(f)


def tau_pair(xtag, ttag):
    x, t = report(xtag), report(ttag)
    return (
        x["flat"]["tau"], x["ci"]["tau"]["ci95"],
        t["flat"]["tau"], t["ci"]["tau"]["ci95"],
    )


def load_metrics():
    """Parse metrics.jsonl tolerantly (one host-side append glitch left a
    stray non-JSON prefix on a line; skip to the opening brace)."""
    out = []
    with open(METRICS) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s.startswith("{"):
                s = s[s.index("{"):]
            out.append(json.loads(s))
    return out


def bench_index():
    """(tag, dataset, batch, per_turn, draft-model) -> latest bench row."""
    rows = {}
    for r in load_metrics():
        if r.get("kind") != "bench" or str(r.get("tag", "")).endswith("exactness"):
            continue
        cfg = r.get("spec_config") or {}
        key = (r.get("tag"), r.get("dataset", "").split("/")[-1],
               r.get("batch"), bool(r.get("per_turn")), cfg.get("model"))
        rows[key] = r  # append-only log: latest run wins
    return rows


def _style_ax(ax, horiz=False):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    (ax.xaxis if horiz else ax.yaxis).grid(True, color=GRID, linewidth=1)
    (ax.yaxis if horiz else ax.xaxis).grid(False)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def fig1_tau():
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)
    xs = range(len(DRAFTS))
    bw = 0.34
    for i, (_, _, xt, tt) in enumerate(DRAFTS):
        xtau, xci, ttau, tci = tau_pair(xt, tt)
        for off, tau, ci, col in ((-bw / 2 - 0.012, xtau, xci, C_XLAM),
                                  (bw / 2 + 0.012, ttau, tci, C_TB)):
            ax.bar(i + off, tau, width=bw, color=col, zorder=3)
            ax.plot([i + off, i + off], ci, color=INK2, lw=1.2, zorder=4)
            for y in ci:
                ax.plot([i + off - 0.045, i + off + 0.045], [y, y],
                        color=INK2, lw=1.2, zorder=4)
    # story labels only (selective): the untuned in-domain baseline, the
    # Stage-1 TB column (no transfer), and the TB-KD TB column (transfer fixed)
    for i in (0, 2, 3):
        xtag, ttag = DRAFTS[i][2], DRAFTS[i][3]
        xtau, xci, ttau, tci = tau_pair(xtag, ttag)
        ax.text(i + bw / 2 + 0.012, tci[1] + 0.10, f"{ttau:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK,
                fontweight="bold")
    for i in (0, 2):
        xtau, xci, _, _ = tau_pair(DRAFTS[i][2], DRAFTS[i][3])
        ax.text(i - bw / 2 - 0.012, xci[1] + 0.10, f"{xtau:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK,
                fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d[1] for d in DRAFTS], fontsize=8.5, color=INK)
    ax.set_ylabel("τ  (accepted draft tokens / step, max 5)", fontsize=9, color=INK2)
    ax.set_ylim(0, 5.3)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    handles = [Rectangle((0, 0), 1, 1, fc=C_XLAM), Rectangle((0, 0), 1, 1, fc=C_TB)]
    ax.legend(handles, ["xLAM-500 (in-domain)", "TB-500 (transfer, held-out tools)"],
              loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2,
              handlelength=1.1, handleheight=0.9)
    ax.set_title("Stage-wise acceptance τ, k=5 greedy — in-domain training gains,\n"
                 "xLAM-only training does not transfer, TB training lifts both",
                 fontsize=10, color=INK, pad=10)
    return fig

# --- CONTINUED ---


def exactness_worst():
    """(n_exact, n_prompts, short name) of the worst gate, from files."""
    worst = None
    for name, path in (("untuned", "draft05_k5"), ("n-gram", "ngram_k5"),
                       ("Stage-1", "stage1_k5"), ("TB-KD", "tbkd_k5"),
                       ("Stage-2 mixed", "stage2_k5"), ("Stage-2 TB-only", "s2tb_k5")):
        with open(os.path.join(EXACT, path + ".json")) as f:
            d = json.load(f)
        if worst is None or d["n_exact"] < worst[0]:
            worst = (d["n_exact"], d["n_prompts"], name)
    return worst


def fig2_wallclock():
    b = bench_index()
    fig, axes = plt.subplots(
        1, 3, figsize=(9.6, 3.1), dpi=200,
        gridspec_kw={"width_ratios": [2.2, 1, 1.15], "wspace": 0.42})
    fig.patch.set_facecolor(SURFACE)

    def tok_s(tag, model, dataset="xlam_eval.parquet", batch=1, per_turn=False):
        r = b.get((tag, dataset, batch, per_turn, model))
        if r is None:
            raise KeyError(f"missing bench row: {tag} {model} {dataset} b{batch}")
        return r["median_gen_tok_s"]

    ar_b1 = tok_s("ar", None)
    rows_b1 = [  # (label, tag, model, color)
        ("AR (14B)", "ar", None, GRAY),
        ("n-gram", "ngram", None, C_NGRAM),
        ("Untuned 0.5B", "draft_model", "drafts/coder-0.5b-padded", C_XLAM),
        ("Stage-1 KD", "draft_model", "checkpoints/stage1/final", C_XLAM),
        ("GKD TB-only", "draft_model", "checkpoints/stage2_tb/final", C_XLAM),
    ]
    ar_b32 = tok_s("ar", None, batch=32)
    rows_b32 = [("AR", "ar", None, GRAY),
                ("n-gram", "ngram", None, C_NGRAM),
                ("GKD TB-only", "draft_model", "checkpoints/stage2_tb/final", C_XLAM)]
    ar_tb = tok_s("ar", None, "tb_eval.parquet", 1, True)
    rows_tb = [("AR", "ar", None, GRAY),
               ("n-gram", "ngram", None, C_NGRAM),
               ("GKD TB-only", "draft_model", "checkpoints/stage2_tb/final", C_XLAM)]

    def panel(ax, rows, base, title, xmax, dataset, batch=1, per_turn=False):
        _style_ax(ax, horiz=True)
        for y, (name, tag, model, col) in enumerate(rows):
            v = tok_s(tag, model, dataset, batch, per_turn)
            ax.barh(y, v, height=0.55, color=col, zorder=3)
            ax.text(v + xmax * 0.012, y, f"{v:.0f}  ({v/base:.2f}×)",
                    va="center", ha="left", fontsize=8, color=INK, fontweight="bold")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xticks([])
        ax.set_title(title, fontsize=9, color=INK, pad=6)

    panel(axes[0], rows_b1, ar_b1, "xLAM-500, batch 1", 224, "xlam_eval.parquet", 1, False)
    panel(axes[1], rows_b32, ar_b32, "xLAM, batch 32", 1700, "xlam_eval.parquet", 32, False)
    panel(axes[2], rows_tb, ar_tb, "TB-500 per-turn (transfer)", 175, "tb_eval.parquet", 1, True)
    n_exact, n_prompts, gate = exactness_worst()
    fig.suptitle("Wall-clock generated tok/s (vLLM, median of 3, greedy, k=5) — speedup labels vs same-batch AR;\n"
                 f"exactness gates {n_exact}/{n_prompts} token-identical at worst ({gate}; "
                 "a bf16 near-tie, not the algorithm)",
                 fontsize=10, color=INK, y=1.04)
    return fig


def fig3_regions():
    """The region split, both evals. xLAM panel: cold-start + tag fix.
    TB panel (the asymmetric-transfer proof): Stage-1 (xLAM-trained) sits
    on the untuned baseline in every region — the skill has nothing to
    attach to on TB — while TB-KD lifts every region to a flat 0.83–0.89."""
    picks = [("Untuned 0.5B", "draft05", "tb_draft05", GRAY),
             ("Stage-1 KD (xLAM)", "stage1_draft05", "tb_stage1_draft05", C_XLAM),
             ("TB-KD (TB ctxs)", "tbkd_draft05_on_xlam", "tbkd_draft05_on_tb", C_TB)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1.16], "wspace": 0.18})
    fig.patch.set_facecolor(SURFACE)
    bw = 0.26
    for ax, evalname, codes, xmax in (
            (axes[0], "xLAM-500 (in-domain)", REGIONS, 1.02),
            (axes[1], "TB-500 (transfer, held-out tools)", REGIONS_TB, 1.02)):
        _style_ax(ax)
        labels = [lbl for _, lbl in codes]
        xs = range(len(codes))
        for k, (name, xtag, ttag, col) in enumerate(picks):
            d = report(xtag if evalname.startswith("xLAM") else ttag)
            a = [d["subcut_alpha"][c] if c in ("10", "11") else d["region_alpha"][c]
                 for c, _ in codes]
            offs = (k - 1) * (bw + 0.018)
            ax.bar([x + offs for x in xs], a, width=bw, color=col, zorder=3,
                   label=name)
            if name.startswith("TB-KD"):  # selective: the story series only
                for x, v in zip(xs, a):
                    ax.text(x + offs, v + 0.012, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=7, color=INK)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=7.6, color=INK, rotation=12)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(evalname, fontsize=9.5, color=INK, pad=6)
        if ax is axes[0]:
            ax.set_ylabel("per-token acceptance α", fontsize=9, color=INK2)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8,
                   labelcolor=INK2)
    tb_kd = report("tbkd_draft05_on_tb")
    vals = [tb_kd["region_alpha"][c] for c, _ in REGIONS_TB if c != "10" and c != "11"]
    vals += [tb_kd["subcut_alpha"]["10"], tb_kd["subcut_alpha"]["11"]]
    lo, hi = min(vals), max(vals)
    nv = tb_kd["region_n_verified"]
    n0 = nv["0"]
    n_tot = sum(nv.values())
    pct = 100 * n0 / n_tot
    fig.suptitle("Where the gain lives — α by output region (k=5, greedy).\n"
                 "Left: Stage-1's in-domain gain is cold start + wrapper tags, not diffuse. "
                 "Right: on TB it lifts nothing —\nonly TB-context training (TB-KD) raises every region to a flat "
                 f"{lo:.2f}–{hi:.2f}; region 0 (turn-overrun proposals, {pct:.0f}% of TB positions) not drawn",
                 fontsize=9.5, color=INK, y=1.12)
    return fig


def fig4_positions():
    """Small multiples (one panel per eval) — the single-panel version's
    end-labels collided (xLAM endpoints within 0.012)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), dpi=200,
                             gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor(SURFACE)
    series = [("Untuned", "draft05", GRAY, "-"),
              ("Stage-1 KD", "stage1_draft05", C_XLAM, "-"),
              ("TB-KD", "tbkd_draft05_on_xlam", C_TB, "-")]
    for ax, (name, tag) in zip(axes, (("xLAM-500 (in-domain)", None),
                                       ("TB-500 (transfer)", None))):
        _style_ax(ax)
        xs = range(1, 6)
        for disp, tag0, col, ls in series:
            t = tag0 if "xLAM" in name else {"draft05": "tb_draft05",
                                             "stage1_draft05": "tb_stage1_draft05",
                                             "tbkd_draft05_on_xlam": "tbkd_draft05_on_tb"}[tag0]
            a = report(t)["flat"]["alpha_n"]
            ax.plot(xs, a, color=col, linewidth=2, linestyle=ls, zorder=3,
                    marker="o", markersize=4, markerfacecolor=col,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=disp)
        # endpoints converge (0.89/0.89/0.90) — end-labels would collide;
        # legend + table carry identity instead (spec: leader lines or
        # legend fallback for converging series)
        ax.legend(loc="lower right", frameon=False, fontsize=7.8,
                  labelcolor=INK2, handlelength=1.4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"pos {i}" for i in xs], fontsize=8.5, color=INK)
        ax.set_xlim(0.9, 6.6)
        ax.set_title(name, fontsize=9, color=INK, pad=6)
        ax.set_ylim(0.72, 1.0) if "xLAM" in name else ax.set_ylim(0.72, 0.94)
        if ax is axes[0]:
            ax.set_ylabel("per-position acceptance αₙ", fontsize=9, color=INK2)
    # title claim derived from the artifacts, not hand-typed
    t_x = report("tbkd_draft05_on_xlam")["flat"]["alpha_n"]
    t_t = report("tbkd_draft05_on_tb")["flat"]["alpha_n"]
    u_x = report("draft05")["flat"]["alpha_n"]
    u_t = report("tb_draft05")["flat"]["alpha_n"]
    claim_x = "≥ untuned at 4 of 5 positions" if sum(a >= b for a, b in zip(t_x, u_x)) == 4 else \
              "≥ untuned at every position" if all(a >= b for a, b in zip(t_x, u_x)) else "mixed vs untuned"
    claim_t = "≥ untuned at every position" if all(a >= b for a, b in zip(t_t, u_t)) else \
              f"≥ untuned at {sum(a >= b for a, b in zip(t_t, u_t))} of 5 positions"
    fig.suptitle(f"Per-position αₙ — TB-KD {claim_x} on xLAM, {claim_t} on TB;\n"
                 "colors match fig3 (gray untuned / blue xLAM-trained / orange TB-trained). "
                 "Stage-1 tracks untuned on TB — the no-transfer signature",
                 fontsize=9.5, color=INK, y=1.06)
    return fig


# ---- programmatic verification (image reads unavailable in this workspace) --
def check_fig(fig, name, expect_patches):
    """Geometry checks over every text artist: pairwise overlap and
    canvas bounds. Complements the color validator (which cannot see
    layout)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    seen = set()
    for ax in fig.axes:
        texts = [t for t in ax.texts if t.get_text().strip()]
        keep = []
        for t in texts:
            if id(t) in seen:
                continue
            seen.add(id(t))
            keep.append(t)
        for i in range(len(keep)):
            bb1 = keep[i].get_window_extent(renderer)
            for j in range(i + 1, len(keep)):
                bb2 = keep[j].get_window_extent(renderer)
                if bb1.overlaps(bb2):
                    # bar-cap value labels on adjacent groups are close but
                    # must not overlap; title/suptitle excluded via ax.texts
                    problems.append(f"{name}: text overlap "
                                    f"{keep[i].get_text()[:24]!r} vs "
                                    f"{keep[j].get_text()[:24]!r}")
        for t in keep:
            bb = t.get_window_extent(renderer)
            if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.x1 or bb.y1 > fig.bbox.y1:
                problems.append(f"{name}: text out of canvas {t.get_text()[:24]!r}")
    n_patches = sum(len(ax.patches) for ax in fig.axes)
    if expect_patches is not None and n_patches != expect_patches:
        problems.append(f"{name}: expected {expect_patches} patches, drew {n_patches}")
    return problems


def check_png(path, name):
    """PNG header + dimension sanity; returns problems list."""
    problems = []
    with open(path, "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        problems.append(f"{name}: not a valid PNG")
        return problems
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    if w < 400 or h < 300:
        problems.append(f"{name}: implausible dimensions {w}x{h}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figs = {
        "fig1_stage_wise_tau": (fig1_tau, len(DRAFTS) * 2),
        "fig2_wall_clock": (fig2_wallclock, 5 + 3 + 3),
        "fig3_region_alpha": (fig3_regions, 3 * len(REGIONS) + 3 * (len(REGIONS_TB))),
        "fig4_per_position": (fig4_positions, 0),
    }
    all_problems = []
    for name, (fn, npatch) in figs.items():
        fig = fn()
        all_problems += check_fig(fig, name, npatch)
        path = os.path.join(args.out, name + ".png")
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
        all_problems += check_png(path, name)
        print("wrote", path)
    with open(os.path.join(args.out, "table.md"), "w") as f:
        f.write("# Numbers behind the figures (table-view twin)\n\n"
                "## Fig 1 — stage-wise τ (pooled [bootstrap 95% CI])\n\n"
                "| Draft | xLAM τ | TB τ |\n|---|---|---|\n")
        for tname, _, xt, tt in DRAFTS:
            xtau, xci, ttau, tci = tau_pair(xt, tt)
            f.write(f"| {tname} | {xtau:.2f} [{xci[0]:.2f}–{xci[1]:.2f}] "
                    f"| {ttau:.2f} [{tci[0]:.2f}–{tci[1]:.2f}] |\n")
        b = bench_index()
        f.write("\n## Fig 2 — wall-clock (median of 3, greedy, k=5)\n\n"
                "| Panel | Config | tok/s | speedup |\n|---|---|---|---|\n")
        rows = [("AR", "ar", None),
                ("n-gram", "ngram", None),
                ("Untuned", "draft_model", "drafts/coder-0.5b-padded"),
                ("Stage-1 KD", "draft_model", "checkpoints/stage1/final"),
                ("GKD TB-only", "draft_model", "checkpoints/stage2_tb/final")]
        base = b[("ar", "xlam_eval.parquet", 1, False, None)]["median_gen_tok_s"]
        for cfg, tag, model in rows:
            v = b[(tag, "xlam_eval.parquet", 1, False, model)]["median_gen_tok_s"]
            f.write(f"| xLAM b1 | {cfg} | {v:.1f} | {v/base:.2f}× |\n")
        base32 = b[("ar", "xlam_eval.parquet", 32, False, None)]["median_gen_tok_s"]
        for cfg, tag, model in (("AR", "ar", None), ("n-gram", "ngram", None),
                                ("GKD TB-only", "draft_model", "checkpoints/stage2_tb/final")):
            v = b[(tag, "xlam_eval.parquet", 32, False, model)]["median_gen_tok_s"]
            f.write(f"| xLAM b32 | {cfg} | {v:.1f} | {v/base32:.2f}× |\n")
        basetb = b[("ar", "tb_eval.parquet", 1, True, None)]["median_gen_tok_s"]
        for cfg, tag, model in (("AR", "ar", None), ("n-gram", "ngram", None),
                                ("GKD TB-only", "draft_model", "checkpoints/stage2_tb/final")):
            v = b[(tag, "tb_eval.parquet", 1, True, model)]["median_gen_tok_s"]
            f.write(f"| TB per-turn | {cfg} | {v:.1f} | {v/basetb:.2f}× |\n")
        f.write("\n## Fig 3 — α by region (codes: 1 prose / 2 JSON / 3 tags / "
                "10 name / 11 args / 4 final answer)\n\n"
                "| Draft | Eval | " + " | ".join(
                    lbl for _, lbl in REGIONS_TB) + " | region 0 (not drawn) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for disp, xtag, ttag in (("Untuned 0.5B", "draft05", "tb_draft05"),
                                 ("Stage-1 KD", "stage1_draft05", "tb_stage1_draft05"),
                                 ("TB-KD", "tbkd_draft05_on_xlam", "tbkd_draft05_on_tb")):
            for evname, tag in (("xLAM", xtag), ("TB", ttag)):
                d = report(tag)
                vals = [d["subcut_alpha"].get(c) if c in ("10", "11")
                        else d["region_alpha"].get(c) for c, _ in REGIONS_TB]
                r0 = d["region_alpha"].get("0")
                f.write(f"| {disp} | {evname} | " +
                        " | ".join(f"{v:.3f}" if v is not None else "—" for v in vals) +
                        f" | {r0:.3f} (n={d['region_n_verified']['0']}) |\n")
        f.write("\n## Fig 4 — per-position αₙ\n\n"
                "| Draft | Eval | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 |\n|---|---|---|---|---|---|---|\n")
        for disp, xtag, ttag in (("Untuned", "draft05", "tb_draft05"),
                                 ("Stage-1 KD", "stage1_draft05", "tb_stage1_draft05"),
                                 ("TB-KD", "tbkd_draft05_on_xlam", "tbkd_draft05_on_tb")):
            for evname, tag in (("xLAM", xtag), ("TB", ttag)):
                a = report(tag)["flat"]["alpha_n"]
                f.write(f"| {disp} | {evname} | " +
                        " | ".join(f"{v:.3f}" for v in a) + " |\n")
    print("wrote", os.path.join(args.out, "table.md"))
    if all_problems:
        print("\n".join(all_problems), file=sys.stderr)
        sys.exit(1)
    print("all figures passed geometry checks")


if __name__ == "__main__":
    main()
