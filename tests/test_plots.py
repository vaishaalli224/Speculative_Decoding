"""Golden tests for src/analysis/plots.py — the memo figures.

Runs wherever matplotlib + the committed results/ artifacts exist
(homebrew python3 on this Mac; skipped in .venv, which has no
matplotlib). Everything these tests assert is data-integrity:
the drawn values must equal the committed report JSONs / metrics rows,
and the geometry checks must pass — so a report format change or a
regression in the plotting code fails here rather than shipping a
wrong figure into the memo."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
for p in ("results/events", "results/exactness", "results/metrics.jsonl"):
    if not os.path.exists(p):
        pytest.skip(f"{p} not present (results battery not copied locally)",
                    allow_module_level=True)

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "plots", REPO / "src" / "analysis" / "plots.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def _all_texts(fig):
    out = []
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts if t.get_text().strip()]
    return out


def test_fig1_bars_equal_tau_pairs():
    fig = plots.fig1_tau()
    heights = sorted(round(p.get_height(), 2)
                     for ax in fig.axes for p in ax.patches)
    expected = set()
    for _, _, xt, tt in plots.DRAFTS:
        xtau, _, ttau, _ = plots.tau_pair(xt, tt)
        expected |= {round(xtau, 2), round(ttau, 2)}
    assert set(heights) == expected
    assert len(heights) == 2 * len(plots.DRAFTS)


def test_fig2_values_match_metrics_and_labels_carry_speedups():
    fig = plots.fig2_wallclock()
    b = plots.bench_index()
    labels = _all_texts(fig)
    # every drawn bar's tok/s appears in a label (rounded to int)
    for ax in fig.axes:
        for patch in ax.patches:
            w = patch.get_width()
            assert any(f"{w:.0f}" in lbl for lbl in labels), \
                f"bar {w:.1f} has no value label"
    # the b32 panel's AR baseline is the b32 row, not b1
    ar_b32 = b[("ar", "xlam_eval.parquet", 32, False, None)]["median_gen_tok_s"]
    widths = sorted(p.get_width() for ax in fig.axes for p in ax.patches)
    assert ar_b32 in widths


def test_fig2_worst_gate_matches_exactness_files():
    n_exact, n_prompts, name = plots.exactness_worst()
    files = {"untuned": "draft05_k5", "n-gram": "ngram_k5", "Stage-1": "stage1_k5",
             "TB-KD": "tbkd_k5", "Stage-2 mixed": "stage2_k5", "Stage-2 TB-only": "s2tb_k5"}
    for name_, f in files.items():
        d = json.load(open(REPO / "results" / "exactness" / (f + ".json")))
        assert d["n_exact"] >= n_exact, f"{name_} is worse than reported worst"
    assert (n_exact, n_prompts) == (49, 50) and name == "Stage-1"


def test_fig3_tb_panel_shows_stage1_and_tb_kd():
    """The figure the user asked for: TB region split incl. Stage-1."""
    fig = plots.fig3_regions()
    assert len(fig.axes) == 2
    heights = [p.get_height() for ax in fig.axes for p in ax.patches]
    # 3 drafts × (5 xLAM regions + 6 TB regions) = 33 bars
    assert len(heights) == 3 * (len(plots.REGIONS) + len(plots.REGIONS_TB))
    exp = set()
    for tag, codes in (("draft05", plots.REGIONS), ("stage1_draft05", plots.REGIONS),
                       ("tbkd_draft05_on_xlam", plots.REGIONS),
                       ("tb_draft05", plots.REGIONS_TB),
                       ("tb_stage1_draft05", plots.REGIONS_TB),
                       ("tbkd_draft05_on_tb", plots.REGIONS_TB)):
        d = plots.report(tag)
        for c, _ in codes:
            v = d["subcut_alpha"][c] if c in ("10", "11") else d["region_alpha"][c]
            exp.add(round(v, 3))
    assert set(round(h, 3) for h in heights) == exp


def test_fig3_title_claims_derived_not_typed():
    fig = plots.fig3_regions()
    title = fig._suptitle.get_text()
    tb_kd = plots.report("tbkd_draft05_on_tb")
    vals = [tb_kd["region_alpha"][c] for c, _ in plots.REGIONS_TB
            if c not in ("10", "11")]
    vals += [tb_kd["subcut_alpha"]["10"], tb_kd["subcut_alpha"]["11"]]
    assert f"{min(vals):.2f}–{max(vals):.2f}" in title  # the flat range
    nv = tb_kd["region_n_verified"]
    pct = 100 * nv["0"] / sum(nv.values())
    assert f"{pct:.0f}%" in title  # the region-0 share


def test_fig4_lines_equal_alpha_n_and_claims_true():
    fig = plots.fig4_positions()
    assert sum(len(ax.lines) for ax in fig.axes) == 6
    title = fig._suptitle.get_text()
    # "at every position" claims must be true position-wise
    if "every position on xLAM" in title:
        tx = plots.report("tbkd_draft05_on_xlam")["flat"]["alpha_n"]
        ux = plots.report("draft05")["flat"]["alpha_n"]
        assert all(a >= b for a, b in zip(tx, ux))
    if "every position on TB" in title:
        tt = plots.report("tbkd_draft05_on_tb")["flat"]["alpha_n"]
        ut = plots.report("tb_draft05")["flat"]["alpha_n"]
        assert all(a >= b for a, b in zip(tt, ut))


def test_geometry_checks_pass_on_all_figs():
    checks = {"fig1": (plots.fig1_tau, 2 * len(plots.DRAFTS)),
              "fig2": (plots.fig2_wallclock, 5 + 3 + 3),
              "fig3": (plots.fig3_regions,
                       3 * len(plots.REGIONS) + 3 * len(plots.REGIONS_TB)),
              "fig4": (plots.fig4_positions, 0)}
    problems = []
    for name, (fn, n) in checks.items():
        fig = fn()
        problems += plots.check_fig(fig, name, n)
    assert problems == []


def test_table_md_is_table_view_twin_of_all_figs():
    t = (REPO / "results" / "plots" / "table.md").read_text()
    for section, needle in (
            ("fig1 tau", "## Fig 1"), ("fig2 wall-clock", "## Fig 2"),
            ("fig3 regions", "## Fig 3"), ("fig4 positions", "## Fig 4")):
        assert needle in t, f"table.md missing {section}"
    # spot values that must never drift
    assert "3.32 [3.23–3.43]" in t      # untuned xLAM tau
    assert "2.63 [2.54–2.77]" in t      # untuned TB tau
    assert "1365.8" in t                # s2tb b32 tok/s
    assert "0.623" in t                 # untuned wrapper tags (the cold-start story)
    # region 0 is disclosed, not hidden (14% of TB positions)
    assert "region 0 (not drawn)" in t
