"""Golden tests for the Stage-2 context builder (plan.md §2 Stage 2, §6.7).

Layer 1 (torch-free, vLLM-free): the P1-P6 conventions against synthetic
records and sentinel token ids — no raw data or tokenizer needed for the
pure cores. Where the local rendered pools exist, real-data tests pin the
frozen-index math (P1) and the first real contexts' shape (P2/P3) at raw
scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_prep.build_stage2_contexts import (
    DEFAULT_MAX_CTX,
    DEFAULT_SOURCES,
    interleave,
    left_truncate,
    tb_contexts_from_record,
    xlam_stage2_idx,
)

REPO = Path(__file__).resolve().parents[1]

IM = 7  # sentinel <|im_start|> id — every block starts here in fixtures

RAW_XLAM = REPO / "data" / "raw" / "xlam"
RAW_TB = REPO / "data" / "raw" / "toolbench_default"


def tb_record(n_turns: int, per_block: int = 6, query_id: int = 99):
    """A synthetic rendered TB record: system block, then n_turns rounds of
    (assistant block with labels, tool-response block with none). Token 7
    marks each block start (the <|im_start|> sentinel); ids are positional
    so truncation is reconstructible by eye."""
    ids, labels = [], []
    # system block: [IM, s1..s4, end]
    ids += [IM, 101, 102, 103, 104]
    labels += [-100] * 5
    for t in range(n_turns):
        # assistant block: [IM, 'assistant', a1..a3, end] — a1..a3 labeled
        ids += [IM, 200, 201, 202, 203]
        labels += [-100, -100, 201, 202, 203]
        # tool response block: [IM, 'tool', o1..o3, end] — no labels
        ids += [IM, 300, 301, 302, 303]
        labels += [-100] * 5
    return {"input_ids": ids, "labels": labels, "query_id": query_id}


def block_starts(ids: list[int]) -> list[int]:
    return [i for i, t in enumerate(ids) if t == IM]


class TestXlamIndex:
    def test_pool_minus_carve_in_pool_order(self):
        pool = [10, 3, 7, 1, 9]
        stage1 = {3, 1}
        assert xlam_stage2_idx(pool, list(stage1)) == [10, 7, 9]  # pool order

    def test_carve_outside_pool_refused(self):
        # frozen files desynced — hard error, never a silent gap
        with pytest.raises(AssertionError, match="outside the training pool"):
            xlam_stage2_idx([1, 2, 3], [4])

    @pytest.mark.skipif(not RAW_XLAM.exists(), reason="raw xLAM not downloaded")
    def test_real_frozen_math(self):
        pool = json.loads((REPO / "frozen" / "xlam_train_pool_idx.json").read_text())
        s1 = json.loads((REPO / "frozen" / "stage1_idx.json").read_text())
        idx = xlam_stage2_idx(pool, s1)
        assert len(idx) == 52_794  # 57,794 - 5,000, measured
        assert idx == [i for i in pool if i not in set(s1)]


class TestLeftTruncate:
    def test_short_context_untouched(self):
        ctx = [IM, 1, 2, IM, 3, 4]
        assert left_truncate(ctx, 100, IM) == ctx

    def test_drops_oldest_whole_turns_until_fit(self):
        # 3 conversation blocks + header; cap forces dropping the 1st block
        ctx = [IM, 1, 2, IM, 3, 4, IM, 5, 6, IM, 7, 8]
        out = left_truncate(ctx, 10, IM)
        assert out == [IM, 1, 2, IM, 5, 6, IM, 7, 8]  # block 0 kept, oldest dropped

    def test_fit_keeps_everything_when_cap_allows(self):
        ctx = [IM, 1, 2, IM, 3, 4, IM, 5, 6, IM, 7, 8]
        assert left_truncate(ctx, 12, IM) == ctx

    def test_never_mid_turn_cut(self):
        # every output boundary is a block start (P3): out = block 0 + a
        # suffix starting at some block boundary — never a mid-block join
        ctx = [IM, 1, 2, IM, 3, 4, IM, 5, 6, IM, 7, 8]  # 4 blocks of 3
        out = left_truncate(ctx, 9, IM)
        assert out == ctx[:3] + ctx[6:]  # block 0 + blocks 2,3
        assert out[0] == IM and out[3] == IM and out[6] == IM

    def test_unfittable_returns_none(self):
        # block 0 + header alone exceed the cap -> None (skip, never mangle)
        ctx = [IM, 1, 2, 3, 4, 5, IM, 6]
        assert left_truncate(ctx, 6, IM) is None

    def test_single_conversation_block_unfittable_none(self):
        # only block 0 + header: bounds has len 2 -> loop never runs -> None
        ctx = [IM, 1, 2, IM, 3]
        assert left_truncate(ctx, 3, IM) is None


class TestTbContexts:
    def test_one_context_per_assistant_turn(self):
        rec = tb_record(3)
        ctxs, st = tb_contexts_from_record(rec, max_ctx=4096, im_start_id=IM)
        assert len(ctxs) == 3
        assert st == {"n_turns": 3, "n_kept": 3, "n_truncated": 0, "n_skipped": 0}
        # each cut is at the tool-response block's IM of its round — i.e.
        # ends right before an assistant block, header included
        starts = [i for i, l in enumerate(rec["labels"]) if l != -100 and (
            i == 0 or rec["labels"][i - 1] == -100)]
        for c, s in zip(ctxs, starts):
            assert c["prompt_ids"] == rec["input_ids"][:s]
            assert c["turn"] == 0 or c["turn"] > 0
            assert c["source"] == "tb"
            assert c["query_id"] == 99

    def test_post_observation_states_included(self):
        # turn t>0 contexts end after a tool-response block (post-obs)
        rec = tb_record(2)
        ctxs, _ = tb_contexts_from_record(rec, max_ctx=4096, im_start_id=IM)
        # the turn-1 cut contains the turn-0 tool response (ids 300..303)
        assert 300 in ctxs[1]["prompt_ids"]
        assert 301 in ctxs[1]["prompt_ids"]

    def test_header_property_enforced(self):
        # a record whose labels begin deep INSIDE a block (not right after
        # an assistant header) fails P2 loudly instead of producing a
        # context the model cannot sample from
        ids = [IM, 1, 2, IM, 3, 4, 5, 6, 7, 8]
        rec = {"input_ids": ids, "labels": [-100] * 8 + [8],
               "query_id": 1}
        # labels start at 8 -> ctx = ids[:8] ends 5 tokens past the last IM
        with pytest.raises(AssertionError, match="assistant header"):
            tb_contexts_from_record(rec, 4096, IM)

    def test_over_cap_turn_truncated_then_skipped(self):
        # The tightest legal cut for a turn keeps block 0 + the tool block
        # immediately before its header (the header's own IM is excluded),
        # so a turn is only SKIPPED when that preceding block is oversized.
        # Fixture: round 1's tool block is huge -> turn 2 is unfittable at a
        # cap that turn 1 (small preceding block) still truncates into.
        ids, labels = [], []
        ids += [IM, 101, 102, 103, 104]
        labels += [-100] * 5
        for wt in (3, 12, 3):  # tool-block payload widths per round
            ids += [IM, 200, 210, 211, 212]           # assistant block
            labels += [-100, -100, 210, 211, 212]
            ids += [IM, 300] + [310 + i for i in range(wt)]  # tool block
            labels += [-100] * (2 + wt)
        rec = {"input_ids": ids, "labels": labels, "query_id": 5}
        starts = [i for i, l in enumerate(rec["labels"]) if l != -100 and (
            i == 0 or rec["labels"][i - 1] == -100)]
        assert starts == [7, 17, 36]
        bs = block_starts(ids)
        cap = 15
        # turn 0: 7 <= cap untouched; turn 1: 17 > cap, tightest 12 fits;
        # turn 2: 36 > cap, tightest (block 0 + huge r1-tool + header) = 21
        ctxs, st = tb_contexts_from_record(rec, cap, IM)
        assert st["n_turns"] == 3
        assert st["n_kept"] == 2
        assert st["n_truncated"] == 1
        assert st["n_skipped"] == 1
        assert [c["turn"] for c in ctxs] == [0, 1]
        assert len(ctxs[1]["prompt_ids"]) == 12 <= cap
        assert ctxs[1]["prompt_ids"][0] == IM  # block 0 preserved (P3)
        # truncated turn-1 ctx = block 0 + round-0 tool block + header
        assert ctxs[1]["prompt_ids"] == ids[: bs[1]] + ids[bs[2]: starts[1]]

    def test_empty_labels_no_contexts(self):
        rec = {"input_ids": [IM, 1, 2], "labels": [-100, -100, -100],
               "query_id": 1}
        ctxs, st = tb_contexts_from_record(rec, 100, IM)
        assert ctxs == [] and st["n_turns"] == 0


class TestInterleave:
    def test_alternates_groups_then_drains(self):
        a = [[("x", 1)], [("x", 2)], [("x", 3)]]
        b = [[("t", 1), ("t", 2)], [("t", 3)]]
        out = list(interleave(iter(a), iter(b)))
        assert out == [("x", 1), ("t", 1), ("t", 2), ("x", 2), ("t", 3),
                       ("x", 3)]

    def test_one_side_empty(self):
        out = list(interleave(iter([["a"], ["b"]]), iter([])))
        assert out == ["a", "b"]

    def test_both_empty(self):
        assert list(interleave(iter([]), iter([]))) == []

    def test_lazy(self):
        # interleave pulls one group at a time (P5) — a generator that
        # fails on its 3rd next() proves the first two groups were not
        # pre-rendered

        def boom():
            yield [1]
            yield [2]
            raise RuntimeError("rendered too far")

        got = []
        it = interleave(boom(), iter([]))
        got.append(next(it))
        got.append(next(it))
        assert got == [1, 2]

    def test_source_order_matches_flag(self):
        # DEFAULT_SOURCES = "xlam,tb": xLAM leads; the stream builder
        # itself is covered by the real-data tests below
        assert DEFAULT_SOURCES == "xlam,tb"
        assert DEFAULT_MAX_CTX == 4096


@pytest.fixture(scope="module")
def real_stats():
    """12 real contexts through the full stream — shared by the tests
    below (one render pass, several assertions)."""
    from src.data_prep.build_stage2_contexts import build_stage2_contexts

    st: dict = {}
    ctxs = build_stage2_contexts(limit=12, stats=st)
    return ctxs, st


class TestRealStream:
    """Real raw data, small limits — the P1-P4 stream at raw scale."""

    @pytest.mark.skipif(
        not (RAW_XLAM.exists() and RAW_TB.exists()),
        reason="raw xLAM/ToolBench not downloaded",
    )
    def test_stream_interleaves_and_ends_with_header(self, real_stats):
        ctxs, st = real_stats
        assert len(ctxs) == 12
        assert st["n_contexts"] == 12
        # P4: alternating CONVERSATIONS (xLAM first); a TB conversation's
        # multiple turn-contexts stay adjacent, so the pattern is
        # xlam, (tb-group of k contexts), xlam, (tb-group), ...
        assert ctxs[0]["source"] == "xlam"
        assert ctxs[1]["source"] == "tb"
        # the TB group is contiguous and then another xLAM context follows
        first_tb = [c for c in ctxs if c["source"] == "tb"]
        qid0 = first_tb[0]["query_id"]
        grp = [c for c in first_tb if c["query_id"] == qid0]
        pos = ctxs.index(grp[0])
        assert all(
            ctxs[pos + i]["query_id"] == qid0 for i in range(len(grp))
        )
        assert pos + len(grp) < len(ctxs)
        assert ctxs[pos + len(grp)]["source"] == "xlam"
        assert all(c["source"] in ("xlam", "tb") for c in ctxs)
        # every context ends with a fresh assistant header (P2)
        from src.data_prep.render import get_tokenizer

        tok = get_tokenizer()
        for c in ctxs:
            assert tok.decode(c["prompt_ids"]).endswith("<|im_start|>assistant\n")
            assert len(c["prompt_ids"]) <= 4096  # P3 cap

    @pytest.mark.skipif(
        not (RAW_XLAM.exists() and RAW_TB.exists()),
        reason="raw xLAM/ToolBench not downloaded",
    )
    def test_limit_counts_contexts_and_counts_honestly(self, real_stats):
        ctxs, st = real_stats
        assert st["n_xlam"] + st["n_tb"] == 12  # stream-level consumed counts
        assert st["n_xlam"] == len([c for c in ctxs if c["source"] == "xlam"])
        assert st["n_tb"] == len([c for c in ctxs if c["source"] == "tb"])
        # n_tb_conv counts conversations the iterator SAW (<= kept contexts:
        # each conversation contributes >= 1 context)
        assert st["n_tb_conv"] <= st["n_tb"]
        # TB contexts within a conversation are turn-ordered 0,1,2,...
        tb = [c for c in ctxs if c["source"] == "tb"]
        if len(tb) >= 2:
            turns = [c["turn"] for c in tb if c["query_id"] == tb[0]["query_id"]]
            assert turns == list(range(len(turns)))

    @pytest.mark.skipif(
        not (RAW_XLAM.exists() and RAW_TB.exists()),
        reason="raw xLAM/ToolBench not downloaded",
    )
    def test_xlam_context_matches_stage1_prompt_render(self, real_stats):
        """The xLAM stream's rendered prompt is the SAME render Stage 1
        used for its prompts — token-identical, from the same raw row."""
        ctxs, _ = real_stats
        xctxs = [c for c in ctxs if c["source"] == "xlam"]
        assert xctxs, "interleave put xLAM first — must have some"
        from datasets import load_from_disk

        from src.data_prep.build_distill_data import render_stage1_prompt

        raw = load_from_disk(RAW_XLAM)["train"]
        c = xctxs[0]
        assert c["prompt_ids"] == render_stage1_prompt(raw[c["query_id"]])
        assert c["turn"] is None


@pytest.mark.skipif(
    not (RAW_XLAM.exists() and RAW_TB.exists()),
    reason="raw xLAM/ToolBench not downloaded",
)
class TestRealStreamCLI:
    """The stats CLI (P6) — run as a subprocess to pin its output shape."""

    def test_stats_cli(self):
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-m", "src.data_prep.build_stage2_contexts",
             "stats", "--limit", "8"],
            capture_output=True, text=True, cwd=REPO,
        )
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        assert out["n_contexts"] == 8
        assert out["lengths"]["all"]["n"] == 8
        assert set(out["lengths"]) == {"all", "xlam", "tb"}
        assert out["n_xlam"] + out["n_tb"] == 8
