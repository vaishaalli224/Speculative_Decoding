"""Golden tests for the acceptance-metrics evaluation script (plan.md §4.1–4.3,
§6.4). Every counting convention is pinned here against hand-computed cases,
so the instrumented loop (§6.5) can be validated by emitting events and
asserting this script's output.

Hand-computed reference cases (single prompt, k=3):

  Events:
    step 0: draft [t0 t1 t2], mask [T T F], correction c0
            -> verified 2, accepted 2, bonus 1, emitted 3
    step 1: draft [t3 t4 t5], mask [T T T], correction c1
            -> verified 3, accepted 3, bonus 1, emitted 4
    step 2: draft [t6 t7 t8], mask [T F .], correction c2
            -> mask must have exactly 3 entries? No — the loop may propose
               fewer than k tokens (draft emitted eos mid-proposal). Mask
               [T F] with 2 tokens: verified 1, accepted 1, bonus 1,
               emitted 2

  flat:
    alpha   = (2+3+1) / (3+3+2) = 6/8 = 0.75
    tau     = (2+3+1)/3 = 2.0
    bonus   = 3/3 = 1.0
    emitted = 3+4+2 = 9
  per-position (verified(n) counts steps that REACHED position n):
    n=0: 3 steps, 3 accepted -> 1.0
    n=1: 3 steps, 2 accepted -> 2/3
    n=2: 2 steps (step 2 rejected at n=1 so never reached n=2),
         1 accepted -> 1/2
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.eval_acceptance import (
    SUB_ARGS,
    SUB_NAME,
    _event_stats,
    assign_token_regions,
    bootstrap_ci,
    events_by_prompt,
    flat_metrics,
    full_report,
    load_events,
    load_records,
    name_vs_arguments_cut,
    per_prompt_records,
    _turn_starts_from_labels,
)
from src.data_prep.render import (
    REGION_FINAL_ANSWER,
    REGION_PROSE,
    REGION_TAG,
    REGION_TOOL_CALL,
    unrle,
)

FROZEN = Path(__file__).resolve().parents[1] / "frozen"
REPO = Path(__file__).resolve().parents[1]


def ev(qid, step, tokens, mask, corr=None, eos=False, turn=None):
    return {
        "query_id": qid,
        "step": step,
        "draft_tokens": tokens,
        "accept_mask": mask,
        "correction_token": corr,
        "eos": eos,
        **({"turn": turn} if turn is not None else {}),
    }


EVENTS_REF = [
    ev("q0", 0, [10, 11, 12], [True, True, False], corr=99),
    ev("q0", 1, [13, 14, 15], [True, True, True], corr=98),
    ev("q0", 2, [16, 17], [True, False], corr=97),
]


class TestEventStats:
    def test_leading_true_counting(self):
        assert _event_stats(EVENTS_REF[0]) == {
            "verified": 2, "accepted": 2, "scored": 3, "bonus": 1, "emitted": 3,
        }

    def test_accept_after_reject_is_schema_violation(self):
        bad = ev("q", 0, [1, 2, 3], [True, False, True])
        with pytest.raises(ValueError):
            _event_stats(bad)

    def test_full_accept(self):
        assert _event_stats(EVENTS_REF[1])["verified"] == 3

    def test_short_proposal(self):
        assert _event_stats(EVENTS_REF[2]) == {
            "verified": 1, "accepted": 1, "scored": 2, "bonus": 1, "emitted": 2,
        }


class TestFlatMetrics:
    def test_reference_case(self):
        m = flat_metrics(EVENTS_REF)
        assert m["alpha"] == pytest.approx(0.75)
        assert m["tau"] == pytest.approx(2.0)
        assert m["bonus_rate"] == pytest.approx(1.0)
        assert m["tokens_emitted"] == 9
        assert m["n_events"] == 3
        assert m["n_prompts"] == 1

    def test_per_position_reference(self):
        m = flat_metrics(EVENTS_REF)
        assert m["alpha_n"][0] == pytest.approx(1.0)
        assert m["alpha_n"][1] == pytest.approx(2 / 3)
        assert m["alpha_n"][2] == pytest.approx(0.5)
        assert m["pos_scored"] == [3, 3, 2]
        assert m["pos_accepted"] == [3, 2, 1]

    def test_multi_prompt_alpha_is_step_pooled(self):
        # alpha pools scored verdicts across prompts; tau is per-event mean
        evs = EVENTS_REF + [
            ev("q1", 0, [20], [False], corr=50),        # 1 scored, 0 accepted
            ev("q1", 1, [21, 22], [True, True], corr=51),
        ]
        m = flat_metrics(evs)
        # scored: q0 (3+3+2) + q1 (1+2) = 11; accepted: 6 + 2 = 8
        assert m["pos_scored"][0] == 5
        assert m["alpha"] == pytest.approx(8 / 11)
        assert m["tau"] == pytest.approx((2 + 3 + 1 + 0 + 2) / 5)

    def test_k_max_caps_positions(self):
        m = flat_metrics(EVENTS_REF, k_max=2)
        assert len(m["alpha_n"]) == 2

    def test_load_events_validates(self, tmp_path):
        p = tmp_path / "e.jsonl"
        with open(p, "w") as f:
            for e in EVENTS_REF:
                f.write(json.dumps(e) + "\n")
        assert len(load_events(p)) == 3
        # malformed: mask length mismatch
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps(ev("q", 0, [1], [True, True])) + "\n")
        with pytest.raises(ValueError):
            load_events(bad)
        # empty file
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError):
            load_events(empty)


class TestRegionAssignment:
    """Cursor arithmetic + region anchoring on a synthetic record."""

    # record: 4 prompt tokens (region 0), then 6 assistant tokens with
    # regions [1, 3, 2, 2, 2, 4]
    RMAP = [0, 0, 0, 0, 1, 3, 2, 2, 2, 4]
    N_PROMPT = 4

    def test_cursor_advances_by_emitted(self):
        events = [
            ev("q", 0, [100, 101, 102], [True, True, False], corr=103),
            ev("q", 1, [104, 105], [True, True], corr=None),
        ]
        # step0: proposals at positions 4,5,6 -> regions [1, 3, 2]
        #        accepted 2 -> emitted tokens at 4,5 ([1,3]) + corr at 6 ([2])
        # step1: cursor = 4+3 = 7; proposals at 7,8 -> [2, 2]; accepted 2
        #        emitted [2,2], no correction
        out = assign_token_regions(events, self.RMAP, self.N_PROMPT)
        (p0, e0), (p1, e1) = out
        assert p0 == [REGION_PROSE, REGION_TAG, REGION_TOOL_CALL]
        assert e0 == [REGION_PROSE, REGION_TAG, REGION_TOOL_CALL]
        assert p1 == [REGION_TOOL_CALL, REGION_TOOL_CALL]
        assert e1 == [REGION_TOOL_CALL, REGION_TOOL_CALL]

    def test_turn_field_resets_cursor(self):
        # multi-turn record: 2 assistant turns. turn 0 spans pos 4-6
        # (loss), pos 7 is the between-turn context token (no loss),
        # turn 1 starts at pos 8.
        rmap = [0, 0, 0, 0, 1, 3, 0, 1, 1, 4]
        labels = [-100, -100, -100, -100, 5, 6, 7, -100, 8, 9]
        events = [
            ev("q", 0, [100], [True], corr=None, turn=0),
            ev("q", 1, [110], [True], corr=None, turn=1),
        ]
        out = assign_token_regions(events, rmap, 4, labels)
        (p0, e0), (p1, e1) = out
        assert p0 == [REGION_PROSE]  # pos 4
        assert p1 == [REGION_PROSE]  # pos 8 (turn-1 start), NOT pos 7 (ctx)
        assert e0 == [REGION_PROSE]
        assert e1 == [REGION_PROSE]

    def test_turn_boundaries_from_labels(self):
        labels = [-100, -100, 1, 2, -100, 3, 4, 5]
        assert _turn_starts_from_labels(labels) == [2, 5]
        assert _turn_starts_from_labels([1, 2]) == [0]
        assert _turn_starts_from_labels([-100, -100]) == []

    def test_overrun_positions_are_prose(self):
        # draft proposes past the record end -> positions beyond map = prose.
        # map has 10 tokens; proposal n sits at output position 4+n.
        # n=5 -> pos 9 = final-answer token; n=6 -> pos 10 = beyond -> prose
        events = [ev("q", 0, list(range(20)), [True] * 20, corr=None)]
        out = assign_token_regions(events, self.RMAP, self.N_PROMPT)
        prop, emit = out[0]
        assert prop[5] == REGION_FINAL_ANSWER  # pos 9, inside the map
        assert prop[6] == REGION_PROSE  # pos 10, beyond the 10-token map
        assert len(prop) == 20
        assert emit[-1] == REGION_PROSE  # emitted token at pos 23

    def test_region_alpha_split(self):
        events = [
            ev("q", 0, [100, 101, 102], [True, True, False], corr=103),
            ev("q", 1, [104, 105], [True, True], corr=None),
        ]
        # coarse: proposals in prose(1): positions 4 (step0 n=0) -> 1 verified,
        # 1 accepted. tag(3): 1v 1a. tool-call(2): step0 n=2 (rejected) 1v 0a,
        # step1 both accepted 2v 2a -> 3v 2a. final-answer(4): none reached.
        rep = full_report(events, records=[self._rec()])
        assert rep["region_alpha"]["1"] == pytest.approx(1.0)
        assert rep["region_alpha"]["3"] == pytest.approx(1.0)
        assert rep["region_alpha"]["2"] == pytest.approx(2 / 3)
        assert "4" not in rep["region_alpha"]
        assert rep["region_n_verified"] == {"1": 1, "2": 3, "3": 1}
        # emitted region-2 tokens: step0 corr at pos 6, step1 accepted at
        # pos 7,8 -> 3
        assert rep["region_n_emitted"]["2"] == 3

    def _rec(self):
        # single assistant turn spanning pos 4..9 (regions 1,3,2,2,2,4),
        # all loss-bearing; the correction lands at pos 6 (region 2)
        return {
            "query_id": "q",
            "input_ids": [1] * 10,
            "regions": [[0, 4], [1, 1], [3, 1], [2, 3], [4, 1]],
            "labels": [-100] * 4 + [5, 6, 7, 8, 9, 10],
            "n_prompt_tokens": 4,
            "n_tokens": 10,
        }


class TestSubCut:
    def test_name_vs_arguments_split(self):
        # payload tokens decode piecewise to the target's real tool-call
        # shape: '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        # 4 payload tokens: '{"name": "get_weather", ' / '"arguments"' /
        # ': {"city": "Paris"' / '}}' — cut at the '"arguments"' key.
        pieces = ['{"name": "get_weather", ', '"arguments"', ': {"city": "Paris"', "}}"]
        n_payload = len(pieces)
        rmap = [0, 0, 1, 3] + [2] * n_payload + [4]
        ids = list(range(len(rmap)))

        def decode(ids_slice):
            i0 = ids_slice[0] - 4
            return "".join(pieces[i0 : i0 + len(ids_slice)])

        sub = name_vs_arguments_cut(rmap, ids, decode)
        payload_sub = sub[4 : 4 + n_payload]
        # piece 0 (the '"name"' side) is SUB_NAME; pieces 1.. are SUB_ARGS
        assert payload_sub[0] == SUB_NAME
        assert all(s == SUB_ARGS for s in payload_sub[1:])
        # non-payload regions pass through unchanged
        assert sub[:4] == [0, 0, 1, 3]
        assert sub[-1] == REGION_FINAL_ANSWER

    def test_cut_inside_token_goes_to_args(self):
        # '"name": "f", "arg' + 'uments": {}' — the '"arguments"' key starts
        # mid-token-2; that token begins the args side (end > cut)
        pieces = ['{"name": "f", ', '"arg', 'uments": {}}']
        rmap = [3] + [2] * 3
        ids = list(range(4))

        def decode(ids_slice):
            i0 = ids_slice[0] - 1
            return "".join(pieces[i0 : i0 + len(ids_slice)])

        sub = name_vs_arguments_cut(rmap, ids, decode)
        assert sub[1] == SUB_NAME  # before the cut
        assert sub[2] == SUB_ARGS  # contains the cut start
        assert sub[3] == SUB_ARGS

    def test_no_arguments_key_all_name(self):
        pieces = ['{"name": ', '"f"}']
        rmap = [3] + [2] * 2
        ids = list(range(3))

        def decode(ids_slice):
            i0 = ids_slice[0] - 1
            return "".join(pieces[i0 : i0 + len(ids_slice)])

        sub = name_vs_arguments_cut(rmap, ids, decode)
        assert sub[1:] == [SUB_NAME, SUB_NAME]


class TestBootstrapCI:
    def test_ci_over_prompts(self):
        by = events_by_prompt(EVENTS_REF)
        assert set(by) == {"q0"}
        ci = bootstrap_ci(EVENTS_REF, "alpha")
        assert ci["n_prompts"] == 1
        assert ci["mean"] == pytest.approx(0.75)
        assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]

    def test_ci_two_prompts(self):
        evs = EVENTS_REF + [
            ev("q1", 0, [20], [False], corr=50),
        ]
        ci = bootstrap_ci(evs, "alpha")
        assert ci["n_prompts"] == 2

    def test_tau_ci(self):
        ci = bootstrap_ci(EVENTS_REF, "tau")
        assert ci["mean"] == pytest.approx(2.0)


class TestPerPrompt:
    def test_records(self):
        recs = per_prompt_records(EVENTS_REF)
        assert len(recs) == 1
        r = recs[0]
        assert r["query_id"] == "q0"
        assert r["n_steps"] == 3
        assert r["n_scored"] == 8
        assert r["n_accepted"] == 6
        assert r["alpha"] == pytest.approx(0.75)
        assert r["tau"] == pytest.approx(2.0)
        assert r["n_emitted"] == 9


class TestSelfConsistencyConvention:
    """The §4.4 self-consistency test the loop must satisfy: draft = target
    (greedy) ⇒ every draft token accepted. This pins the event shape the
    loop must emit for that case: full-accept events with a bonus token."""

    def test_all_accepted_tau_equals_k(self):
        k = 5
        events = [
            ev(f"q{i}", s, list(range(k)), [True] * k, corr=999, eos=True)
            for i in range(3)
            for s in range(4)  # 4 steps, last one eos
        ]
        m = flat_metrics(events, k_max=k)
        assert m["alpha"] == 1.0
        # tau counts accepted draft tokens per step: k, not k+1 (bonus
        # reported separately)
        assert m["tau"] == pytest.approx(k)
        assert m["bonus_rate"] == 1.0
        assert all(a == 1.0 for a in m["alpha_n"])


class TestLoadRecords:
    """The shared frozen-record loader (moved here so the loop, the bench,
    and this CLI share exactly one definition): committed frozen parquets
    are the GPU host's only record source (data/ is gitignored)."""

    def test_loads_frozen_parquet_with_limit(self):
        recs = load_records(str(FROZEN / "xlam_eval.parquet"), limit=3)
        assert len(recs) == 3
        # frozen order: the first-N subset must be a prefix of the full set
        assert recs[0]["query_id"] == load_records(str(FROZEN / "xlam_eval.parquet"))[0]["query_id"]
        for r in recs:
            assert r["input_ids"] and r["regions"] and r["labels"]
            assert 0 < r["n_prompt_tokens"] < len(r["input_ids"])

    def test_loads_save_to_disk_dir(self):
        # the local processed datasets (same loader form as the frozen
        # parquets; both consumers depend on it)
        recs = load_records(str(REPO / "data" / "processed" / "xlam" / "eval"))
        assert len(recs) == 500

    def test_empty_records_fail_loudly(self, tmp_path):
        import pyarrow.parquet as pq
        import pyarrow as pa

        p = tmp_path / "empty.parquet"
        pq.write_table(pa.table({"query_id": []}), p)
        with pytest.raises(SystemExit):
            load_records(str(p))

    @pytest.mark.skipif(
        not (FROZEN / "tb_eval.parquet").exists(),
        reason="frozen/ not built",
    )
    def test_frozen_parquet_roundtrip_matches_processed(self):
        # TB-500 frozen parquet must equal the local processed build —
        # the frozen file IS what the GPU day measures on
        frozen_recs = load_records(str(FROZEN / "tb_eval.parquet"))
        local_recs = load_records(str(REPO / "data" / "processed" / "toolbench" / "eval"))
        assert [r["query_id"] for r in frozen_recs] == [r["query_id"] for r in local_recs]
        assert frozen_recs[0]["input_ids"] == local_recs[0]["input_ids"]
