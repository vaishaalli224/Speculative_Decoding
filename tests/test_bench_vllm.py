"""Golden tests for the vLLM bench harness (§6.6).

Layer 1 (this file) is vLLM-free and torch-free: a scripted fake engine
pins the harness's B1–B7 conventions against hand-computed metrics. The
real-engine path (VLLMEngine) runs only on the H100 host — vLLM has no
macOS wheels — and is exercised by the GPU-day step-0 smoke test, not here.

The scripted engine mirrors VLLMEngine's contract: token-id prompts in,
token-id lists out, one output list per input prompt, input order.
"""

from __future__ import annotations

import json

import pytest

from src.serving.bench_vllm import (
    chunked,
    exactness_compare,
    load_prompts,
    run_bench,
    read_outputs,
    write_outputs,
    append_metrics,
)
from src.serving.spec_configs import (
    METHODS,
    NGRAM_LOOKUP_MAX,
    NGRAM_LOOKUP_MIN,
    build_spec_config,
    ngram_config,
    draft_model_config,
    validate_spec_config,
)

DRAFT = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedEngine:
    """Deterministic 'engine': generates exactly max_tokens new token ids
    per prompt (deterministic function of the prompt — same prompt, same
    completion), recording every call for B1/B2 assertions. `clock`, when
    given, is ticked 1.0s per generate call so wall accounting is asserted
    without sleeping; `slow_calls` maps a 0-based call index to extra tick
    seconds (a 'slow' run for median tests)."""

    def __init__(self, cap: int | None = None, clock=None, slow_calls=None):
        self.calls: list[list[list[int]]] = []
        self.cap = cap
        self.clock = clock
        self.slow_calls = slow_calls or {}

    def generate(self, prompts, params):
        self.calls.append([list(p) for p in prompts])
        if self.clock is not None:
            self.clock.tick(1.0 + self.slow_calls.get(len(self.calls) - 1, 0.0))
        outs = []
        for p in prompts:
            n = params["max_tokens"]
            if self.cap is not None:
                n = min(n, self.cap)
            base = (sum(p) % 900) + 1
            outs.append([(base + j) % 10_000 for j in range(n)])
        return outs


class TimingWrapper:
    """Deterministic clock: starts at 100.0. The engine calls tick() per
    generate call (advancing the clock); run_bench calls time_fn (=this
    __call__) only for t0/t1. wall = sum of the ticked call durations."""

    def __init__(self):
        self.t = 100.0

    def tick(self, seconds: float = 1.0) -> None:
        self.t += seconds

    def __call__(self):
        return self.t


def _prompts(n=4, plen=3):
    return [
        {"query_id": i, "prompt_ids": [10 * i + j for j in range(plen)]}
        for i in range(n)
    ]


PARAMS = {"max_tokens": 4, "temperature": 0.0, "seed": 1234}


# ---------------------------------------------------------------------------
# B1 chunking
# ---------------------------------------------------------------------------


class TestChunking:
    def test_chunks_in_order_remainder_last(self):
        chunks = list(chunked(list(range(10)), 3))
        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_exact_multiple_no_remainder(self):
        assert list(chunked(list(range(6)), 3)) == [[0, 1, 2], [3, 4, 5]]

    def test_batch_larger_than_list(self):
        assert list(chunked([1, 2], 8)) == [[1, 2]]

    def test_batch_must_be_positive(self):
        with pytest.raises(ValueError):
            list(chunked([1], 0))


class TestRunBenchBatching:
    def test_batch_b_sends_b_prompts_per_call(self):
        """B1: 'batch 8' means the engine never sees more than 8 concurrent
        requests; one call per chunk, chunks in frozen order."""
        eng = ScriptedEngine()
        run_bench(eng, _prompts(10), PARAMS, batch=8, runs=1, warmup=0)
        assert [len(c) for c in eng.calls] == [8, 2]
        # chunk contents stay in frozen order
        assert eng.calls[0][0] == _prompts(10)[0]["prompt_ids"]
        assert eng.calls[1][0] == _prompts(10)[8]["prompt_ids"]

    def test_batch_1_one_call_per_prompt(self):
        eng = ScriptedEngine()
        run_bench(eng, _prompts(3), PARAMS, batch=1, runs=1, warmup=0)
        assert [len(c) for c in eng.calls] == [1, 1, 1]

    def test_warmup_untimed_and_excluded_from_calls(self):
        """B2: warmup prompts run once, before run 1, and their outputs are
        discarded — the timed calls start from prompt 0 again."""
        eng = ScriptedEngine()
        run_bench(eng, _prompts(6), PARAMS, batch=2, runs=2, warmup=2)
        # call 0 = warmup chunk (2 prompts), then 2 runs x 3 chunks of 2
        assert [len(c) for c in eng.calls] == [2] + [2, 2, 2] * 2
        assert eng.calls[1][0] == _prompts(6)[0]["prompt_ids"]

    def test_all_prompts_covered_once_per_run(self):
        eng = ScriptedEngine()
        m = run_bench(eng, _prompts(9), PARAMS, batch=4, runs=1, warmup=0)
        seen = [p for c in eng.calls for p in c]
        assert sorted(map(tuple, seen)) == sorted(
            tuple(p["prompt_ids"]) for p in _prompts(9)
        )
        assert m["n_prompts"] == 9


# ---------------------------------------------------------------------------
# B2-B4 timing, medians, outputs
# ---------------------------------------------------------------------------


class TestTimingAndMedians:
    def test_wall_is_sum_of_generate_calls_only(self):
        """B2: wall_s for a run = the sum of its chunk-call durations —
        nothing else. 6 prompts / batch 2 = 3 chunk calls x 1.0s = 3.0s."""
        clock = TimingWrapper()
        eng = ScriptedEngine(clock=clock)
        m = run_bench(eng, _prompts(6), PARAMS, batch=2, runs=1,
                      warmup=0, time_fn=clock)
        assert m["per_run"][0]["wall_s"] == pytest.approx(3.0)
        assert len(eng.calls) == 3

    def test_warmup_is_outside_the_wall(self):
        """B2: warmup calls are untimed — the run's wall excludes them."""
        clock = TimingWrapper()
        eng = ScriptedEngine(clock=clock)
        m = run_bench(eng, _prompts(6), PARAMS, batch=2, runs=1, warmup=2,
                      time_fn=clock)
        assert len(eng.calls) == 4  # 1 warmup + 3 timed
        assert m["per_run"][0]["wall_s"] == pytest.approx(3.0)

    def test_median_wall_across_runs(self):
        """B4: 3 runs, wall = median. 4 prompts / batch 4 = 1 call per run;
        slow_calls={2: 9.0} makes run 3 (call idx 2) the slow one — the
        median must be a fast run's wall."""
        clock = TimingWrapper()
        eng = ScriptedEngine(clock=clock, slow_calls={2: 9.0})
        m = run_bench(eng, _prompts(4), PARAMS, batch=4, runs=3, warmup=0,
                      time_fn=clock)
        walls = [r["wall_s"] for r in m["per_run"]]
        assert walls == [pytest.approx(1.0), pytest.approx(1.0), pytest.approx(10.0)]
        assert m["median_wall_s"] == pytest.approx(1.0)

    def test_median_gen_tok_s_across_runs(self):
        """B4: 3 runs, gen_tok_s = median. Slow run 3 halves its rate; the
        median must be a fast run's rate."""
        clock = TimingWrapper()
        eng = ScriptedEngine(clock=clock, slow_calls={2: 1.0})  # run 3: 2s
        m = run_bench(eng, _prompts(4), PARAMS, batch=4, runs=3, warmup=0,
                      time_fn=clock)
        gtok = [r["gen_tok_s"] for r in m["per_run"]]
        assert gtok[:2] == [pytest.approx(16.0), pytest.approx(16.0)]
        assert gtok[2] == pytest.approx(8.0)
        assert m["median_gen_tok_s"] == pytest.approx(16.0)

    def test_gen_tok_s_excludes_prompt_tokens(self):
        """B3: gen_tok_s = completion tokens / wall. 4 prompts x 3-token
        prompts = 12 prompt tokens; max_tokens=4 -> 16 gen tokens; 1 chunk
        call = 1.0s wall."""
        clock = TimingWrapper()
        eng = ScriptedEngine(clock=clock)
        m = run_bench(eng, _prompts(4, plen=3), PARAMS, batch=4, runs=1,
                      warmup=0, time_fn=clock)
        r = m["per_run"][0]
        assert r["gen_tokens"] == 16
        assert r["prompt_tokens"] == 12
        assert r["gen_tok_s"] == pytest.approx(16 / 1.0)
        assert r["total_tok_s"] == pytest.approx(28 / 1.0)
        assert m["median_gen_tok_s"] == pytest.approx(16.0)

    def test_outputs_are_run1_and_complete(self):
        """B4: saved outputs are run 1's, one entry per prompt in order."""
        eng = ScriptedEngine()
        m = run_bench(eng, _prompts(5), PARAMS, batch=2, runs=3, warmup=0)
        assert [o["query_id"] for o in m["outputs"]] == [0, 1, 2, 3, 4]
        assert len(eng.calls) == 9  # 3 runs x 3 chunks (5 = 2+2+1)
        # run 1's last chunk (call idx 2) holds prompt qid 4
        p4 = _prompts(5)[4]["prompt_ids"]
        base = (sum(p4) % 900) + 1
        assert m["outputs"][4]["output_ids"] == [
            (base + j) % 10_000 for j in range(4)
        ]
        # every output is max_tokens long (no EOS in the fake)
        assert all(len(o["output_ids"]) == 4 for o in m["outputs"])

    def test_runs_must_be_positive_and_prompts_nonempty(self):
        with pytest.raises(ValueError):
            run_bench(ScriptedEngine(), [], PARAMS)
        with pytest.raises(ValueError):
            run_bench(ScriptedEngine(), _prompts(1), PARAMS, runs=0)

    def test_engine_output_shape_violations_caught(self):
        """The adapter contract: one token-id list per prompt, input order."""
        class BadLen(ScriptedEngine):
            def generate(self, prompts, params):
                return [[1]] * (len(prompts) - 1)

        with pytest.raises(AssertionError):
            run_bench(BadLen(), _prompts(4), PARAMS, batch=4, runs=1)
        class BadType(ScriptedEngine):
            def generate(self, prompts, params):
                return ["text"] * len(prompts)

        with pytest.raises(AssertionError):
            run_bench(BadType(), _prompts(2), PARAMS, batch=2, runs=1)


# ---------------------------------------------------------------------------
# B6 exactness
# ---------------------------------------------------------------------------


def _outs(qid, ids):
    return {"query_id": qid, "output_ids": ids}


class TestExactness:
    def test_identical_sequences_pass(self):
        a = [_outs(0, [1, 2, 3]), _outs(1, [9, 8])]
        rep = exactness_compare(a, [dict(o) for o in a])
        assert rep["all_exact"]
        assert rep["n_exact"] == 2
        assert rep["n_mismatch"] == 0

    def test_divergence_at_position_reported(self):
        a = [_outs(0, [1, 2, 3, 4])]
        b = [_outs(0, [1, 2, 9, 4])]
        rep = exactness_compare(a, b)
        assert not rep["all_exact"]
        assert rep["mismatches"][0]["first_divergence"] == 2
        assert rep["mismatches"][0]["len_baseline"] == 4
        assert rep["mismatches"][0]["len_spec"] == 4

    def test_length_difference_is_mismatch_with_first_div(self):
        """Equal ids => equal lengths; a truncation is a mismatch whose first
        divergence is at the shorter length (no token pair differs)."""
        a = [_outs(0, [1, 2, 3])]
        b = [_outs(0, [1, 2, 3, 4])]
        rep = exactness_compare(a, b)
        assert rep["n_mismatch"] == 1
        assert rep["mismatches"][0]["first_divergence"] == 3

    def test_order_desync_is_hard_error(self):
        with pytest.raises(AssertionError):
            exactness_compare([_outs(0, [1])], [_outs(1, [1])])

    def test_length_mismatch_of_lists_is_hard_error(self):
        with pytest.raises(AssertionError):
            exactness_compare([_outs(0, [1])], [_outs(0, [1]), _outs(1, [2])])

    def test_mixed_prompts_per_prompt_verdicts(self):
        a = [_outs(0, [1]), _outs(1, [2, 2]), _outs(2, [3])]
        b = [_outs(0, [1]), _outs(1, [2, 3]), _outs(2, [3])]
        rep = exactness_compare(a, b)
        assert rep["n_exact"] == 2
        assert rep["n_mismatch"] == 1
        assert rep["mismatches"][0]["query_id"] == 1


# ---------------------------------------------------------------------------
# B5 prompt loading (real frozen parquets)
# ---------------------------------------------------------------------------


class TestLoadPrompts:
    def test_default_one_prompt_per_record_turn0_context(self):
        prompts = load_prompts("frozen/xlam_eval.parquet", limit=3)
        assert len(prompts) == 3
        # prompt = ids up to the first assistant token (n_prompt_tokens)
        import pyarrow.parquet as pq

        t = pq.read_table("frozen/xlam_eval.parquet").to_pylist()[:3]
        assert [p["query_id"] for p in prompts] == [r["query_id"] for r in t]
        for p, r in zip(prompts, t):
            assert p["prompt_ids"] == r["input_ids"][: r["n_prompt_tokens"]]
            assert len(p["prompt_ids"]) == r["n_prompt_tokens"]

    def test_limit_is_first_n_frozen_order(self):
        p3 = load_prompts("frozen/xlam_eval.parquet", limit=3)
        p5 = load_prompts("frozen/xlam_eval.parquet", limit=5)
        assert p3 == p5[:3]

    def test_per_turn_expands_tb500(self):
        """B5/C6: TB-500 records expand to one teacher-forced prompt per
        assistant turn; turn starts match the analyzer's own helper
        (imported, so they cannot desync). The "turn" key exists only on
        multi-turn records (C6); the first 5 TB records include one
        single-turn conversation, which must come through unchanged."""
        from src.analysis.eval_acceptance import _turn_starts_from_labels

        prompts = load_prompts("frozen/tb_eval.parquet", limit=5, per_turn=True)
        import pyarrow.parquet as pq

        t = pq.read_table("frozen/tb_eval.parquet").to_pylist()[:5]
        expected = []
        for r in t:
            starts = _turn_starts_from_labels(r["labels"])
            for turn_i, s in enumerate(starts):
                expected.append((
                    r["query_id"],
                    turn_i if len(starts) > 1 else None,
                    tuple(r["input_ids"][:s]),
                ))
        got = [(p["query_id"], p.get("turn"), tuple(p["prompt_ids"])) for p in prompts]
        assert got == expected
        multi = [p for p in prompts if p.get("turn") is not None]
        assert multi and any(p["turn"] > 0 for p in multi)
        single = [p for p in prompts if "turn" not in p]
        assert len(single) == 1  # the single-turn record among the first 5

    def test_per_turn_single_turn_records_match_default(self):
        """Single-turn records (xLAM) expand to exactly the default prompt —
        per_turn must not change the turn-0 context."""
        per_turn = load_prompts("frozen/xlam_eval.parquet", limit=3, per_turn=True)
        default = load_prompts("frozen/xlam_eval.parquet", limit=3)
        assert per_turn == default


# ---------------------------------------------------------------------------
# B7 IO helpers
# ---------------------------------------------------------------------------


class TestIO:
    def test_outputs_roundtrip(self, tmp_path):
        outs = [_outs(0, [1, 2]), _outs(1, [3])]
        p = str(tmp_path / "sub" / "o.jsonl")
        write_outputs(p, outs)
        assert read_outputs(p) == outs

    def test_metrics_appends_one_json_line_per_run(self, tmp_path):
        p = str(tmp_path / "m.jsonl")
        append_metrics(p, {"tag": "a", "gen_tok_s": 1.0})
        append_metrics(p, {"tag": "b"})
        lines = [json.loads(l) for l in open(p)]
        assert [l["tag"] for l in lines] == ["a", "b"]
        assert lines[0]["gen_tok_s"] == 1.0


# ---------------------------------------------------------------------------
# spec_configs (§1.5/§4.6)
# ---------------------------------------------------------------------------


class TestSpecConfigs:
    def test_draft_model_shape_verbatim(self):
        cfg = draft_model_config(DRAFT, num_speculative_tokens=5)
        assert cfg == {
            "method": "draft_model",
            "model": DRAFT,
            "num_speculative_tokens": 5,
            "max_model_len": 16384,
        }

    def test_ngram_shape_verbatim(self):
        cfg = ngram_config(num_speculative_tokens=4)
        assert cfg == {
            "method": "ngram",
            "num_speculative_tokens": 4,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
        }

    def test_build_dispatch(self):
        assert build_spec_config("ar") is None
        assert build_spec_config("draft_model", DRAFT, 3)["num_speculative_tokens"] == 3
        assert build_spec_config("ngram", num_speculative_tokens=7)["method"] == "ngram"

    def test_ngram_k7_for_sweep(self):
        """§4.6: n-gram sweeps the same k grid (3/5/7) as the drafts."""
        assert build_spec_config("ngram", num_speculative_tokens=7)["num_speculative_tokens"] == 7

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            build_spec_config("eagle3")
        assert "eagle3" not in METHODS

    def test_rejects_bad_k(self):
        with pytest.raises(ValueError):
            draft_model_config(DRAFT, 0)
        with pytest.raises(ValueError):
            ngram_config(0)
        with pytest.raises(ValueError):
            draft_model_config(DRAFT, True)  # bool is not a k

    def test_rejects_missing_draft_model(self):
        with pytest.raises(ValueError):
            draft_model_config("", 5)
        with pytest.raises(ValueError):
            draft_model_config(None, 5)  # type: ignore[arg-type]

    def test_rejects_bad_ngram_window(self):
        with pytest.raises(ValueError):
            ngram_config(4, prompt_lookup_min=6, prompt_lookup_max=5)
        with pytest.raises(ValueError):
            ngram_config(4, prompt_lookup_min=0)

    def test_ngram_must_not_carry_model_key(self):
        with pytest.raises(ValueError):
            validate_spec_config({"method": "ngram", "num_speculative_tokens": 4,
                                  "model": DRAFT})

    def test_draft_max_model_len_configurable(self):
        cfg = draft_model_config(DRAFT, 5, max_model_len=8192)
        assert cfg["max_model_len"] == 8192

    def test_lookup_defaults_match_plan(self):
        assert (NGRAM_LOOKUP_MIN, NGRAM_LOOKUP_MAX) == (2, 5)
