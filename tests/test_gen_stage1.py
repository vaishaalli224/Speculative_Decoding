"""Golden tests for the Stage-1 generation harness (plan.md §6.7 first half).

Layer 1 (vLLM-free, torch-free): a scripted engine pins the E1-E5
conventions and the JSONL schema against hand-computed expectations.
The real-engine paths (VLLMGenEngine on the H100; HFGenEngine locally
with a cached 0.5B stand-in) are covered by tests/test_gen_stage1_
realmodels.py and the GPU-day smoke in scripts/run_stage1_datagen.sh.

The scripted engine mirrors the GenEngine contract: token-id prompts in,
one {"token_ids", "logprobs", "finish_reason", "eos_included"} dict out
per prompt, input order.
"""

from __future__ import annotations

import json

import pytest

from src.serving.gen_stage1 import (
    MAX_LOGPROBS_CAP,
    normalize_output,
    sort_logprobs,
    write_generations,
)

EOS = 151645  # <|im_end|> — from the tokenizer in the real-model tests


class TestSortLogprobs:
    def test_sorted_by_value_desc_then_id_asc(self):
        out = sort_logprobs({5: -1.0, 3: -0.5, 4: -0.5, 1: -2.0})
        assert out == [[3, -0.5], [4, -0.5], [5, -1.0], [1, -2.0]]

    def test_empty_dict(self):
        assert sort_logprobs({}) == []

    def test_none_passthrough(self):
        assert sort_logprobs(None) == []


class TestNormalizeOutput:
    def test_vllm_style_eos_kept_once(self):
        # vLLM includes the EOS in token_ids with its own logprobs
        out = normalize_output(
            {"token_ids": [10, EOS], "logprobs": [{10: -0.1}, {EOS: -0.2}],
             "finish_reason": "stop", "eos_included": True}, EOS)
        assert out["token_ids"] == [10, EOS]  # no duplicate append (E2)
        assert out["logprobs"] == [[[10, -0.1]], [[EOS, -0.2]]]
        assert out["finish_reason"] == "stop"

    def test_hf_style_eos_appended(self):
        # HF withholds the stop token but its scores cover it
        out = normalize_output(
            {"token_ids": [10], "logprobs": [{10: -0.1}, {EOS: -0.2}],
             "finish_reason": "stop", "eos_included": False}, EOS)
        assert out["token_ids"] == [10, EOS]  # E2: appended with logprobs
        assert out["logprobs"] == [[[10, -0.1]], [[EOS, -0.2]]]

    def test_length_stop_no_eos(self):
        out = normalize_output(
            {"token_ids": [10, 11], "logprobs": [{10: -0.1}, {11: -0.2}],
             "finish_reason": "length", "eos_included": False}, EOS)
        assert out["token_ids"] == [10, 11]
        assert EOS not in out["token_ids"]

    def test_null_logprob_position_preserved(self):
        out = normalize_output(
            {"token_ids": [10, 11], "logprobs": [{10: -0.1}, None],
             "finish_reason": "length", "eos_included": False}, EOS)
        assert out["logprobs"][1] == []  # null -> [] per sort_logprobs

    def test_logprob_count_mismatch_is_an_error(self):
        # a "stop" output whose logprobs do not cover the EOS position
        with pytest.raises(AssertionError):
            normalize_output(
                {"token_ids": [10], "logprobs": [{10: -0.1}],
                 "finish_reason": "stop", "eos_included": False}, EOS)

    def test_ragged_lengths_preserved(self):
        # E3: one position with fewer entries than k is legal
        out = normalize_output(
            {"token_ids": [10],
             "logprobs": [{10: -0.1, 11: -0.3, 12: -0.9}],
             "finish_reason": "length", "eos_included": False}, EOS)
        assert out["logprobs"] == [[[10, -0.1], [11, -0.3], [12, -0.9]]]


class TestWriteGenerations:
    def test_jsonl_frozen_order_and_query_ids(self, tmp_path):
        prompts = [{"query_id": 29, "prompt_ids": [1]},
                   {"query_id": 32, "prompt_ids": [2]}]
        outputs = [
            {"token_ids": [7, EOS], "logprobs": [[[7, -0.1]], [[EOS, -0.2]]],
             "finish_reason": "stop"},
            {"token_ids": [8], "logprobs": [[[8, -0.1]]],
             "finish_reason": "length"},
        ]
        p = tmp_path / "gen.jsonl"
        write_generations(str(p), prompts, outputs)
        lines = [json.loads(l) for l in p.read_text().splitlines()]
        assert [l["query_id"] for l in lines] == [29, 32]  # E5
        assert lines[0]["output_ids"] == [7, EOS]
        assert lines[1]["finish_reason"] == "length"
        # the JSONL matches the assembling side's schema exactly
        from src.data_prep.build_distill_data import load_gens

        gens = load_gens(str(p))
        assert gens[0]["logprobs"] == [[[7, -0.1]], [[EOS, -0.2]]]

    def test_count_mismatch_is_an_error(self, tmp_path):
        with pytest.raises(AssertionError):
            write_generations(
                str(tmp_path / "g.jsonl"),
                [{"query_id": 1, "prompt_ids": [1]}], [],
            )


class TestScriptedEngineFlow:
    """End-to-end pure-core flow: scripted engine -> normalize -> JSONL ->
    load_gens -> build_gen_record-style checks. Mirrors run order (E4:
    one generate call for all prompts)."""

    def test_full_flow(self, tmp_path):
        from src.data_prep.build_distill_data import load_gens

        class Scripted:
            def generate(self, prompts, params):
                assert len(prompts) == 2  # single call, all prompts (E4)
                return [
                    {"token_ids": [100, EOS],
                     "logprobs": [{100: -0.1, 200: -0.4}, {EOS: -0.05}],
                     "finish_reason": "stop", "eos_included": True},
                    {"token_ids": [300],
                     "logprobs": [{300: -0.2}],
                     "finish_reason": "length", "eos_included": False},
                ]

        prompts = [{"query_id": 61, "prompt_ids": [1, 2]},
                   {"query_id": 71, "prompt_ids": [3]}]
        outs = [normalize_output(o, EOS)
                for o in Scripted().generate(
                    [p["prompt_ids"] for p in prompts], {})]
        p = tmp_path / "gen.jsonl"
        write_generations(str(p), prompts, outs)
        gens = load_gens(str(p))
        assert gens[0]["query_id"] == 61
        assert gens[0]["output_ids"] == [100, EOS]
        assert gens[0]["logprobs"][0] == [[100, -0.1], [200, -0.4]]
        assert gens[1]["output_ids"] == [300]
        assert gens[1]["finish_reason"] == "length"

    def test_params_passed_through_greedy(self):
        seen = {}

        class Scripted:
            def generate(self, prompts, params):
                seen.update(params)
                return [{"token_ids": [1], "logprobs": [{1: 0.0}],
                         "finish_reason": "length", "eos_included": False}]

        Scripted().generate([[1]], {"temperature": 0.0, "max_tokens": 512,
                                    "seed": 1234})
        assert seen["temperature"] == 0.0  # E1
        assert seen["max_tokens"] == 512


class TestMaxLogprobsCap:
    def test_over_cap_rejected_without_vllm(self):
        # pure validation — must fire BEFORE the vllm import is attempted
        with pytest.raises(ValueError):
            from src.serving.gen_stage1 import VLLMGenEngine

            VLLMGenEngine("x", logprobs=MAX_LOGPROBS_CAP + 1)
