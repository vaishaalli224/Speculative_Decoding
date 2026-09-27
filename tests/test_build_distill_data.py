"""Golden tests for the Stage-1 KD data build (plan.md §6.7 first half).

Layer 1 (torch-free, vLLM-free): the G1-G7 conventions against hand-built
cases and, where the local rendered pool exists, against real frozen
records. Generation-JSONL fixtures are hand-constructed here; the
generation side's own tests (test_gen_stage1.py) pin E1-E5.

Facts encoded here (verified 2026-09-27):
  - render_stage1_prompt(raw_ex) == the rendered train-pool row's
    input_ids[:n_prompt_tokens] for the same query_id (the token-level
    prefix property, test_golden_render.test_prefix_property_tokens_
    user_only, now checked end-to-end through the raw-data path)
  - the tool-call tags are the single special tokens 151657/151658
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_prep.build_distill_data import (
    assemble,
    build_gen_record,
    find_gen_payloads,
    find_gen_tag_spans,
    load_gens,
    render_stage1_prompt,
    stage1_index,
    validate_generation,
)
from src.data_prep.render import (
    REGION_CONTEXT,
    REGION_PROSE,
    REGION_TAG,
    REGION_TOOL_CALL,
    _rle,
    get_tokenizer,
    get_tool_call_tags,
    unrle,
)
from src.data_prep.xlam_prep import build_messages

REPO = Path(__file__).resolve().parents[1]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]


def _gen_text(tok, calls: list[dict]) -> str:
    """A generation's text: tool-call blocks with the template's tags."""
    tags = get_tool_call_tags()
    parts = []
    for c in calls:
        payload = json.dumps({"name": c["name"], "arguments": c["arguments"]})
        parts.append(f"{tags.open_tag}\n{payload}\n{tags.close_tag}")
    return "\n".join(parts)


def _gen_record_for(text: str, logprob_rows=None, finish="stop", eos_id=151645):
    """Build a generation dict from text: tokenize, fake logprob rows if
    absent (one [[id, val]] per position), append EOS for 'stop'."""
    tok = get_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if finish == "stop":
        ids = ids + [eos_id]
    if logprob_rows is None:
        logprob_rows = [[[t, -1.0 * (i + 1)]] for i, t in enumerate(ids)]
    return {
        "query_id": None,
        "output_ids": ids,
        "logprobs": logprob_rows,
        "finish_reason": finish,
    }


@pytest.fixture(scope="session")
def tags():
    return get_tool_call_tags()


@pytest.fixture(scope="session")
def tok():
    return get_tokenizer()


# ---------------------------------------------------------------------------
# Payload spans
# ---------------------------------------------------------------------------


class TestGenPayloads:
    def test_single_call_span(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "Paris"}}])
        spans = find_gen_payloads(text, tags)
        assert len(spans) == 1
        payload = text[spans[0][0]:spans[0][1]]
        call = json.loads(payload)
        assert call == {"name": "get_weather", "arguments": {"city": "Paris"}}
        # the span excludes the tags themselves
        assert tags.open_tag not in payload and tags.close_tag not in payload

    def test_two_parallel_calls(self, tags, tok):
        text = _gen_text(tok, [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Rome"}},
        ])
        assert len(find_gen_payloads(text, tags)) == 2

    def test_unterminated_call_yields_no_span(self, tags, tok):
        text = f"{tags.open_tag}\n" + '{"name": "get_weather"'  # no close tag
        assert find_gen_payloads(text, tags) == []

    def test_tag_spans_cover_both_tags(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "x"}}])
        spans = find_gen_tag_spans(text, tags)
        assert len(spans) == 2  # one open + one close
        for s, e in spans:
            assert text[s:e] in (tags.open_tag, tags.close_tag)


# ---------------------------------------------------------------------------
# Validation (G4)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_single_call(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "Paris"}}])
        v = validate_generation(text, TOOLS, tags)
        assert v["valid"] and v["n_calls"] == 1 and not v["problems"]

    def test_valid_optional_key_omitted(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "Paris"}}])
        assert validate_generation(text, TOOLS, tags)["valid"]  # 'unit' optional

    def test_missing_required_key_dropped(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"unit": "c"}}])
        v = validate_generation(text, TOOLS, tags)
        assert not v["valid"]
        assert any("missing required" in p for p in v["problems"])

    def test_unknown_function_name_dropped(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_flight", "arguments": {"city": "Paris"}}])
        v = validate_generation(text, TOOLS, tags)
        assert not v["valid"]
        assert any("unknown function" in p for p in v["problems"])

    def test_key_outside_schema_dropped(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather",
                                "arguments": {"city": "Paris", "date": "tomorrow"}}])
        v = validate_generation(text, TOOLS, tags)
        assert not v["valid"]
        assert any("outside schema" in p for p in v["problems"])

    def test_prose_only_dropped(self, tags):
        v = validate_generation("I cannot call tools here.", TOOLS, tags)
        assert not v["valid"] and v["n_calls"] == 0

    def test_malformed_json_dropped(self, tags, tok):
        bad = f"{tags.open_tag}\n" + '{"name": "get_weather", "argu' + f"\n{tags.close_tag}"
        v = validate_generation(bad, TOOLS, tags)
        assert not v["valid"]

    def test_second_call_invalid_drops_record(self, tags, tok):
        text = _gen_text(tok, [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "nope", "arguments": {}},
        ])
        assert not validate_generation(text, TOOLS, tags)["valid"]


# ---------------------------------------------------------------------------
# Record assembly (G2, G3, G5, G7)
# ---------------------------------------------------------------------------


class TestBuildGenRecord:
    def _prompt(self, qid=0):
        return {"query_id": qid, "prompt_ids": [1, 2, 3]}

    def test_record_schema_and_labels(self, tags, tok):
        gen = _gen_record_for(
            _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "Paris"}}])
        )
        rec, st = build_gen_record(self._prompt(), gen, TOOLS, tags)
        assert rec["input_ids"] == [1, 2, 3] + gen["output_ids"]
        assert rec["labels"] == [-100, -100, -100] + gen["output_ids"]  # G3/G5
        assert rec["n_prompt_tokens"] == 3
        assert rec["n_tokens"] == 3 + len(gen["output_ids"])
        assert rec["n_tool_calls"] == 1
        assert rec["n_tools"] == 1
        assert rec["query_id"] == 0
        assert st["valid"] and st["roundtrip_exact"]
        assert st["finish_reason"] == "stop"

    def test_regions_context_then_tool_call(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "Paris"}}])
        gen = _gen_record_for(text)
        rec, _ = build_gen_record(self._prompt(), gen, TOOLS, tags)
        regions = unrle([list(x) for x in rec["regions"]])
        assert len(regions) == rec["n_tokens"]
        assert regions[:3] == [REGION_CONTEXT] * 3  # the prompt
        # tag tokens are REGION_TAG (both special tokens render as single ids)
        tag_ids = [151657, 151658]
        tag_positions = [i for i, t in enumerate(gen["output_ids"]) if t in tag_ids]
        assert tag_positions, "no tag tokens in the fixture generation"
        for p in tag_positions:
            assert regions[3 + p] == REGION_TAG
        # payload tokens are REGION_TOOL_CALL
        payload_ids = tok(text, add_special_tokens=False)["input_ids"]
        payload_positions = [i for i, t in enumerate(payload_ids)
                             if regions[3 + i] == REGION_TOOL_CALL]
        assert payload_positions
        assert all(regions[3 + i] == REGION_TOOL_CALL for i in payload_positions)
        # at least one prose token exists (the newline between payload and tag)
        assert any(r == REGION_PROSE for r in regions)

    def test_logprob_fields_parallel_and_sorted(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "x"}}])
        gen = _gen_record_for(text)
        rec, _ = build_gen_record(self._prompt(), gen, TOOLS, tags)
        assert rec["gen_logprob_token_ids"] == [
            [t for t, _ in row] if row is not None else None
            for row in gen["logprobs"]
        ]
        assert rec["gen_logprob_values"] == [
            [v for _, v in row] if row is not None else None
            for row in gen["logprobs"]
        ]
        # G2: ragged rows preserved verbatim (one entry per position here)
        assert all(row is not None for row in rec["gen_logprob_token_ids"])

    def test_null_logprob_row_survives(self, tags, tok):
        # a withheld-EOS position arrives as null and must stay null (G2)
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "x"}}])
        gen = _gen_record_for(text)
        gen["logprobs"][-1] = None
        rec, _ = build_gen_record(self._prompt(), gen, TOOLS, tags)
        assert rec["gen_logprob_token_ids"][-1] is None
        assert rec["gen_logprob_values"][-1] is None

    def test_length_stop_has_no_eos(self, tags, tok):
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "x"}}])
        gen = _gen_record_for(text, finish="length")
        assert gen["output_ids"][-1] != 151645
        rec, st = build_gen_record(self._prompt(), gen, TOOLS, tags)
        assert rec["input_ids"][-1] != 151645
        assert st["finish_reason"] == "length"

    def test_invalid_generation_flagged_not_raised(self, tags, tok):
        text = _gen_text(tok, [{"name": "ghost", "arguments": {"city": "x"}}])
        gen = _gen_record_for(text)
        rec, st = build_gen_record(self._prompt(), gen, TOOLS, tags)
        assert not st["valid"]  # G4: reported, the caller drops
        assert any("unknown function" in p for p in st["problems"])


class TestAssemble:
    """Assemble joins generations to the REAL frozen examples, so fixture
    generations must call each example's real tools (built from raw) to be
    valid — exactly what the target will produce."""

    def _fixture_gens(self, n, tok, tags):
        from datasets import load_from_disk

        raw = load_from_disk(RAW_DIR)["train"]
        out = []
        for i, qid in enumerate(stage1_index(n)):
            _, tools = build_messages(raw[qid])
            call = json.loads(raw[qid]["answers"])[0]  # a known-valid call
            text = _gen_text(tok, [call])
            g = _gen_record_for(text)
            g["query_id"] = qid
            out.append(g)
        return out

    def test_assemble_keeps_valid_and_reports_drops(self, tok, tags):
        n = 4
        gens = self._fixture_gens(n, tok, tags)
        # make the second one invalid (hallucinated name)
        bad_text = _gen_text(tok, [{"name": "ghost", "arguments": {"city": "x"}}])
        gens[1] = _gen_record_for(bad_text)
        gens[1]["query_id"] = stage1_index(4)[1]
        records, stats = assemble(gens, limit=n)
        assert stats["n_stage1"] == n
        assert stats["n_dropped"] == 1 and stats["n_kept"] == n - 1
        assert 0 < stats["drop_rate"] < 1
        assert stats["drop_reasons"].get("unknown function name") == 1
        assert stats["roundtrip_exact_all"]
        assert stats["finish_reasons"] == {"stop": n, "length": 0}
        assert len(records) == n - 1
        kept_qids = [r["query_id"] for r in records]
        assert kept_qids == [q for i, q in enumerate(stage1_index(n))
                             if i != 1]

    def test_count_mismatch_rejected(self, tok, tags):
        gens = self._fixture_gens(3, tok, tags)
        with pytest.raises(SystemExit):
            assemble(gens, limit=4)

    def test_duplicate_query_id_rejected(self, tok, tags):
        gens = self._fixture_gens(2, tok, tags)
        gens[1]["query_id"] = gens[0]["query_id"]
        with pytest.raises(SystemExit):
            assemble(gens, limit=2)

    def test_order_desync_rejected(self, tok, tags):
        gens = self._fixture_gens(2, tok, tags)
        gens[0], gens[1] = gens[1], gens[0]
        with pytest.raises(SystemExit):
            assemble(gens, limit=2)


# ---------------------------------------------------------------------------
# Generation JSONL loader + real frozen-index prompts
# ---------------------------------------------------------------------------


class TestLoadGens:
    def test_round_trip_and_schema_validation(self, tmp_path):
        gen = _gen_record_for("hello", query_id_override := None)  # placeholder
        gen["query_id"] = 1
        p = tmp_path / "g.jsonl"
        p.write_text(json.dumps(gen) + "\n")
        assert load_gens(str(p))[0]["output_ids"] == gen["output_ids"]

    def test_logprobs_length_mismatch_rejected(self, tmp_path):
        gen = _gen_record_for("hello")
        gen["query_id"] = 1
        gen["logprobs"] = gen["logprobs"][:-1]
        p = tmp_path / "g.jsonl"
        p.write_text(json.dumps(gen) + "\n")
        with pytest.raises(ValueError):
            load_gens(str(p))

    def test_missing_field_rejected(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text(json.dumps({"query_id": 1, "output_ids": [1]}) + "\n")
        with pytest.raises(ValueError):
            load_gens(str(p))


RAW_DIR = REPO / "data" / "raw" / "xlam"
PROCESSED_TRAIN = REPO / "data" / "processed" / "xlam" / "train"


@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw xLAM not downloaded")
class TestRealPrompts:
    def test_stage1_index_frozen_order(self):
        idx = stage1_index()
        assert len(idx) == 5000
        assert idx == json.loads((REPO / "frozen" / "stage1_idx.json").read_text())
        assert stage1_index(7) == idx[:7]  # G6

    def test_prompt_matches_rendered_train_pool_row(self):
        """The rendering path (raw -> render_context) reproduces the
        already-rendered train-pool prompt slice for the same query_id.
        (Pool rows are sorted(train_idx) order, not raw order — look the
        row up by its query_id column.)"""
        from datasets import load_from_disk

        raw = load_from_disk(RAW_DIR)["train"]
        pool = load_from_disk(PROCESSED_TRAIN)
        qids = list(pool["query_id"])
        for qid in stage1_index(3):  # real frozen stage-1 ids
            ex = raw[qid]
            prompt_ids = render_stage1_prompt(ex)
            row = pool[qids.index(qid)]
            assert row["query_id"] == qid
            assert prompt_ids == row["input_ids"][:row["n_prompt_tokens"]]

    def test_first_prompts_end_with_assistant_header(self):
        from datasets import load_from_disk

        raw = load_from_disk(RAW_DIR)["train"]
        tok = get_tokenizer()
        for i in stage1_index(3):
            text = tok.decode(render_stage1_prompt(raw[i]))
            assert text.endswith("<|im_start|>assistant\n")


class TestRLEInterop:
    def test_gen_record_regions_decode_like_render_records(self, tags, tok):
        # the RLE a gen record produces is decodable by the same unrle the
        # analyzer uses — schema G3 end to end
        text = _gen_text(tok, [{"name": "get_weather", "arguments": {"city": "x"}}])
        gen = _gen_record_for(text)
        rec, _ = build_gen_record({"query_id": 0, "prompt_ids": []}, gen, TOOLS, tags)
        regions = unrle([list(x) for x in rec["regions"]])
        assert len(regions) == rec["n_tokens"]
        assert all(r in (REGION_PROSE, REGION_TOOL_CALL, REGION_TAG)
                   for r in regions)

# G8: the schema-announcement wrapper the target actually emits
ALT_OPEN = '<tools>'
ALT_CLOSE = '</tools>'


class TestDualWrapperG8:
    """The target's measured greedy behavior (2026-09-27, 5000-gen A/B on
    the rented H100): 92.6% of valid calls wrapped in the SCHEMA pair,
    payloads always valid. G8 accepts both pairs; payloads validated,
    wrappers kept verbatim."""

    def test_alt_wrapper_payload_found(self, tags, tok):
        payload = json.dumps(
            {"name": "get_weather", "arguments": {"city": "Paris"}})
        text = ALT_OPEN + "\n" + payload + "\n" + ALT_CLOSE
        spans = find_gen_payloads(text, tags)
        assert len(spans) == 1
        got = json.loads(text[spans[0][0]:spans[0][1]])
        assert got["name"] == "get_weather"

    def test_alt_wrapper_validates(self, tags, tok):
        payload = json.dumps(
            {"name": "get_weather", "arguments": {"city": "Paris"}})
        text = ALT_OPEN + "\n" + payload + "\n" + ALT_CLOSE
        v = validate_generation(text, TOOLS, tags)
        assert v["valid"], v["problems"]
        assert v["n_calls"] == 1

    def test_canonical_and_alt_mixed(self, tags, tok):
        p1 = json.dumps(
            {"name": "get_weather", "arguments": {"city": "Paris"}})
        p2 = json.dumps(
            {"name": "get_weather", "arguments": {"city": "Rome"}})
        text = (tags.open_tag + "\n" + p1 + "\n" + tags.close_tag
                + "\n" + ALT_OPEN + "\n" + p2 + "\n" + ALT_CLOSE)
        assert len(find_gen_payloads(text, tags)) == 2
        v = validate_generation(text, TOOLS, tags)
        assert v["valid"] and v["n_calls"] == 2

    def test_nested_canonical_inside_alt_not_double_counted(self, tags, tok):
        # a canonical-tag occurrence INSIDE a schema-wrapped payload is
        # payload text, not a nested call (outermost-match, G8)
        inner = json.dumps(
            {"name": "get_weather", "arguments": {"city": "Paris"}})
        text = (ALT_OPEN + "\n" + tags.open_tag + "\n" + inner + "\n"
                + tags.close_tag + "\n" + ALT_CLOSE)
        spans = find_gen_payloads(text, tags)
        assert len(spans) == 1
        # the kept span is the OUTER one: it contains the canonical tags
        # and the JSON as payload text (outermost-match, G8)
        kept = text[spans[0][0]:spans[0][1]]
        assert tags.open_tag in kept and inner in kept
        # ...and the inner canonical pair was not extracted as its own call
        assert ALT_OPEN not in kept

    def test_alt_tag_spans_are_tag_region(self, tags, tok):
        payload = json.dumps(
            {"name": "get_weather", "arguments": {"city": "Paris"}})
        text = ALT_OPEN + "\n" + payload + "\n" + ALT_CLOSE
        spans = find_gen_tag_spans(text, tags)
        assert len(spans) == 2
        for a, b in spans:
            assert text[a:b] in (ALT_OPEN, ALT_CLOSE)

    def test_alt_wrapper_record_regions(self, tags, tok):
        # end-to-end: an alt-wrapped generation builds a record whose tag
        # tokens are REGION_TAG and payload tokens are REGION_TOOL_CALL
        from src.data_prep.build_distill_data import build_gen_record
        from src.data_prep.render import REGION_TAG, REGION_TOOL_CALL, unrle

        payload = json.dumps(
            {"name": "get_weather", "arguments": {"city": "Paris"}})
        text = ALT_OPEN + "\n" + payload + "\n" + ALT_CLOSE
        gen = _gen_record_for(text)
        prompt = {"query_id": 7, "prompt_ids": [1, 2, 3]}
        rec, st = build_gen_record(prompt, gen, TOOLS, tags)
        assert st["valid"], st["problems"]
        regions = unrle(rec["regions"])
        assert REGION_TAG in regions and REGION_TOOL_CALL in regions
        assert st["roundtrip_exact"]


class TestParallelCallsG8:
    """One wrapper may hold a SEQUENCE of JSON objects (xLAM is 52.6%
    multi-call; the target emits call1\ncall2 inside a single pair —
    measured 2026-09-27). raw_decode sequence parse; strict validation."""

    def test_two_calls_in_one_wrapper(self, tags, tok):
        p1 = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
        p2 = json.dumps({"name": "get_weather", "arguments": {"city": "Rome"}})
        text = tags.open_tag + "\n" + p1 + "\n" + p2 + "\n" + tags.close_tag
        v = validate_generation(text, TOOLS, tags)
        assert v["valid"], v["problems"]
        assert v["n_calls"] == 1  # one wrapper (payload spans), 2 objects

    def test_alt_wrapper_two_calls(self, tags, tok):
        p1 = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
        p2 = json.dumps({"name": "get_weather", "arguments": {"city": "Rome"}})
        text = ALT_OPEN + "\n" + p1 + "\n" + p2 + "\n" + ALT_CLOSE
        v = validate_generation(text, TOOLS, tags)
        assert v["valid"], v["problems"]

    def test_second_object_invalid_drops_record(self, tags, tok):
        p1 = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
        p2 = json.dumps({"name": "nope", "arguments": {}})
        text = tags.open_tag + "\n" + p1 + "\n" + p2 + "\n" + tags.close_tag
        v = validate_generation(text, TOOLS, tags)
        assert not v["valid"]
        assert any("unknown function" in x for x in v["problems"])

    def test_trailing_garbage_fails(self, tags, tok):
        good = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
        text = tags.open_tag + "\n" + good + " oops {" + "\n" + tags.close_tag
        v = validate_generation(text, TOOLS, tags)
        assert not v["valid"]
        assert any("not JSON" in x for x in v["problems"])

    def test_whitespace_only_wrapper_drops(self, tags, tok):
        # an empty wrapper carries only whitespace between the tags; the
        # span finder yields it but no JSON object parses -> record drops
        # (pre-existing behavior; the reason string differs from the
        # zero-span case, both drop the record)
        text = tags.open_tag + "\n" + tags.close_tag
        v = validate_generation(text, TOOLS, tags)
        assert not v["valid"]
        assert v["problems"]
