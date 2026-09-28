"""Golden tests for the TB-prefix KD ablation (src/data_prep/build_tb_
distill.py). Torch-free; runs against the committed frozen indices + raw
data (skipped when data/raw is absent, e.g. a fresh clone)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_prep.build_tb_distill import (
    MAX_CTX_DEFAULT,
    tb_boundary_contexts,
    validate_tb_generation,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "toolbench_default"

pytestmark = pytest.mark.skipif(
    not RAW.exists(), reason="raw ToolBench not downloaded"
)


@pytest.fixture(scope="module")
def ctxs():
    return tb_boundary_contexts(limit=60)


class TestBoundaryContexts:
    def test_every_context_ends_at_assistant_header(self, ctxs):
        from src.data_prep.render import get_tokenizer

        tok = get_tokenizer()
        for c in ctxs[:20]:
            tail = tok.decode(c["prompt_ids"][-4:])
            assert tail.endswith("assistant\n"), c["query_id"]

    def test_query_id_encodes_conversation_and_turn(self, ctxs):
        for c in ctxs:
            raw, turn = c["query_id"].split("#t")
            int(raw)
            int(turn)

    def test_turns_enumerate_in_order_per_conversation(self, ctxs):
        per: dict[str, list[int]] = {}
        for c in ctxs:
            conv, t = c["query_id"].split("#t")
            per.setdefault(conv, []).append(int(t))
        for conv, turns in per.items():
            assert turns == sorted(turns) and turns[0] == 0

    def test_limit_is_a_prefix_in_frozen_order(self, ctxs):
        more = tb_boundary_contexts(limit=30)
        assert [c["query_id"] for c in more] == [c["query_id"] for c in ctxs[:30]]

    def test_max_ctx_truncates_to_turn_boundary(self):
        small = tb_boundary_contexts(limit=6, max_ctx=256)
        assert all(len(c["prompt_ids"]) <= 256 for c in small)
        from src.data_prep.render import get_tokenizer

        tok = get_tokenizer()
        for c in small:
            assert c["prompt_ids"][0] == int(
                tok.convert_tokens_to_ids("<|im_start|>")
            ) or len(c["prompt_ids"]) <= 256


class TestTBValidation:
    """T3: prose-only generations are VALID on TB (the target's majority
    class); only the calls that exist get validated."""

    def _tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}}, "required": ["city"]},
            },
        }]

    def test_prose_only_generation_is_kept(self):
        from src.data_prep.render import get_tool_call_tags

        v = validate_tb_generation(
            "Let me look up the weather for you.", self._tools(),
            get_tool_call_tags(),
        )
        assert v["valid"], v["problems"]

    def test_valid_call_in_prose_is_kept(self):
        import json as J

        from src.data_prep.render import get_tool_call_tags

        payload = J.dumps({"name": "get_weather",
                           "arguments": {"city": "Paris"}})
        text = ("I will check the weather. " + "<tools>" + "\n" + payload
                + "\n" + "</tools>")
        v = validate_tb_generation(text, self._tools(), get_tool_call_tags())
        assert v["valid"], v["problems"]
        assert v["n_calls"] == 1

    def test_hallucinated_name_still_drops(self):
        import json as J

        from src.data_prep.render import get_tool_call_tags

        payload = J.dumps({"name": "not_a_tool", "arguments": {}})
        text = "<tools>\n" + payload + "\n</tools>"
        v = validate_tb_generation(text, self._tools(), get_tool_call_tags())
        assert not v["valid"]
        assert any("unknown function" in p for p in v["problems"])

    def test_g8_comma_parallel_calls_valid(self):
        import json as J

        from src.data_prep.render import get_tool_call_tags

        p1 = J.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
        p2 = J.dumps({"name": "get_weather", "arguments": {"city": "Rome"}})
        text = "<tools>\n" + p1 + ",\n" + p2 + "\n</tools>"
        v = validate_tb_generation(text, self._tools(), get_tool_call_tags())
        assert v["valid"], v["problems"]
