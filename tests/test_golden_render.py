"""Golden tests for the target-template rendering + region labeling (plan.md
§1.3 "guarded by a golden snapshot test", §6.2). Facts verified against
Qwen/Qwen2.5-Coder-14B-Instruct on 2026-09-26 and encoded here as assertions:

  - system prompt: default Qwen system line + '# Tools' + <tools> JSON schemas
  - assistant tool call = one JSON object per call between the tool-call tags
  - the tool-call tags are single special tokens (ids 151657 / 151658)
  - dict-form arguments render as a nested object (string form double-encodes)
  - tool responses render inside user-role blocks
  - add_generation_prompt=True appends a NEW assistant header AFTER the last
    turn — so a finished conversation's generation prompt is NOT a prefix of
    its full render. Loss spans therefore come from assistant-turn char spans
    (render.find_assistant_turns), never from a prompt cut.
  - <|im_end|> carries loss (the draft must learn to stop); the trailing
    newline after it does not.

The tags themselves are never hand-typed here (plan rule); they are extracted
at runtime by src.data_prep.render.get_tool_call_tags().
"""

from __future__ import annotations

import json

import pytest

from src.data_prep.render import (
    IM_END,
    IM_START,
    REGION_CONTEXT,
    REGION_PROSE,
    REGION_TAG,
    REGION_TOOL_CALL,
    build_record,
    find_assistant_turns,
    find_tag_spans,
    get_tokenizer,
    get_tool_call_tags,
    render_context,
    render_sequence,
    tokenize_with_regions,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    }
]

MSGS_SINGLE = [
    {"role": "user", "content": "weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    },
]

MSGS_TOOL_RESPONSE = MSGS_SINGLE + [
    {"role": "tool", "name": "get_weather", "content": '{"temp_c": 18}'},
]


def _assistant_section(full: str) -> str:
    start = full.rindex(IM_START + "assistant\n") + len(IM_START + "assistant\n")
    end = full.index(IM_END, start)
    return full[start:end]


@pytest.fixture(scope="session")
def tags():
    return get_tool_call_tags()


class TestTemplateFacts:
    def test_system_prompt_announces_tools(self, tags):
        full = render_sequence(MSGS_SINGLE, TOOLS)
        assert "# Tools" in full
        assert "<tools>" in full and "</tools>" in full
        assert json.dumps(TOOLS[0]) in full  # schema appears verbatim

    def test_tool_call_tags_are_single_special_tokens(self, tags):
        tok = get_tokenizer()
        for tag in (tags.open_tag, tags.close_tag):
            ids = tok.encode(tag, add_special_tokens=False)
            assert len(ids) == 1, f"{ascii(tag)} is not a single token: {ids}"
            assert ids[0] in (151657, 151658)

    def test_dict_arguments_render_nested(self, tags):
        assistant = _assistant_section(render_sequence(MSGS_SINGLE, TOOLS))
        payload = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
        assert payload in assistant  # nested object, not double-escaped
        assert '\\"' not in assistant

    def test_generation_prompt_appends_new_header(self, tags):
        # NOT a prefix of the full render — the design constraint that broke
        # the naive prompt-cut labeling (verified 2026-09-26)
        full = render_sequence(MSGS_TOOL_RESPONSE, TOOLS)
        prompt = render_context(MSGS_TOOL_RESPONSE, TOOLS)
        assert not full.startswith(prompt)
        assert prompt.startswith(full)
        assert prompt.endswith(IM_START + "assistant\n")

    def test_prefix_property_tokens_user_only(self, tags):
        # for a *context* (no assistant turn yet), the context render IS a
        # token-level prefix of the full sequence render
        tok = get_tokenizer()
        full = render_sequence(MSGS_TOOL_RESPONSE, TOOLS)
        ctx = render_context(MSGS_TOOL_RESPONSE[:1], TOOLS)
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        assert f_ids[: len(c_ids)] == c_ids

    def test_tool_response_in_user_block(self, tags):
        full = render_sequence(MSGS_TOOL_RESPONSE, TOOLS)
        resp = '{"temp_c": 18}'
        assert resp in full
        # response appears after the assistant call's <|im_end|>
        a_end = full.index(IM_END, full.rindex(IM_START + "assistant"))
        assert full.index(resp) > a_end


class TestAssistantSpanLabeling:
    def test_find_assistant_turns_single(self, tags):
        full = render_sequence(MSGS_SINGLE, TOOLS)
        spans = find_assistant_turns(full)
        assert len(spans) == 1
        s, e = spans[0]
        assert full[s : e].endswith(IM_END)
        # span starts after the header
        assert full[s - len(IM_START + "assistant\n") : s] == IM_START + "assistant\n"

    def test_find_assistant_turns_multi(self, tags):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "hello", "tool_calls": []},
            {"role": "tool", "name": "get_weather", "content": "r1"},
            {"role": "assistant", "content": "done"},
        ]
        full = render_sequence(msgs, TOOLS)
        spans = find_assistant_turns(full)
        assert len(spans) == 2
        texts = [full[s:e] for s, e in spans]
        assert texts[0].startswith("hello")
        assert texts[1].startswith("done")

    def test_build_record_single_call(self, tags):
        rec = build_record(MSGS_SINGLE, TOOLS, query_id=0)
        tok = get_tokenizer()
        regions = [c for c, n in rec["regions"] for _ in range(n)]  # unrle
        ids, labels = rec["input_ids"], rec["labels"]
        assert len(ids) == len(labels) == len(regions)

        # loss tokens are exactly the assistant-turn tokens
        loss_idx = [i for i, l in enumerate(labels) if l != -100]
        assert all(labels[i] == ids[i] for i in loss_idx)
        assert all(regions[i] != REGION_CONTEXT for i in loss_idx)
        assert all(regions[i] == REGION_CONTEXT for i in range(len(ids)) if labels[i] == -100)

        # <|im_end|> is the last loss token
        assert tok.decode([ids[loss_idx[-1]]]) == IM_END

        # tool-call payload decodes to exactly the JSON we passed
        tc_idx = [i for i, c in enumerate(regions) if c == REGION_TOOL_CALL]
        payload = tok.decode([ids[i] for i in tc_idx])
        assert json.loads(payload) == {
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }

    def test_build_record_response_carries_no_loss(self, tags):
        rec = build_record(MSGS_TOOL_RESPONSE, TOOLS, query_id=0)
        full = render_sequence(MSGS_TOOL_RESPONSE, TOOLS)
        resp = '{"temp_c": 18}'
        r_start = full.index(resp)
        offs = get_tokenizer()(
            full, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        enc_idx = next(i for i, (a, b) in enumerate(offs) if a <= r_start < b)
        assert rec["labels"][enc_idx] == -100

    def test_build_record_multi_turn_no_straddle(self, tags):
        # multi-turn conversation: every loss token inside one assistant span
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "hi again"},
        ]
        rec = build_record(msgs, TOOLS, query_id=0)
        regions = [c for c, n in rec["regions"] for _ in range(n)]
        loss_idx = [i for i, l in enumerate(rec["labels"]) if l != -100]
        full = render_sequence(msgs, TOOLS)
        spans = find_assistant_turns(full)
        # reconstruct char coverage of loss tokens; verify containment
        enc = get_tokenizer()(full, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        for i in loss_idx:
            s, e = offs[i]
            assert any(a_s <= s and e <= a_e for a_s, a_e in spans), i

    def test_regions_rle_roundtrip(self, tags):
        rec = build_record(MSGS_SINGLE, TOOLS, query_id=0)
        from src.data_prep.render import unrle

        flat = unrle(rec["regions"])
        assert len(flat) == rec["n_tokens"]
        assert set(flat) <= {REGION_CONTEXT, REGION_PROSE, REGION_TAG, REGION_TOOL_CALL}


class TestXlamConversion:
    """Guard the xLAM schema conversion (xlam_prep.convert_xlam_tool)."""

    @pytest.fixture
    def raw_tool(self):
        return {
            "name": "time_zone_api",
            "description": "Fetches time zone info.",
            "parameters": {
                "q": {"description": "Query parameter", "type": "str"},
                "lang": {"description": "Language", "type": "str, optional", "default": "en"},
                "limit": {"description": "Max results", "type": "int, default=100"},
                "coords": {"description": "Coordinates", "type": "List[float]"},
            },
        }

    def test_conversion(self, raw_tool):
        from src.data_prep.xlam_prep import convert_xlam_tool

        conv = convert_xlam_tool(raw_tool)
        fn = conv["function"]
        assert fn["name"] == "time_zone_api"
        props = fn["parameters"]["properties"]
        assert props["q"]["type"] == "string"
        assert props["lang"]["type"] == "string"
        assert props["lang"]["default"] == "en"
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["default"] == 100
        assert props["coords"]["type"] == "array"
        assert "[type: List[float]]" in props["coords"]["description"]
        # q: plain 'str' -> required. lang/limit: optional/default -> optional.
        # coords: 'List[float]' no marker -> required (container, but not
        # marked optional in xLAM — the answer may still need it).
        assert fn["parameters"]["required"] == ["q", "coords"]

    def test_process_example_end_to_end(self, raw_tool):
        from src.data_prep.xlam_prep import process_example
        from src.data_prep.render import unrle

        ex = {
            "id": 999999,
            "query": "What time zone is postal code G2J in?",
            "tools": json.dumps([raw_tool]),
            "answers": json.dumps(
                [{"name": "time_zone_api", "arguments": {"q": "G2J", "lang": "fr"}}]
            ),
        }
        rec = process_example(ex)
        regions = unrle(rec["regions"])
        assert len(regions) == rec["n_tokens"]
        assert rec["n_tool_calls"] == 1
        assert rec["n_tools"] == 1
        assert rec["n_answers"] == 1
        n_prompt = rec["n_prompt_tokens"]
        assert all(l == -100 for l in rec["labels"][:n_prompt])
        assert any(l != -100 for l in rec["labels"])
        # loss tokens <=> non-context regions, everywhere (not just after n_prompt:
        # the trailing \n after the final <|im_end|> is boilerplate context)
        for i, (l, c) in enumerate(zip(rec["labels"], regions)):
            assert (l != -100) == (c != REGION_CONTEXT), i
        # and n_prompt_tokens is exactly the first loss token
        assert rec["labels"][n_prompt] != -100
        tok = get_tokenizer()
        tc_idx = [i for i, c in enumerate(regions) if c == REGION_TOOL_CALL]
        payload = tok.decode([rec["input_ids"][i] for i in tc_idx])
        assert json.loads(payload) == {
            "name": "time_zone_api",
            "arguments": {"q": "G2J", "lang": "fr"},
        }
