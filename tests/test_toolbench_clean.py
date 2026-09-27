"""Golden tests for ToolBench cleaning (plan.md §3.2, §6.2).

Facts verified against data/raw/toolbench_default on 2026-09-26 and encoded
here as assertions (see src/data_prep/toolbench_clean.py's docstring for the
measured rates):

  - conversations column is columnar {"from": [...], "value": [...]}
  - id = "Step N: <query>"; first user turn = query + "Begin!\n"
  - Action Input JSON can contain true/false (json, not ast) and trailing
    garbage (raw_decode); single-quoted python-literal payloads fall back
    to ast.literal_eval
  - "Arguments:" is a variant of "Action Input:"
  - multi-block turns: the LAST parseable block is the executed call;
    earlier blocks are retry drafts (never checked against the API list)
  - tool schemas: required+optional -> OpenAI required-only; example_value
    dropped; Finish excluded from the rendered tool list
  - give_answer Finish -> final-answer prose; its span is REGION_FINAL_ANSWER
  - function envelopes pass through verbatim (may be truncated mid-JSON)
  - the AutoGPT scaffold system prompt is dropped entirely
"""

from __future__ import annotations

import json

import pytest

from src.data_prep.render import (
    REGION_CONTEXT,
    REGION_FINAL_ANSWER,
    REGION_PROSE,
    REGION_TAG,
    REGION_TOOL_CALL,
    build_record,
    find_assistant_turns,
    get_tokenizer,
    render_sequence,
    unrle,
)
from src.data_prep.toolbench_clean import (
    FINISH_NAME,
    _BEGIN_SUFFIX as BEGIN_SUFFIX,
    convert_conversation,
    convert_tb_tool,
    conv_list,
    norm_query,
    parse_assistant_turn,
    parse_api_list,
)

# ---------------------------------------------------------------------------
# Fixtures built from real mirror-data shapes (transcribed from raw rows)
# ---------------------------------------------------------------------------

_API_ENTRY = {
    "name": "get_weather",
    "description": 'This is the subfunction for tool "weather", you can use this tool.',
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "", "example_value": "Paris"},
            "unit": {"type": "string", "description": "temp unit"},
        },
        "required": ["city"],
        "optional": ["unit"],
    },
}

RAW_SYSTEM = (
    "You are AutoGPT, you can use many tools(functions) to do the following task.\n"
    "...\nSpecifically, you have access of the following APIs: "
    + repr([_API_ENTRY])
)

RAW_ASSISTANT_CALL = (
    "Thought: I need the weather for Paris.\n"
    "Action: get_weather\n"
    'Action Input: {\n  "city": "Paris"\n}'
)

RAW_FUNCTION = '{"error": "", "response": "{\'temp\': 18}"}'

RAW_ASSISTANT_FINISH = (
    "Thought: I have the weather now.\n"
    "Action: Finish\n"
    'Action Input: {\n  "return_type": "give_answer",\n'
    '  "final_answer": "It is 18 degrees in Paris."\n}'
)

RAW_QUERY = "What's the weather in Paris?"


def _raw_example(extra_turns: list[dict] | None = None) -> dict:
    """A raw mirror-style record with columnar conversations."""
    turns = [
        {"from": "system", "value": RAW_SYSTEM},
        {"from": "user", "value": RAW_QUERY + BEGIN_SUFFIX},
        {"from": "assistant", "value": RAW_ASSISTANT_CALL},
        {"from": "function", "value": RAW_FUNCTION},
        {"from": "assistant", "value": RAW_ASSISTANT_FINISH},
    ]
    if extra_turns:
        turns += extra_turns
    return {
        "id": "Step 3: " + RAW_QUERY,
        "conversations": {
            "from": [t["from"] for t in turns],
            "value": [t["value"] for t in turns],
        },
    }


class TestParsingPrimitives:
    def test_conv_list_columnar(self):
        ex = _raw_example()
        convs = conv_list(ex)
        assert [c["from"] for c in convs] == [
            "system", "user", "assistant", "function", "assistant",
        ]
        assert convs[1]["value"].endswith(BEGIN_SUFFIX)

    def test_norm_query_strips_step_prefix(self):
        assert norm_query("Step 12:  What's\n the weather? ") == "What's the weather?"

    def test_parse_api_list(self):
        apis = parse_api_list(RAW_SYSTEM)
        assert apis is not None and len(apis) == 1
        assert apis[0]["name"] == "get_weather"
        assert "optional" in apis[0]["parameters"]  # TB-native shape

    def test_parse_assistant_call(self):
        parsed = parse_assistant_turn(RAW_ASSISTANT_CALL)
        assert parsed["calls"] == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
        assert parsed["thought"] == "I need the weather for Paris."

    def test_json_true_false_payload(self):
        # ast.literal_eval fails on true/false; json raw_decode must win
        v = (
            "Thought: t\nAction: search\n"
            'Action Input: {\n  "q": "x",\n  "autocorrect": true,\n  "n": 5\n}'
        )
        parsed = parse_assistant_turn(v)
        assert parsed["calls"][0]["arguments"] == {
            "q": "x", "autocorrect": True, "n": 5
        }

    def test_payload_with_trailing_garbage(self):
        v = (
            "Thought: t\nAction: Finish\n"
            'Action Input: {\n  "return_type": "give_answer",\n'
            '  "final_answer": "done"\n}\nAction: Finish\nActio'
        )
        parsed = parse_assistant_turn(v)
        assert parsed["calls"][-1]["arguments"]["final_answer"] == "done"

    def test_arguments_variant(self):
        v = "Thought: t\nAction: f\nArguments: {\n  \"x\": 1\n}"
        parsed = parse_assistant_turn(v)
        assert parsed["calls"] == [{"name": "f", "arguments": {"x": 1}}]

    def test_single_quoted_payload_fallback(self):
        v = "Thought: t\nAction: f\nAction Input: {'x': 'y'}"
        parsed = parse_assistant_turn(v)
        assert parsed["calls"] == [{"name": "f", "arguments": {"x": "y"}}]

    def test_multi_block_last_parseable_wins(self):
        # retry draft block first (different fn), real call last — as in
        # raw ex 27; earlier drafts are never executed
        v = (
            "Thought: I will try a different function.\n"
            "Here is my new action:\n"
            "Action: draft_fn\n"
            "Arguments: {\n  \"postcode\": 9999\n}\n"
            "\nAction: get_weather\n"
            'Action Input: {\n  "city": "Paris"\n}'
        )
        parsed = parse_assistant_turn(v)
        assert len(parsed["calls"]) == 2
        assert parsed["calls"][-1]["name"] == "get_weather"

    def test_unparseable_block_skipped(self):
        # a block whose payload never closes (truncated) yields no call from
        # it; the later parseable block is the real one
        v = (
            "Thought: t\nAction: Finish\n"
            'Action Input: {\n  "return_type": "give_answer",\n'
            '  "final_answer": "Here is the historical weat'
        )
        assert parse_assistant_turn(v) is None


class TestSchemaConversion:
    def test_convert_tb_tool(self):
        apis = parse_api_list(RAW_SYSTEM)
        conv = convert_tb_tool(apis[0])
        fn = conv["function"]
        assert fn["name"] == "get_weather"
        props = fn["parameters"]["properties"]
        assert props["city"] == {"type": "string"}  # empty desc + example dropped
        assert props["unit"] == {"type": "string", "description": "temp unit"}
        # optional-list params are NOT in required
        assert fn["parameters"]["required"] == ["city"]

    def test_enum_preserved(self):
        raw = {
            "name": "f",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {
                    "rt": {"type": "string", "enum": ["give_answer", "give_up"]}
                },
                "required": ["rt"],
                "optional": [],
            },
        }
        conv = convert_tb_tool(raw)
        assert conv["function"]["parameters"]["properties"]["rt"]["enum"] == [
            "give_answer", "give_up"
        ]


class TestConvertConversation:
    def test_end_to_end_shape(self):
        msgs, tools, meta = convert_conversation(_raw_example())
        assert [m["role"] for m in msgs] == [
            "user", "assistant", "tool", "assistant",
        ]
        # Begin! stripped from the first user turn
        assert msgs[0]["content"] == RAW_QUERY
        # thought kept as content, call in tool_calls with dict arguments
        assert msgs[1]["content"] == "I need the weather for Paris."
        assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
        # function envelope verbatim (truncation-tolerant)
        assert msgs[2]["content"] == RAW_FUNCTION
        assert msgs[2]["name"] == "get_weather"
        # Finish turn: thought + final answer folded into prose content
        # (no tool_calls key — it is a pure-prose assistant turn)
        assert "tool_calls" not in msgs[3]
        assert msgs[3]["content"] == (
            "I have the weather now.\n\nIt is 18 degrees in Paris."
        )
        assert meta["final_answer"] == "It is 18 degrees in Paris."
        # Finish excluded from rendered tools
        assert [t["function"]["name"] for t in tools] == ["get_weather"]
        assert meta["query"] == RAW_QUERY

    def test_mid_conversation_user_retry_prompt_kept(self):
        # shape from raw data: call -> response -> RETRY USER turn -> new call
        retry = (
            "This is not the first time you try this task, all previous "
            "trails failed. [...] Here are some previous actions candidates:\n[...]"
        )
        turns = [
            {"from": "system", "value": RAW_SYSTEM},
            {"from": "user", "value": RAW_QUERY + BEGIN_SUFFIX},
            {"from": "assistant", "value": RAW_ASSISTANT_CALL},
            {"from": "function", "value": RAW_FUNCTION},
            {"from": "user", "value": retry},
            {"from": "assistant", "value": RAW_ASSISTANT_FINISH},
        ]
        ex = {
            "id": "Step 2: " + RAW_QUERY,
            "conversations": {
                "from": [t["from"] for t in turns],
                "value": [t["value"] for t in turns],
            },
        }
        msgs, tools, meta = convert_conversation(ex)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        assert msgs[3]["content"] == retry  # kept verbatim

    def test_mid_conversation_finish_dropped(self):
        # a Finish call followed by more turns = mid-conv Finish -> drop
        ex = _raw_example(
            extra_turns=[{"from": "user", "value": "retry\n" + BEGIN_SUFFIX}]
        )
        assert convert_conversation(ex) is None

    def test_give_up_dropped(self):
        v = (
            "Thought: I cannot proceed.\nAction: Finish\n"
            'Action Input: {\n  "return_type": "give_up_and_restart"\n}'
        )
        turns = [
            {"from": "system", "value": RAW_SYSTEM},
            {"from": "user", "value": RAW_QUERY + BEGIN_SUFFIX},
            {"from": "assistant", "value": v},
        ]
        ex = {
            "id": "Step 1: " + RAW_QUERY,
            "conversations": {
                "from": [t["from"] for t in turns],
                "value": [t["value"] for t in turns],
            },
        }
        assert convert_conversation(ex) is None

    def test_hallucinated_action_name_dropped(self):
        v = (
            "Thought: t\nAction: not_in_api_list\n"
            'Action Input: {\n  "x": "1"\n}'
        )
        turns = [
            {"from": "system", "value": RAW_SYSTEM},
            {"from": "user", "value": RAW_QUERY + BEGIN_SUFFIX},
            {"from": "assistant", "value": v},
        ]
        ex = {
            "id": "Step 1: " + RAW_QUERY,
            "conversations": {
                "from": [t["from"] for t in turns],
                "value": [t["value"] for t in turns],
            },
        }
        assert convert_conversation(ex) is None

    def test_retry_draft_with_bad_name_not_dropped(self):
        # an UNEXECUTED draft block (not last) may name a nonexistent fn —
        # only the executed (last) call is checked against the API list
        v = (
            "Thought: t\n"
            "Here is my new action:\n"
            "Action: `get_weather`\n"
            "Action Input: {\n  \"city\": \"Paris\"\n}\n"
            "\nAction: get_weather\n"
            'Action Input: {\n  "city": "Paris"\n}'
        )
        turns = [
            {"from": "system", "value": RAW_SYSTEM},
            {"from": "user", "value": RAW_QUERY + BEGIN_SUFFIX},
            {"from": "assistant", "value": v},
            {"from": "function", "value": RAW_FUNCTION},
            {"from": "assistant", "value": RAW_ASSISTANT_FINISH},
        ]
        ex = {
            "id": "Step 1: " + RAW_QUERY,
            "conversations": {
                "from": [t["from"] for t in turns],
                "value": [t["value"] for t in turns],
            },
        }
        msgs, tools, meta = convert_conversation(ex)
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_system_prompt_dropped_from_render(self):
        msgs, tools, _ = convert_conversation(_raw_example())
        assert all(m["role"] != "system" for m in msgs)  # conversion drops it
        full = render_sequence(msgs, tools)
        assert "AutoGPT" not in full
        assert "Action Input" not in full  # scaffold format fully converted away


class TestBuildRecordToolbench:
    def test_regions_label_final_answer(self):
        msgs, tools, meta = convert_conversation(_raw_example())
        rec = build_record(
            msgs, tools, query_id=0, final_answer_text=meta["final_answer"]
        )
        regions = unrle(rec["regions"])
        tok = get_tokenizer()

        # labels <-> regions: loss exactly on non-context tokens
        for i, (l, c) in enumerate(zip(rec["labels"], regions)):
            assert (l != -100) == (c != REGION_CONTEXT), i

        # tool-call payload round-trips from region-2 tokens alone
        tc_idx = [i for i, c in enumerate(regions) if c == REGION_TOOL_CALL]
        payload = tok.decode([rec["input_ids"][i] for i in tc_idx])
        assert json.loads(payload) == {"name": "get_weather", "arguments": {"city": "Paris"}}

        # final answer round-trips from region-4 tokens alone
        fa_idx = [i for i, c in enumerate(regions) if c == REGION_FINAL_ANSWER]
        assert fa_idx, "no region-4 tokens"
        answer = tok.decode([rec["input_ids"][i] for i in fa_idx])
        assert meta["final_answer"] in answer
        # and the answer is inside the last assistant turn (has loss)
        assert all(rec["labels"][i] != -100 for i in fa_idx)

        # two assistant turns, one tool call, Finish not among tools
        assert rec["n_tool_calls"] == 1
        assert rec["n_tools"] == 1
        assert len(find_assistant_turns(render_sequence(msgs, tools))) == 2

    def test_final_answer_span_desync_asserts(self):
        # if the template stopped inserting content verbatim, the span
        # computation must fail loudly, not mislabel silently
        msgs, tools, meta = convert_conversation(_raw_example())
        with pytest.raises(AssertionError):
            build_record(
                msgs, tools, query_id=0, final_answer_text="NOT THE ANSWER" * 3
            )

    def test_thought_prose_is_region_prose(self):
        msgs, tools, meta = convert_conversation(_raw_example())
        rec = build_record(
            msgs, tools, query_id=0, final_answer_text=meta["final_answer"]
        )
        regions = unrle(rec["regions"])
        tok = get_tokenizer()
        prose_idx = [i for i, c in enumerate(regions) if c == REGION_PROSE]
        prose = tok.decode([rec["input_ids"][i] for i in prose_idx])
        assert "I need the weather for Paris." in prose  # first thought
        assert "I have the weather now." in prose  # Finish thought
