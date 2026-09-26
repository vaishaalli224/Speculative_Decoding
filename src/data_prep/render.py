"""Shared rendering + region-annotation helpers for the data pipeline.

Single source of truth for everything that touches the target's chat template
(plan.md §1.3: "never hand-roll these strings"). The tool-call delimiters are
extracted at runtime from a probe render of the *target's* chat template, so a
tokenizer/template update cannot silently desync the region labels from the
rendered text.

Rendering model (verified against the template 2026-09-26):
  - render_sequence(msgs, tools)  = full training sequence; a conversation's
    turns render as <|im_start|>role\n...<|im_end|>\n blocks. add_generation_
    prompt=True appends a *new* assistant header after the last turn, so the
    generation prompt of a finished conversation is NOT a prefix cut of it —
    instead, an inference context is rendered from the context messages only
    (render_context).
  - assistant turn char span = from just after `<|im_start|>assistant\n`
    through `<|im_end|>` inclusive. Loss lives exactly there: the template
    always supplies the header at inference, and the model must learn to emit
    everything up to and including <|im_end|> (the trailing \n after it is
    boilerplate, no loss).

Region codes (plan.md §4.3) — compact ints, cheap to store per token:
  0 = prompt / context (no loss)
  1 = assistant prose (free-form text)
  2 = assistant tool-call JSON (payload between the tool-call tags)
  3 = tool-call tag token itself (boundary; counted in the tool-call region)
  4 = assistant final-answer prose (xLAM has none; ToolBench Finish calls)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TARGET_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

# The workspace .env (repo root's parent) holds HF_TOKEN.
ENV_PATH = Path(__file__).resolve().parents[2].parent / ".env"

# Region codes
REGION_CONTEXT = 0
REGION_PROSE = 1
REGION_TOOL_CALL = 2
REGION_TAG = 3
REGION_FINAL_ANSWER = 4

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_HEADER = IM_START + "assistant\n"


@dataclass(frozen=True)
class ToolCallTags:
    """Delimiters extracted from a probe render of the target's template."""

    open_tag: str
    close_tag: str

    def find_calls(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) char spans of every tool-call JSON payload.

        The span covers the payload between the tags (exclusive of the tags
        themselves). Only tool calls inside *assistant turns* count — the
        system prompt's instruction block contains an example tag pair that
        must not be labeled as a tool call (verified 2026-09-26). An
        unterminated call yields no span; its trailing text is treated as
        prose by the region labeler.
        """
        spans = []
        for a_s, a_e in find_assistant_turns(text):
            pos = a_s
            while True:
                s = text.find(self.open_tag, pos, a_e)
                if s < 0:
                    break
                payload_start = s + len(self.open_tag)
                e = text.find(self.close_tag, payload_start, a_e)
                if e < 0:
                    break
                if payload_start < e:  # skip empty pairs
                    spans.append((payload_start, e))
                pos = e + len(self.close_tag)
        return spans


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    """Load the target tokenizer once per process (loads .env for HF_TOKEN)."""
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TARGET_MODEL)


@lru_cache(maxsize=1)
def get_tool_call_tags() -> ToolCallTags:
    """Probe the target's template once and extract the tag strings.

    Renders one assistant tool_call message through the target's template and
    pulls the delimiters out of the assistant section. This is the only place
    the tags are derived — callers must not hand-type them anywhere else.
    """
    tok = get_tokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "probe_fn",
                "description": "probe",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        }
    ]
    msgs = [
        {"role": "user", "content": "probe"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "probe_fn", "arguments": {"x": "1"}},
                }
            ],
        },
    ]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    a_start = text.rindex(ASSISTANT_HEADER) + len(ASSISTANT_HEADER)
    a_end = text.index(IM_END, a_start)
    assistant = text[a_start:a_end]

    # One call renders as: <open>\n{json}\n<close> — take first/last lines.
    lines = [ln for ln in assistant.split("\n") if ln]
    if len(lines) < 3:
        raise RuntimeError(f"probe render unexpected: {assistant!r}")
    open_tag, close_tag = lines[0], lines[-1]
    if open_tag == close_tag:
        raise RuntimeError("probe render produced identical open/close tool-call tags")
    return ToolCallTags(open_tag=open_tag, close_tag=close_tag)


def render_sequence(messages: list[dict], tools: list[dict]) -> str:
    """Full training-sequence render of the conversation."""
    return get_tokenizer().apply_chat_template(messages, tools=tools, tokenize=False)


def render_context(messages: list[dict], tools: list[dict]) -> str:
    """Inference-time prompt: context messages + a fresh assistant header."""
    return get_tokenizer().apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )


def find_assistant_turns(text: str) -> list[tuple[int, int]]:
    """Char spans of assistant turns: (just after header, through <|im_end|>).

    The header `<|im_start|>assistant\n` is template-supplied at inference and
    carries no loss; `<|im_end|>` does (the model must learn to stop). The
    newline after <|im_end|> is boilerplate and stays outside the span.
    """
    spans = []
    pos = 0
    while True:
        h = text.find(ASSISTANT_HEADER, pos)
        if h < 0:
            break
        start = h + len(ASSISTANT_HEADER)
        e = text.find(IM_END, start)
        if e < 0:
            break  # unterminated turn: no loss span for it
        spans.append((start, e + len(IM_END)))
        pos = e + len(IM_END)
    return spans


def find_tag_spans(text: str, tags: ToolCallTags) -> list[tuple[int, int]]:
    """Char spans of the open/close tag occurrences themselves."""
    spans = []
    for tag in (tags.open_tag, tags.close_tag):
        pos = 0
        while True:
            i = text.find(tag, pos)
            if i < 0:
                break
            spans.append((i, i + len(tag)))
            pos = i + len(tag)
    return spans


def tokenize_with_regions(
    text: str,
    tool_call_spans: list[tuple[int, int]],
    tag_spans: list[tuple[int, int]],
    final_answer_spans: list[tuple[int, int]] | None = None,
) -> dict:
    """Tokenize `text` and label every token with its region code.

    A token gets the region of the span containing its start offset (spans
    are disjoint by construction: tag spans are checked first). Anything not
    in a span is prose. Region 0 (context) is assigned afterwards by the
    caller for tokens outside all assistant turns.

    Returns {input_ids, offsets, regions} — offsets are kept so downstream
    analysis can re-derive finer cuts (e.g. argument keys vs. values, §4.3
    extra cut) without re-tokenizing.
    """
    tok = get_tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = [tuple(o) for o in enc["offset_mapping"]]

    def region_of(start: int) -> int:
        for s, e in tag_spans:
            if s <= start < e:
                return REGION_TAG
        for s, e in tool_call_spans:
            if s <= start < e:
                return REGION_TOOL_CALL
        if final_answer_spans:
            for s, e in final_answer_spans:
                if s <= start < e:
                    return REGION_FINAL_ANSWER
        return REGION_PROSE

    regions = [region_of(s) for s, _ in offsets]
    return {"input_ids": ids, "offsets": offsets, "regions": regions}


def build_record(
    messages: list[dict],
    tools: list[dict],
    query_id: Any,
) -> dict:
    """Render + tokenize + label one conversation -> train/eval record.

    Labels: loss exactly on assistant-turn tokens (find_assistant_turns), -100
    elsewhere. A token must not straddle an assistant-turn boundary — that
    would mean the tokenizer merged template boilerplate into a loss token,
    which the Qwen pretokenizer does not do for our data; assert to be sure.
    """
    full_text = render_sequence(messages, tools)
    tags = get_tool_call_tags()
    tool_call_spans = tags.find_calls(full_text)
    tag_spans = find_tag_spans(full_text, tags)
    assistant_spans = find_assistant_turns(full_text)

    enc = tokenize_with_regions(full_text, tool_call_spans, tag_spans)
    ids, offsets, regions = enc["input_ids"], enc["offsets"], enc["regions"]

    labels = [-100] * len(ids)
    for i, (s, e) in enumerate(offsets):
        for a_s, a_e in assistant_spans:
            if s >= a_s and e <= a_e:
                labels[i] = ids[i]
                break
            if s < a_s < e or s < a_e < e:
                raise AssertionError(
                    f"token {i} {full_text[s:e]!r} straddles assistant-turn boundary"
                )

    # context region: everything outside assistant turns
    regions = [
        r if labels[i] != -100 else REGION_CONTEXT
        for i, r in enumerate(regions)
    ]

    loss_idx = [i for i, l in enumerate(labels) if l != -100]
    return {
        "input_ids": ids,
        "labels": labels,
        "regions": _rle(regions),
        "n_prompt_tokens": loss_idx[0] if loss_idx else len(ids),
        "n_tokens": len(ids),
        "n_tool_calls": len(tool_call_spans),
        "n_tools": len(tools),
        "query_id": query_id,
    }


def _rle(codes: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for c in codes:
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return out


def unrle(pairs: list[list[int]]) -> list[int]:
    """Inverse of _rle — used by tests and downstream analysis."""
    return [c for c, n in pairs for _ in range(n)]
