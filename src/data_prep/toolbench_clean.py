"""ToolBench cleaning: ShareGPT mirror -> target-format conversations
(plan.md §3.2, §6.2).

Verified format facts (measured on data/raw/toolbench_default, 2026-09-26):
  - `conversations` is COLUMNAR: {"from": [...], "value": [...]}.
  - `id` is always "Step N: <query>"; the first user turn is the query plus a
    "Begin!\n" suffix (5000/5000). id minus the Step prefix == user minus
    Begin! exactly.
  - Tool schemas: python-literal list at the end of the system prompt after
    "APIs:" — ast.literal_eval parses 100%. Entries use required+optional
    lists (OpenAI has only required); properties carry type/description/
    enum/example_value. The built-in Finish tool is in every API list.
  - Assistant turns are ToolLLaMA "Thought:/Action:/Action Input:" (a rare
    variant spells it "Arguments:"). Action Input JSON can contain true/false
    (breaks ast.literal_eval) and trailing garbage after the closing brace —
    json raw_decode handles both; ast is the single-quoted fallback. ~10% of
    turns hold >1 "Action:" block (retry drafts / duplicates): the LAST
    parseable block is the turn's real call.
  - `function` turns hold {"error": ..., "response": ...} envelopes, ~27%
    TRUNCATED at ~1027 chars by the mirror — never parse them; the raw
    envelope string becomes the tool content verbatim.
  - Mid-conversation `user` turns are ToolBench retry scaffolds ("This is
    not the first time you try this task...") — kept verbatim. The target
    template renders them as plain user turns after the tool-response block
    (probed 2026-09-26).

Conversion to the target's format (plan.md §3.2):
  - The AutoGPT scaffold system prompt is DROPPED — it instructs the
    Thought:/Action: output format we are converting away from, and the
    target template supplies its own system line + tool schemas.
  - Finish is dropped from the rendered tool list; a give_answer Finish call
    becomes final-answer prose in the last assistant turn (region 4 via
    render.build_record's final_answer_text).
  - Thought prose is kept as assistant content preceding the tool call.
  - Tool envelopes become {"role": "tool", "name": ..., "content": envelope}.

Cleaning filters (plan.md §3.2 "Filters (measured)"):
  keep a conversation iff:
    - every assistant action name is in the conversation's own API list
      (~5% hallucinated names -> drop, measured), Finish always allowed
    - no mid-conversation Finish call (~0.16% — retry drafts)
    - the LAST assistant turn ends with Finish/give_answer
      (the "reached a final answer" filter; 50,147/187,542 measured)
    - the query survives dedupe (normalized "Step N:"-stripped id)

Carve (plan.md §3.2 "Eval carve"): greedy rare-tool carve -> TB-500: 500
eval conversations, 410 held-out tools, leaving 46,242 Stage 2 prefix
conversations (94.2%) whose tools never appear in the eval set.

CLI:
  toolbench_clean.py scan       # raw-data filter counts (fast, no render)
  toolbench_clean.py clean      # parse + filter + dedupe -> clean index
  toolbench_clean.py carve      # TB-500 eval / prefix-pool split indexes
  toolbench_clean.py build --split {eval,prefixes}   # render + tokenize
  toolbench_clean.py stats --split ...               # length/region report
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

from src.data_prep.render import build_record

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "toolbench_default"
OUT_DIR = REPO_ROOT / "data" / "processed" / "toolbench"
EVAL_SIZE = 500
CARVE_SEED = 42

FINISH_NAME = "Finish"
GIVE_ANSWER = "give_answer"

# ---------------------------------------------------------------------------
# Raw-record parsing (columnar conversations -> turn list)
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"^\s*Step\s*\d+\s*:\s*")
# first user turn = query + "\nBegin!\n" (5000/5000 measured) — strip the
# whole scaffold tail including its newline
_BEGIN_SUFFIX = "\nBegin!\n"


def conv_list(ex: dict) -> list[dict]:
    """Columnar conversations column -> ordered [{"from", "value"}, ...]."""
    c = ex["conversations"]
    return [
        {"from": f, "value": v} for f, v in zip(list(c["from"]), list(c["value"]))
    ]


def norm_query(id_str: str) -> str:
    """Normalized query for dedupe: strip the 'Step N:' prefix, collapse ws."""
    return re.sub(r"\s+", " ", _STEP_RE.sub("", id_str or "").strip()).strip()


def parse_api_list(system_value: str) -> list[dict] | None:
    """System prompt -> list of raw API dicts; None if unparseable."""
    try:
        return ast.literal_eval(system_value.split("APIs:", 1)[1].strip())
    except (ValueError, SyntaxError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Tool-call parsing: "Thought:/Action:/Action Input:" -> (prose, calls)
# ---------------------------------------------------------------------------

# A real action block: an "Action:" line whose name is followed (possibly on
# later lines) by an "Action Input:"/"Arguments:" payload. "Action:" lines
# quoted inside Thought prose do not match because they are not line-initial.
_ACTION_RE = re.compile(
    r"^Action:[ \t]*(?P<name>\S+?)[ \t]*$\n"
    r"(?:^Action Input:|^Arguments:)[ \t]*\n?[ \t]*(?P<payload>.*?)"
    r"(?=^Action:[ \t]*\S+[ \t]*$|\Z)",
    re.M | re.S,
)
_THOUGHT_RE = re.compile(r"^Thought:[ \t]*\n?", re.M)

_JSON_DEC = json.JSONDecoder()


def _parse_payload(text: str):
    """Payload string -> arguments dict. JSON first (true/false/null); the
    single-quoted python-literal form (ToolBench's own convention) second."""
    text = text.strip()
    try:
        return _JSON_DEC.raw_decode(text)[0]
    except ValueError:
        pass
    try:
        val = ast.literal_eval(text)
        return val if isinstance(val, dict) else None
    except (ValueError, SyntaxError):
        return None


def parse_assistant_turn(value: str) -> dict | None:
    """One assistant turn -> {thought, calls: [{name, arguments}]} or None.

    The LAST parseable action block is the turn's real call: ~10% of turns
    contain retry drafts or duplicated blocks, and only the final block is
    the action actually taken (measured 2026-09-26).
    """
    calls: list[dict] = []
    first_block_start = None
    last_parse_end = 0
    for m in _ACTION_RE.finditer(value):
        if first_block_start is None:
            first_block_start = m.start()
        payload = _parse_payload(m.group("payload"))
        if payload is not None:
            calls.append({"name": m.group("name"), "arguments": payload})
            last_parse_end = m.end()
    if not calls:
        return None
    thought = _THOUGHT_RE.sub("", value[:first_block_start]).strip()
    # keep prose that trails the final parsed block (rare; e.g. a stray note)
    tail = value[last_parse_end:].strip()
    if tail:
        thought = f"{thought}\n{tail}".strip()
    return {"thought": thought, "calls": calls}


# ---------------------------------------------------------------------------
# Schema conversion: ToolBench raw API -> OpenAI-style function schema
# ---------------------------------------------------------------------------


def convert_tb_tool(raw: dict) -> dict:
    """required+optional lists -> OpenAI required-only; drop the optional key."""
    params = raw.get("parameters", {})
    props = params.get("properties", {})
    required = list(params.get("required", []))
    optional = list(params.get("optional", []))
    schema_props: dict = {}
    for pname, pd in props.items():
        if isinstance(pd, dict):
            descriptor = dict(pd)
            descriptor.pop("example_value", None)
            if not descriptor.get("description"):
                descriptor.pop("description", None)
            schema_props[pname] = descriptor
        else:
            schema_props[pname] = {"type": "string"}
    schema = {"type": "object", "properties": schema_props}
    req = [p for p in required if p in schema_props]
    if req:
        schema["required"] = req
    return {
        "type": "function",
        "function": {
            "name": raw["name"],
            "description": raw.get("description", ""),
            "parameters": schema,
        },
    }


# ---------------------------------------------------------------------------
# Conversation conversion: raw turns -> target-format messages + tools
# ---------------------------------------------------------------------------


def convert_conversation(
    ex: dict,
) -> tuple[list[dict], list[dict], dict] | None:
    """Raw example -> (messages, tools, meta) or None if it fails cleaning.

    Filters applied here (see module docstring): parseable API list, all
    action names legal, no mid-conversation Finish, last turn = Finish with
    give_answer.
    """
    convs = conv_list(ex)
    if convs[0]["from"] != "system" or convs[1]["from"] != "user":
        return None
    if convs[-1]["from"] != "assistant":
        return None  # trailing turn after the final assistant turn
    apis = parse_api_list(convs[0]["value"])
    if not apis:
        return None
    api_names = {a["name"] for a in apis}

    tools = [convert_tb_tool(a) for a in apis if a["name"] != FINISH_NAME]
    messages: list[dict] = []
    final_answer: str | None = None
    thought_last = ""

    for j, turn in enumerate(convs[1:], start=1):
        role = turn["from"]
        if role == "user":
            content = turn["value"]
            if j == 1 and content.endswith(_BEGIN_SUFFIX):
                content = content[: -len(_BEGIN_SUFFIX)]
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            parsed = parse_assistant_turn(turn["value"])
            if parsed is None:
                return None
            call = parsed["calls"][-1]  # last parseable block = the real call
            is_last_assistant = not any(
                c["from"] == "assistant" for c in convs[j + 1 :]
            )
            if call["name"] == FINISH_NAME:
                # a Finish turn anywhere but the end = mid-conversation
                # Finish (measured ~0.16%) -> drop the conversation
                if not is_last_assistant:
                    return None
                args = call["arguments"]
                if args.get("return_type") != GIVE_ANSWER:
                    return None
                final_answer = args.get("final_answer")
                if not isinstance(final_answer, str) or not final_answer.strip():
                    return None
                messages.append(
                    {"role": "assistant", "content": parsed["thought"] or None}
                )
                thought_last = parsed["thought"]
            else:
                # the executed call must be in this conversation's API list
                # (draft blocks earlier in the turn are never checked: they
                # are retry candidates the model did not actually take)
                if call["name"] not in api_names:
                    return None
                messages.append(
                    {
                        "role": "assistant",
                        "content": parsed["thought"] or None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call["arguments"],
                                },
                            }
                        ],
                    }
                )
        elif role == "function":
            # raw envelope verbatim (may be truncated by the mirror)
            prev = messages[-1] if messages else None
            if not (prev and prev.get("tool_calls")):
                return None  # function turn without a pending call
            name = prev["tool_calls"][-1]["function"]["name"]
            messages.append(
                {"role": "tool", "name": name, "content": turn["value"]}
            )
        else:
            return None

    if final_answer is None:
        return None
    # fold the final answer into the last assistant message's content:
    # thought prose first, then the answer (region-4 spans just the answer).
    # build_record locates the answer via rindex, so meta carries the
    # stripped form — the content fold strips edge whitespace.
    messages[-1]["content"] = (thought_last + "\n\n" + final_answer).strip()
    final_answer = final_answer.strip()

    meta = {
        "query": norm_query(ex["id"]),
        "n_turns_raw": len(convs),
        "n_tools": len(tools),
        "tool_names": sorted({t["function"]["name"] for t in tools}),
        "final_answer": final_answer,
    }
    return messages, tools, meta


# ---------------------------------------------------------------------------
# Stage 1: clean pass — filters + dedupe (plan.md §3.2)
# ---------------------------------------------------------------------------


def _load_train_ds():
    return load_from_disk(RAW_DIR)["train"]


def run_clean(limit: int | None = None) -> dict:
    """Parse + filter + dedupe all raw conversations -> clean index file.

    Dedupe keeps the FIRST occurrence of each normalized query (deterministic
    dataset order); near-duplicates hit the same normalized key and are
    dropped by the same mechanism.
    """
    ds = _load_train_ds()
    n = len(ds) if limit is None else limit
    seen: set[str] = set()
    kept: list[dict] = []
    dropped = Counter()
    for i in range(n):
        res = convert_conversation(ds[i])
        if res is None:
            dropped["filter"] += 1
            continue
        msgs, tools, meta = res
        q = meta["query"]
        if q in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(q)
        kept.append(
            {
                "raw_idx": i,
                "query": q,
                "tool_names": meta["tool_names"],
                "n_turns_raw": meta["n_turns_raw"],
                "n_messages": len(msgs),
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = OUT_DIR / "clean_index.json"
    idx_path.write_text(json.dumps(kept))
    stats = {
        "n_raw": n,
        "n_clean": len(kept),
        "kept_pct": round(100 * len(kept) / n, 1),
        "dropped": dict(dropped),
        "n_distinct_tools": len(
            {t for k in kept for t in k["tool_names"]}
        ),
    }
    (OUT_DIR / "clean_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# Stage 2: TB-500 rare-tool carve (plan.md §3.2)
# ---------------------------------------------------------------------------


def run_carve() -> dict:
    """Greedy rare-tool carve: rank clean convs by summed tool rarity.

    Greedy picks the 500 conversations whose tools have the smallest total
    usage count across the clean pool. All tools appearing in the eval set
    are then excluded from the prefix pool by construction.
    """
    kept = json.loads((OUT_DIR / "clean_index.json").read_text())
    tool_count: Counter[str] = Counter()
    for k in kept:
        tool_count.update(k["tool_names"])
    order = sorted(
        range(len(kept)),
        key=lambda i: sum(tool_count[t] for t in kept[i]["tool_names"]),
    )
    eval_idx = set(order[:EVAL_SIZE])
    eval_tools = set().union(
        *[set(kept[i]["tool_names"]) for i in eval_idx]
    )
    prefix_idx = [
        i for i in range(len(kept)) if not set(kept[i]["tool_names"]) & eval_tools
    ]

    (OUT_DIR / "eval_idx.json").write_text(json.dumps(sorted(eval_idx)))
    (OUT_DIR / "prefix_idx.json").write_text(json.dumps(prefix_idx))
    (OUT_DIR / "heldout_tools.json").write_text(json.dumps(sorted(eval_tools)))

    stats = {
        "n_clean": len(kept),
        "eval_size": len(eval_idx),
        "heldout_tools": len(eval_tools),
        "prefix_pool": len(prefix_idx),
        "prefix_pool_pct": round(100 * len(prefix_idx) / len(kept), 1),
        "seed": CARVE_SEED,
    }
    (OUT_DIR / "carve_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# Stage 3: build — render + tokenize with region maps
# ---------------------------------------------------------------------------

_SPLIT_FILE = {"eval": "eval_idx.json", "prefixes": "prefix_idx.json"}


def build_split(split: str, limit: int | None = None) -> dict:
    """Render + tokenize one split into a HF dataset on disk."""
    ds = _load_train_ds()
    kept = json.loads((OUT_DIR / "clean_index.json").read_text())
    idx = json.loads((OUT_DIR / _SPLIT_FILE[split]).read_text())
    if limit:
        idx = idx[:limit]
    records = []
    for i in idx:
        msgs, tools, meta = convert_conversation(ds[kept[i]["raw_idx"]])
        rec = build_record(
            msgs, tools, query_id=kept[i]["raw_idx"], final_answer_text=meta["final_answer"]
        )
        rec["tool_names"] = meta["tool_names"]
        rec["query"] = meta["query"]
        records.append(rec)
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(out_dir)
    return compute_stats(split)


def compute_stats(split: str) -> dict:
    d = load_from_disk(OUT_DIR / split)
    lens = np.array([len(x) for x in d["input_ids"]])
    tool_region = np.array(
        [sum(n for c, n in x if c in (2, 3)) for x in d["regions"]]
    )
    fa_region = np.array(
        [sum(n for c, n in x if c == 4) for x in d["regions"]]
    )
    return {
        "split": split,
        "n": len(d),
        "tokens_total": int(lens.sum()),
        "n_tokens_median": float(np.median(lens)),
        "n_tokens_p25": float(np.percentile(lens, 25)),
        "n_tokens_p95": float(np.percentile(lens, 95)),
        "n_tokens_max": int(lens.max()),
        "pct_tokens_tool_call_region": round(100 * tool_region.sum() / lens.sum(), 1),
        "pct_tokens_final_answer_region": round(100 * fa_region.sum() / lens.sum(), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd", choices=["clean", "carve", "build", "stats"]
    )
    ap.add_argument("--split", choices=["eval", "prefixes"], default="eval")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "clean":
        print(json.dumps(run_clean(args.limit), indent=2))
    elif args.cmd == "carve":
        print(json.dumps(run_carve(), indent=2))
    elif args.cmd == "build":
        print(json.dumps(build_split(args.split, args.limit), indent=2))
    else:
        print(json.dumps(compute_stats(args.split), indent=2))


if __name__ == "__main__":
    main()
