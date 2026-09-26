"""xLAM prep: carve held-out-function eval + training pool, render through the
target's chat template, tokenize with region maps (plan.md §3.1, §6.2).

Pipeline per example (all 60k usable, verified 2026-09-26):
  raw (query, tools, answers)
    -> convert tools to OpenAI-style schemas (xLAM's native format is a
       name->spec mapping with python-ish type strings)
    -> messages = [user query, assistant tool_calls(answers)]
    -> render via target's apply_chat_template (dict-form arguments!)
    -> tokenize with region labels (tool-call vs prose)

Carve design (measured, plan.md §3.1): functions repeat heavily (median 26
examples/function), so the held-out set must come from the *rarest* functions.
Rank examples by summed function frequency; the 500 rarest become xLAM-500;
every example using any of those functions is excluded from the training pool.

Usage:
  xlam_prep.py carve                 # write eval/train index files
  xlam_prep.py build --split train   # render + tokenize training pool
  xlam_prep.py build --split eval    # render + tokenize the eval set
  xlam_prep.py stats --split train   # length/region distribution report
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

from src.data_prep.render import (
    REGION_CONTEXT,
    REGION_FINAL_ANSWER,
    REGION_PROSE,
    REGION_TAG,
    REGION_TOOL_CALL,
    build_record,
    find_tag_spans,
    get_tokenizer,
    get_tool_call_tags,
    render_context,
    render_sequence,
    tokenize_with_regions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "xlam"
OUT_DIR = REPO_ROOT / "data" / "processed" / "xlam"
EVAL_SIZE = 500
CARVE_SEED = 42

# ---------------------------------------------------------------------------
# Schema conversion: xLAM native -> OpenAI-style
# ---------------------------------------------------------------------------

_PRIMITIVE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _convert_type(type_str: str) -> dict:
    """Map an xLAM python-ish type string to a JSON-Schema type descriptor.

    Total inventory (60k examples, verified 2026-09-26): str/int/float/bool
    with ', optional' / ', default=...' suffixes; List[...], List[List[...]],
    List[Tuple[...]], Tuple[...], Union, Callable, Dict, set. Containers and
    exotic forms map to array/string — their original annotation is preserved
    in the parameter description so the target still sees it.
    """
    base = re.split(r",\s*(?:optional|default)", type_str.strip())[0].strip()
    lowered = base.lower()
    if lowered in _PRIMITIVE:
        return {"type": _PRIMITIVE[lowered]}
    if lowered in ("list", "set", "tuple") or re.match(r"^(List|Tuple)\[", base, re.I):
        return {"type": "array"}
    if lowered == "dict":
        return {"type": "object"}
    return {"type": "string"}  # Union / Callable / anything exotic


def _extract_default(type_str: str, json_type: str):
    """Pull a default value out of an xLAM type annotation, if present.

    Forms in the data (60k inventory, 2026-09-26): "int, optional, default=100",
    "str, optional, default 'London'", "str, optional, default='20'". Numbers
    are coerced for integer/number types; everything else stays a string.
    """
    m = re.search(r"default\s*=?\s*('([^']*)'|([^,]+))", type_str)
    if not m:
        return None
    if m.group(2) is not None:
        val: str | int | float = m.group(2)
    else:
        val = m.group(3).strip()
    if json_type in ("integer", "number"):
        try:
            val = int(val) if json_type == "integer" else float(val)
        except ValueError:
            pass
    return val


def convert_xlam_tool(raw: dict) -> dict:
    """Convert one xLAM tool record to an OpenAI function-calling schema.

    Optionality rule (measured, see repo docs): a parameter is required only
    if its type string has no optional/default marker AND it carries no
    `default` key. xLAM's own answers omit default-bearing "required" params
    ~48% of the time, so default => optional strictly reduces schema-violating
    renders. Original type annotations are kept in the description so the
    target model still sees e.g. 'List[int]'. Defaults may come either as a
    separate `default` key or embedded in the type string ("int, default=100").
    """
    props: dict = {}
    required: list[str] = []
    for pname, spec in raw.get("parameters", {}).items():
        if not isinstance(spec, dict):  # degenerate: bare type string
            spec = {"type": str(spec)}
        type_str = spec.get("type", "str")
        descriptor = _convert_type(type_str)
        desc = (spec.get("description") or "").strip()
        if type_str not in _PRIMITIVE:  # keep the original annotation visible
            desc = f"{desc} [type: {type_str}]".strip()
        if desc:
            descriptor["description"] = desc
        default = spec.get("default", _extract_default(type_str, descriptor["type"]))
        if default is not None:
            descriptor["default"] = default
        if "optional" in type_str or "default" in type_str or default is not None:
            pass  # optional
        else:
            required.append(pname)
        props[pname] = descriptor

    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return {
        "type": "function",
        "function": {
            "name": raw["name"],
            "description": raw.get("description", ""),
            "parameters": schema,
        },
    }


# ---------------------------------------------------------------------------
# Carve (plan.md §3.1: split by tool, rarest functions)
# ---------------------------------------------------------------------------


def build_carve() -> dict:
    ds = load_from_disk(RAW_DIR)["train"]
    tools_per: list[tuple[str, ...]] = []
    fn_examples: dict[str, set[int]] = defaultdict(set)
    for i, ex in enumerate(ds):
        names = tuple(sorted(t["name"].lower() for t in json.loads(ex["tools"])))
        tools_per.append(names)
        for n in set(names):
            fn_examples[n].add(i)
    freq = {n: len(v) for n, v in fn_examples.items()}

    order = sorted(range(len(ds)), key=lambda i: sum(freq[n] for n in tools_per[i]))
    eval_idx = set(order[:EVAL_SIZE])
    eval_fns = set().union(*[set(tools_per[i]) for i in eval_idx])
    excluded = set().union(*[fn_examples[n] for n in eval_fns])
    train_idx = set(range(len(ds))) - excluded
    assert not any(set(tools_per[i]) & eval_fns for i in train_idx), "leak!"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_idx.json").write_text(json.dumps(sorted(eval_idx)))
    (OUT_DIR / "train_idx.json").write_text(json.dumps(sorted(train_idx)))
    (OUT_DIR / "heldout_functions.json").write_text(json.dumps(sorted(eval_fns)))

    stats = {
        "n_examples": len(ds),
        "eval_size": len(eval_idx),
        "heldout_functions": len(eval_fns),
        "train_pool": len(train_idx),
        "train_pool_pct": round(100 * len(train_idx) / len(ds), 1),
        "seed": CARVE_SEED,
    }
    (OUT_DIR / "carve_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _load_split_idx() -> dict[str, set[int]]:
    return {
        "eval": set(json.loads((OUT_DIR / "eval_idx.json").read_text())),
        "train": set(json.loads((OUT_DIR / "train_idx.json").read_text())),
    }


# ---------------------------------------------------------------------------
# Build (render + tokenize with region maps)
# ---------------------------------------------------------------------------


def build_messages(ex: dict) -> tuple[list[dict], list[dict]]:
    """xLAM example -> (messages, openai_tools) for apply_chat_template.

    Single-turn: user query, then one assistant turn holding all answer calls
    (52.6% are multi-call — they become parallel calls in one turn, which the
    target's template renders as consecutive tool-call blocks).
    """
    tools = [convert_xlam_tool(t) for t in json.loads(ex["tools"])]
    answers = json.loads(ex["answers"])
    msgs = [
        {"role": "user", "content": ex["query"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": a["name"], "arguments": a["arguments"]},
                }
                for a in answers
            ],
        },
    ]
    return msgs, tools


def process_example(ex: dict) -> dict:
    """Full transform for one xLAM example -> training/eval record."""
    msgs, tools = build_messages(ex)
    rec = build_record(msgs, tools, query_id=ex["id"])
    rec["n_answers"] = len(json.loads(ex["answers"]))
    return rec


def build_split(split: str, limit: int | None = None) -> dict:
    ds = load_from_disk(RAW_DIR)["train"]
    idx = sorted(_load_split_idx()[split])
    if limit:
        idx = idx[:limit]
    records = [process_example(ds[i]) for i in idx]
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(out_dir)
    return compute_stats(split)


def compute_stats(split: str) -> dict:
    d = load_from_disk(OUT_DIR / split)
    lens = np.array([len(x) for x in d["input_ids"]])
    tool_region = np.array(
        [sum(n for c, n in x if c in (REGION_TOOL_CALL, REGION_TAG)) for x in d["regions"]]
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["carve", "build", "stats"])
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "carve":
        print(json.dumps(build_carve(), indent=2))
    elif args.cmd == "build":
        print(json.dumps(build_split(args.split, args.limit), indent=2))
    else:
        print(json.dumps(compute_stats(args.split), indent=2))


if __name__ == "__main__":
    main()
