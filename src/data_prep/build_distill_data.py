"""Stage-1 KD data build (plan.md §2 Stage 1, §6.7 first half).

Turns the generation JSONL produced by src/serving/gen_stage1.py (the
frozen target's greedy continuations on the 5,000 frozen Stage-1 xLAM
contexts, top-k logprobs per generated token) into the supervised-KD
dataset consumed by training/kd_warmstart.py (§6.7 second half). Pure
data side — vLLM-free and torch-free, golden-tested on the mac dev host.
It also runs on the GPU host right after generation: prompts are rendered
fresh from raw xLAM (downloaded + checksum-verified by setup_day0 /
freeze_splits verify); data/processed is NOT needed there.

Generation JSONL schema (one line per prompt, frozen order — the
committed source of truth; the assembled HF dataset is derived and
gitignored, rebuildable with one command):

  {
    "query_id": <raw xLAM index, frozen/stage1_idx.json order>,
    "output_ids": [t, ...],          # exactly what the target emitted; the
                                     # terminal EOS is ALWAYS included when
                                     # the model stopped (gen_stage1
                                     # normalizes engine differences)
    "logprobs": [[[id, logprob], ...] | null, ...],
                                     # parallel to output_ids; per position
                                     # the target's top-k entries sorted by
                                     # logprob DESC then id ASC; null = no
                                     # teacher distribution for that
                                     # position (the withheld-EOS case)
    "finish_reason": "stop" | "length",
  }

Pinned conventions (golden-tested in tests/test_build_distill_data.py;
the generation side's E1-E5 live in gen_stage1.py):

  G1 greedy: the target sampled at temperature 0.0 — output_ids are the
     target's argmax sequence (deployment mode for tool calling).
  G2 logprob fidelity: entries are stored exactly as the engine returned
     them (floats verbatim), sorted by logprob DESC then id ASC;
     ragged per-position lengths are preserved, never padded or truncated.
  G3 record schema == the repo's universal record schema
     (input_ids/labels/regions RLE/n_prompt_tokens/n_tokens/query_id, as
     built by render.build_record) so trainers and analyzers consume it
     unchanged; generation extras (logprobs, finish_reason, ...) ride
     along. Labels cover exactly the generated tokens (EOS included when
     present — the draft must learn to stop, same convention as
     build_record's assistant spans).
  G4 drop rule (plan §9 mitigation "target hallucinates argument keys
     not in schema"): a generation is kept iff every tool call is valid
     against the example's own schema — name in the tool list, arguments
     keys a subset of the schema properties, required keys present. Prose
     with no tool call at all, or any parse failure, is dropped. Drop
     rate and per-reason counts are logged, never silently repaired.
  G5 EOS/finish: finish_reason is stored verbatim; labels include the
     emitted terminal EOS when the model stopped.
  G6 determinism: frozen order everywhere — prompts in
     frozen/stage1_idx.json order; --limit N = first N in frozen order;
     assembled records keep that order.
  G7 token round-trip: decoding output_ids must re-tokenize to exactly
     output_ids (Qwen BPE + template scaffolding make this hold); if it
     ever fails, region labeling falls back to prefix-decode offsets and
     the mismatch is counted in stats — labels/ids never guessed.

CLI:
  build_distill_data.py prompts [--limit N]      # render stage-1 contexts,
                                                 # sanity report
  build_distill_data.py assemble <gen.jsonl> --out data/processed/stage1_kd
                                                 # join + validate + build
  build_distill_data.py stats <dir>              # length/region/drop report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from src.data_prep.render import (
    REGION_CONTEXT,
    REGION_PROSE,
    REGION_TAG,
    REGION_TOOL_CALL,
    _rle,
    get_tokenizer,
    get_tool_call_tags,
    render_context,
    tokenize_with_regions,
)
from src.data_prep.toolbench_clean import RAW_DIR as TB_RAW_DIR  # noqa: F401
from src.data_prep.xlam_prep import RAW_DIR as XLAM_RAW_DIR, build_messages

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_IDX = REPO_ROOT / "frozen" / "stage1_idx.json"
OUT_DIR = REPO_ROOT / "data" / "processed" / "stage1_kd"


# ---------------------------------------------------------------------------
# Stage-1 contexts (rendered from raw + the frozen index, G6)
# ---------------------------------------------------------------------------


def _load_raw():
    from datasets import load_from_disk

    return load_from_disk(XLAM_RAW_DIR)["train"]


def stage1_index(limit: int | None = None) -> list[int]:
    """Frozen Stage-1 raw indices, file order; --limit = first N (G6)."""
    idx = json.loads(STAGE1_IDX.read_text())
    if limit is not None:
        idx = idx[:limit]
    if not idx:
        raise SystemExit(f"no Stage-1 indices in {STAGE1_IDX}")
    return idx


def render_stage1_prompt(ex: dict) -> list[int]:
    """One raw xLAM example -> its Stage-1 generation prompt as token ids.

    The context is the user turn only (xLAM is single-turn), rendered
    through the target's template with a fresh assistant header — the
    token-level prefix property is golden-tested (tests/test_golden_render
    .test_prefix_property_tokens_user_only) and asserted against the
    already-rendered train pool in tests/test_build_distill_data.py.
    """
    msgs, tools = build_messages(ex)
    text = render_context(msgs[:1], tools)
    return get_tokenizer()(text, add_special_tokens=False)["input_ids"]


def load_stage1_prompts(limit: int | None = None) -> list[dict]:
    """Stage-1 prompts [{"query_id", "prompt_ids"}], frozen order (G6)."""
    raw = _load_raw()
    out = []
    for i in stage1_index(limit):
        out.append({
            "query_id": i,
            "prompt_ids": render_stage1_prompt(raw[i]),
        })
    return out


# ---------------------------------------------------------------------------
# Generation validation (G4)
# ---------------------------------------------------------------------------

# The target's ACTUAL greedy behavior on xLAM prompts (measured 2026-09-27
# on the rented H100, 5000 frozen Stage-1 contexts): 92.6% of valid tool
# calls are wrapped in the SCHEMA-announcement tag pair <tools></tools>,
# not the template-instructed assistant pair <tool_call></tool_call>
# (payloads are always schema-valid). The template's instruction and
# example demonstrably do not flip this at greedy -- verified by A/B
# (default vs explicit system prompts: both leave the wrapper wrong,
# payloads always right). The KD data must mirror what the target will
# actually VERIFY at inference, so both wrapper pairs are accepted
# everywhere below: payloads are validated, wrappers are never repaired
# (a repaired tag would teach the draft to propose tokens the target
# rejects). The deviation itself is a reported finding, not a silent fix.
# Convention G8 (dual wrapper, golden-tested in test_build_distill_data):
#   a tool call in a generation is <open><payload><close> for EITHER
#   pair; nested pairs are outermost-matched (text inside a payload is
#   payload, not a nested call).

_ALT_OPEN, _ALT_CLOSE = '<tools>', '</tools>'


def gen_tag_pairs(tags):
    """The wrapper pairs recognized in generations (G8): the template's
    # assistant pair plus the schema-announcement pair the target actually
    # emits at greedy. Order: assistant pair first (canonical)."""
    return [
        (tags.open_tag, tags.close_tag),
        (_ALT_OPEN, _ALT_CLOSE),
    ]


def find_gen_payloads(text: str, tags) -> list[tuple[int, int]]:
    """Char spans of tool-call JSON payloads in a *generation* (G8).

    Unlike ToolCallTags.find_calls (which scans assistant turns of a full
    render), a generation has no assistant headers yet -- the tags appear
    directly in the generated text -- so the scan is flat over `text`.
    BOTH wrapper pairs are recognized (the template's assistant pair and
    the schema-announcement pair the target actually emits); each pair is
    matched outermost-first, so text inside a payload never double-counts.
    Unterminated calls yield no span (their trailing text is prose for
    region purposes, and the record fails G4 validation anyway).
    """
    raw: list[tuple[int, int]] = []
    for open_tag, close_tag in gen_tag_pairs(tags):
        pos = 0
        while True:
            s = text.find(open_tag, pos)
            if s < 0:
                break
            payload_start = s + len(open_tag)
            e = text.find(close_tag, payload_start)
            if e < 0:
                break
            if payload_start < e:  # skip empty pairs
                raw.append((payload_start, e))
            pos = e + len(close_tag)
    # outermost-match (G8): collect ALL pairs' spans first, then keep a
    # span only if no other span strictly contains it (text inside a
    # payload is payload, not a nested call -- e.g. a canonical pair
    # occurring inside a schema-wrapped payload must not double-count)
    kept: list[tuple[int, int]] = []
    for span in sorted(raw):
        if not any(a <= span[0] and span[1] <= b and (a, b) != span
                   for a, b in raw):
            kept.append(span)
    return kept


def find_gen_tag_spans(text: str, tags) -> list[tuple[int, int]]:
    """Char spans of every open/close tag occurrence in a generation (G8:
    both wrapper pairs -- tag tokens are tag-region regardless of which
    pair the target used)."""
    spans = []
    for open_tag, close_tag in gen_tag_pairs(tags):
        for tag in (open_tag, close_tag):
            pos = 0
            while True:
                i = text.find(tag, pos)
                if i < 0:
                    break
                spans.append((i, i + len(tag)))
                pos = i + len(tag)
    return spans


def _schema_props(tools: list[dict]) -> dict[str, dict]:
    return {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}


_JSON_DEC = json.JSONDecoder()


def _parse_call_sequence(payload: str) -> list | None:
    """Parse a wrapper payload as a sequence of JSON values (G8: parallel
    calls). Returns the list of decoded values, or None on the first
    parse failure / trailing garbage (the record fails G4, not repaired)."""
    vals = []
    i = 0
    n = len(payload)
    dec = _JSON_DEC
    while True:
        while i < n and payload[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            v, j = dec.raw_decode(payload, i)
        except ValueError:
            return None
        vals.append(v)
        i = j
    return vals if vals else None


def validate_generation(text: str, tools: list[dict], tags) -> dict:
    """G4: schema-validate every tool call in a generation against the
    example's own schemas. Valid iff there is >=1 call and every call has
    a known name, argument keys within the schema properties, and all
    required keys present. Returns {valid, n_calls, problems}."""
    problems: list[str] = []
    payloads = []
    for s, e in find_gen_payloads(text, tags):
        payloads.append(text[s:e])
    if not payloads:
        return {"valid": False, "n_calls": 0,
                "problems": ["no tool call in generation"]}
    props = _schema_props(tools)
    names = set(props)
    for p in payloads:
        # G8 (parallel calls): one wrapper may hold a SEQUENCE of JSON
        # objects (xLAM is 52.6% multi-call; the target emits
        # "{call1}\n{call2}" inside a single pair) — parse with
        # raw_decode until the span is exhausted; every object must
        # validate. A non-object or trailing garbage fails the record.
        calls = _parse_call_sequence(p)
        if calls is None:
            problems.append(f"payload not JSON: {p[:60]!r}")
            continue
        for call in calls:
            if not isinstance(call, dict):
                problems.append(f"payload not an object: {str(call)[:60]!r}")
                continue
            name = call.get("name")
            args = call.get("arguments", {})
            if name not in names:
                problems.append(f"unknown function name: {name!r}")
                continue
            if not isinstance(args, dict):
                problems.append(f"arguments not an object for {name!r}")
                continue
            schema = props[name] or {}
            allowed = set(schema.get("properties", {}))
            required = set(schema.get("required", []))
            keys = set(args)
            if keys - allowed:
                problems.append(
                    f"{name}: keys outside schema: {sorted(keys - allowed)}")
            missing = required - keys
            if missing:
                problems.append(f"{name}: missing required: {sorted(missing)}")
    return {"valid": not problems, "n_calls": len(payloads), "problems": problems}


# ---------------------------------------------------------------------------
# Assembly: generation JSONL -> KD dataset (G2, G3, G5, G7)
# ---------------------------------------------------------------------------


def _prefix_decode_offsets(ids: list[int]) -> list[tuple[int, int]]:
    """(start, end) char spans of each token via prefix decoding
    (decode(ids[:i+1]) length) — exact for BPE since decode concatenates
    pieces. G7 fallback."""
    tok = get_tokenizer()
    ends = [len(tok.decode(ids[: i + 1])) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def _gen_token_offsets(ids: list[int]) -> tuple[list[tuple[int, int]], bool]:
    """(start, end) char spans of every generated token; `exact` iff
    re-tokenizing the decoded text reproduces `ids` (G7's check)."""
    tok = get_tokenizer()
    text = tok.decode(ids)
    re_ids = tok(text, add_special_tokens=False)["input_ids"]
    if re_ids == list(ids):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]], True
    # never observed on this stack; fall back rather than guess (G7)
    return _prefix_decode_offsets(ids), False


def build_gen_record(
    prompt: dict,
    gen: dict,
    tools: list[dict],
    tags,
) -> tuple[dict, dict]:
    """One prompt + generation -> KD record (G3) + per-record stats.

    regions: context (0) over the prompt, then per-token regions for the
    generation (prose / tool-call payload / tag) via offset containment —
    same containment rule as render.tokenize_with_regions.
    """
    p_ids = list(prompt["prompt_ids"])
    g_ids = list(gen["output_ids"])
    n_prompt = len(p_ids)
    tok = get_tokenizer()
    text = tok.decode(g_ids)
    offsets, exact = _gen_token_offsets(g_ids)
    payload_spans = find_gen_payloads(text, tags)
    tag_spans = find_gen_tag_spans(text, tags)

    regions = [REGION_CONTEXT] * n_prompt
    for s, e in offsets:
        r = REGION_PROSE
        for ts, te in tag_spans:
            if ts <= s < te:
                r = REGION_TAG
                break
        if r == REGION_PROSE:
            for ps, pe in payload_spans:
                if ps <= s < pe:
                    r = REGION_TOOL_CALL
                    break
        regions.append(r)

    labels = [-100] * n_prompt + list(g_ids)  # G3/G5: loss on generation incl EOS
    val = validate_generation(text, tools, tags)

    record = {
        "input_ids": p_ids + g_ids,
        "labels": labels,
        "regions": _rle(regions),
        "n_prompt_tokens": n_prompt,
        "n_tokens": n_prompt + len(g_ids),
        "n_tool_calls": len(payload_spans),
        "n_tools": len(tools),
        "query_id": prompt["query_id"],
        "gen_ids": g_ids,
        "finish_reason": gen.get("finish_reason"),
        "gen_logprob_token_ids": [
            [t for t, _ in lp] if lp is not None else None
            for lp in gen.get("logprobs", [])
        ],
        "gen_logprob_values": [
            [v for _, v in lp] if lp is not None else None
            for lp in gen.get("logprobs", [])
        ],
    }
    stats = {
        "query_id": prompt["query_id"],
        "n_gen_tokens": len(g_ids),
        "valid": val["valid"],
        "problems": val["problems"],
        "roundtrip_exact": exact,
        "finish_reason": gen.get("finish_reason"),
    }
    return record, stats


def assemble(gens: list[dict], limit: int | None = None) -> tuple[list[dict], dict]:
    """Join generations to prompts (by query_id, 1:1, frozen order), build
    records, apply G4 drops. Returns (records, stats)."""
    idx = stage1_index(limit)
    if len(gens) != len(idx):
        raise SystemExit(
            f"generation count mismatch: {len(gens)} lines vs {len(idx)} "
            f"Stage-1 contexts (re-run generation with the same --limit)"
        )
    gens_by_qid = {}
    for g in gens:
        if g["query_id"] in gens_by_qid:
            raise SystemExit(f"duplicate query_id in generations: {g['query_id']}")
        gens_by_qid[g["query_id"]] = g
    if set(gens_by_qid) != set(idx):
        raise SystemExit(
            "generation query_ids do not match the Stage-1 index — regenerate"
        )

    raw = _load_raw()
    tags = get_tool_call_tags()
    records, per_stats, drops = [], [], []
    for i, g in zip(idx, gens):
        if g["query_id"] != i:
            raise SystemExit(
                f"generation order desync at {i}: got {g['query_id']} (G6)"
            )
        ex = raw[i]
        _, tools = build_messages(ex)
        rec, st = build_gen_record(
            {"query_id": i, "prompt_ids": render_stage1_prompt(ex)},
            g, tools, tags,
        )
        per_stats.append(st)
        if st["valid"]:
            records.append(rec)
        else:
            drops.append(st)

    n = len(per_stats)
    reasons: dict[str, int] = {}
    for st in drops:
        for p in st["problems"]:
            key = p.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    stats = {
        "n_stage1": n,
        "n_kept": len(records),
        "n_dropped": len(drops),
        "drop_rate": round(len(drops) / n, 4) if n else None,
        "drop_reasons": reasons,
        "roundtrip_exact_all": all(s["roundtrip_exact"] for s in per_stats),
        "finish_reasons": {
            fr: sum(1 for s in per_stats if s["finish_reason"] == fr)
            for fr in ("stop", "length")
        },
        "gen_tokens_total": sum(s["n_gen_tokens"] for s in per_stats),
    }
    return records, stats


def load_gens(path: str) -> list[dict]:
    """Read a generation JSONL, validating it against the schema."""
    gens = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            for k in ("query_id", "output_ids", "logprobs", "finish_reason"):
                if k not in g:
                    raise ValueError(f"gen line {ln}: missing {k}")
            if not isinstance(g["output_ids"], list) or not g["output_ids"]:
                raise ValueError(f"gen line {ln}: empty output_ids")
            if len(g["logprobs"]) != len(g["output_ids"]):
                raise ValueError(
                    f"gen line {ln}: logprobs ({len(g['logprobs'])}) != "
                    f"output_ids ({len(g['output_ids'])})"
                )
            gens.append(g)
    if not gens:
        raise ValueError(f"no generations in {path}")
    return gens


def run_assemble(gen_path: str, out_dir: str, limit: int | None = None) -> dict:
    gens = load_gens(gen_path)
    records, stats = assemble(gens, limit)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(out)
    stats["n_records"] = len(records)
    (out / "assemble_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def compute_stats(out_dir: str) -> dict:
    from datasets import load_from_disk

    d = load_from_disk(out_dir)
    lens = np.array([len(x) for x in d["input_ids"]])
    gens = np.array([len(x) for x in d["gen_ids"]])
    by_region: dict[int, int] = {}
    for rle in d["regions"]:
        for c, n in rle:
            by_region[c] = by_region.get(c, 0) + n
    total = int(lens.sum())
    return {
        "n": len(d),
        "tokens_total": total,
        "n_gen_median": float(np.median(gens)),
        "n_gen_p95": float(np.percentile(gens, 95)),
        "n_gen_max": int(gens.max()),
        "n_tokens_median": float(np.median(lens)),
        "n_tokens_p95": float(np.percentile(lens, 95)),
        "n_tokens_max": int(lens.max()),
        "region_tokens": {
            {"0": "context", "1": "prose", "2": "tool_call", "3": "tag",
             "4": "final_answer"}[str(c)]: n
            for c, n in sorted(by_region.items())
        },
        "pct_tokens_tool_call_region": round(
            100 * (by_region.get(REGION_TOOL_CALL, 0) + by_region.get(REGION_TAG, 0))
            / total, 1
        ) if total else None,
        "finish_reasons": {
            fr: sum(1 for x in d["finish_reason"] if x == fr)
            for fr in ("stop", "length")
        },
        "logprob_entries": int(
            sum(len(x) for x in d["gen_logprob_token_ids"] if x is not None)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "assemble", "stats"])
    ap.add_argument("path", nargs="?", default=None,
                    help="gen JSONL (assemble) or dataset dir (stats)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N Stage-1 contexts in frozen order (G6)")
    ap.add_argument("--out", default=str(OUT_DIR),
                    help="output dataset dir (assemble)")
    args = ap.parse_args()
    if args.cmd == "prompts":
        ps = load_stage1_prompts(args.limit)
        lens = np.array([len(p["prompt_ids"]) for p in ps])
        hdr = get_tokenizer().decode(ps[0]["prompt_ids"])
        print(json.dumps({
            "n_prompts": len(ps),
            "n_tokens_total": int(lens.sum()),
            "n_tokens_median": float(np.median(lens)),
            "n_tokens_p95": float(np.percentile(lens, 95)),
            "n_tokens_max": int(lens.max()),
            "first_prompt_ends_with_header": hdr.endswith("<|im_start|>assistant\n"),
        }, indent=2))
    elif args.cmd == "assemble":
        if not args.path:
            raise SystemExit("assemble needs <gen.jsonl>")
        print(json.dumps(run_assemble(args.path, args.out, args.limit), indent=2))
    else:
        if not args.path:
            raise SystemExit("stats needs <dataset dir>")
        print(json.dumps(compute_stats(args.path), indent=2))


if __name__ == "__main__":
    main()
