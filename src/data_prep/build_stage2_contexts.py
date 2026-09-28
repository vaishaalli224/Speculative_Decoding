"""Stage-2 on-policy context builder (plan.md §2 Stage 2, §3.1/§3.2, §6.7).

Supplies the contexts the Stage-2 trainer samples draft continuations on:
the xLAM training pool minus the Stage-1 carve (single-turn, in-domain) plus
ToolBench clean-trajectory prefixes cut before every assistant turn
(multi-turn, post-observation states xLAM cannot supply). No target
generations are involved — a context is just a rendered prompt ending with
a fresh assistant header; the DRAFT samples from it and the target scores
what it sampled (training/onpolicy_gkd.py's O1/O2).

Pinned conventions (golden-tested in tests/test_build_stage2_contexts.py):

  P1 xLAM contexts = pool minus carve: frozen/xlam_train_pool_idx.json
     (57,794 raw indices) minus frozen/stage1_idx.json (5,000), in pool-file
     order (52,794). Rendered by build_distill_data.render_stage1_prompt —
     user turn + fresh assistant header, token-identical to the Stage-1
     generation prompts and to inference.
  P2 TB contexts: frozen/tb_prefix_idx.json (45,023 RAW ToolBench indices —
     freeze_splits already dereferenced the clean-index positions) ->
     convert_conversation -> build_record, then ONE CONTEXT PER
     ASSISTANT-TURN START via the analyzer's _turn_starts_from_labels:
     ctx = input_ids[:start]. Post-observation states are included by
     construction (a cut after a tool-response block is a valid sampling
     point). Every context must end with a fresh assistant header —
     asserted (the last <|im_start|> sits within a few tokens of the end),
     never assumed.
  P3 cap + left-truncation (plan §2.6 "cap ~4-6k, left-truncate oldest
     turns"): --max-ctx 4096. An over-cap context is cut at the LARGEST
     <|im_start|> turn boundary that fits — oldest turns drop whole, never
     a mid-turn cut — while block 0 (the system block, which carries the
     tool schemas) is always preserved. DEVIATION from §2.6's parenthetical
     "trim tool lists": schemas are never trimmed — a trimmed schema list
     would desync what the target scores from what inference serves; the
     turn-boundary cut covers the cap's intent. If even system + one turn +
     header exceeds the cap the context is SKIPPED and counted, never
     mangled. xLAM contexts are never truncated (single-turn: the only
     droppable block would be the query itself) — over-cap xLAM is skipped
     (unreachable at any sane cap; measured pool max ~1.1k tokens).
  P4 interleave 1:1 by conversation (plan §10 default): xLAM and TB
     conversations alternate in frozen order, a TB conversation's multiple
     turn-contexts staying adjacent; when one pool exhausts the stream
     drains the other. Deterministic.
  P5 laziness: the main entry point is a GENERATOR — the GPU day consumes
     ~1-3k of the ~97k-context pool, and nothing renders that is not
     consumed (a TB conversation render costs a full build_record).
     build_stage2_contexts(limit) is the list wrapper for tests/CLI.
  P6 length logging (plan §2 "log the length distribution of the actual
     mix during local prep"): the stats CLI reports the consumed mix's
     length distribution, overall and per source, plus truncation/skip
     counts, into a stats dict the trainer embeds in train_meta.json.

Context dicts: {"query_id" (raw index), "source": "xlam"|"tb",
"turn" (TB turn index; None for single-turn xLAM), "prompt_ids"}.

GPU-host note (mirrors build_distill_data): everything renders fresh from
raw + frozen/ — data/processed is never needed on the GPU host (the frozen
TB prefix file holds raw indices directly, so even the clean_index.json
join table is unnecessary here).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.eval_acceptance import _turn_starts_from_labels
from src.data_prep.render import IM_START, build_record, get_tokenizer
from src.data_prep.toolbench_clean import (
    RAW_DIR as TB_RAW_DIR,
    convert_conversation,
)
from src.data_prep.build_distill_data import _load_raw, render_stage1_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
XLAM_POOL_IDX = REPO_ROOT / "frozen" / "xlam_train_pool_idx.json"
STAGE1_IDX = REPO_ROOT / "frozen" / "stage1_idx.json"
TB_PREFIX_IDX = REPO_ROOT / "frozen" / "tb_prefix_idx.json"

DEFAULT_MAX_CTX = 4096  # plan §2.6: "cap at ~4-6k tokens"
DEFAULT_SOURCES = "xlam,tb"  # P4 order: xLAM conversation first
_HEADER_MAX_TOKENS = 4  # '<|im_start|>assistant\n' tokenizes to <= this many


# ---------------------------------------------------------------------------
# Pure cores (torch-free, tokenizer-free — tests pass sentinel ids)
# ---------------------------------------------------------------------------


def xlam_stage2_idx(pool: list[int], stage1: list[int]) -> list[int]:
    """Pool-minus-carve in pool-file order (P1); a stage1 id outside the
    pool means the frozen files desynced — hard error, never a silent gap."""
    ps, s1 = set(pool), set(stage1)
    if not s1.issubset(ps):
        raise AssertionError(
            f"{len(s1 - ps)} Stage-1 indices are outside the training pool — "
            "frozen/xlam_train_pool_idx.json and frozen/stage1_idx.json "
            "desynced; re-run freeze_splits"
        )
    return [i for i in pool if i not in s1]


def left_truncate(ctx: list[int], max_ctx: int, im_start_id: int) -> list[int] | None:
    """P3: drop oldest whole turns until the context fits, keeping block 0.

    Cut candidates are the positions of im_start_id tokens, EXCLUDING the
    final one (that is the assistant header itself — cutting there would
    leave schemas + a bare header, a context with no conversation content).
    j scans from the SMALLEST kept suffix (all conversation blocks dropped,
    keeping system + header only) upward, so the FIRST fit keeps the MOST
    recent turns. The kept sequence is ctx[:bounds[1]] + ctx[bounds[j]:] —
    block 0 plus the fitting tail — which is exactly a render of the
    truncated conversation (each block is self-contained: starts with
    <|im_start|>, ends with <|im_end|> + newline). Returns None when even
    the tightest legal cut does not fit (caller skips + counts).
    """
    if len(ctx) <= max_ctx:
        return list(ctx)
    bounds = [i for i, t in enumerate(ctx) if t == im_start_id]
    # need: a 2nd block (block 0 ends there) + at least one keepable turn
    # block before the header
    for j in range(1, len(bounds) - 1):
        kept_len = bounds[1] + (len(ctx) - bounds[j])
        if kept_len <= max_ctx:
            return ctx[: bounds[1]] + ctx[bounds[j]:]
    return None


def _assert_generation_header(ctx: list[int], im_start_id: int) -> None:
    """P2: the cut must end with a fresh assistant header. The last
    im_start_id in the context is the header's first token, a few positions
    from the end; a last im_start far from the end means the labels do not
    start right after the header — a data bug, not a truncatable case."""
    if not ctx:
        raise AssertionError("empty context — labels start at position 0?")
    last = len(ctx) - 1 - ctx[::-1].index(im_start_id)
    if len(ctx) - last > _HEADER_MAX_TOKENS:
        raise AssertionError(
            f"context does not end with an assistant header (last im_start "
            f"{_HEADER_MAX_TOKENS}+ tokens from the end) — the record's "
            "labels are not aligned to the template header (P2)"
        )


def tb_contexts_from_record(
    record: dict, max_ctx: int, im_start_id: int
) -> tuple[list[dict], dict]:
    """One rendered TB record -> (contexts, stats) (P2/P3). One context per
    assistant-turn start; each is header-terminated, capped and possibly
    left-truncated; over-cap-and-unfittable turns are skipped and counted."""
    starts = _turn_starts_from_labels(record["labels"])
    contexts: list[dict] = []
    n_truncated = n_skipped = 0
    for t, s in enumerate(starts):
        ctx = list(record["input_ids"])[:s]
        _assert_generation_header(ctx, im_start_id)
        if len(ctx) > max_ctx:
            cut = left_truncate(ctx, max_ctx, im_start_id)
            if cut is None:
                n_skipped += 1
                continue
            ctx = cut
            n_truncated += 1
        contexts.append({
            "query_id": record.get("query_id"),
            "source": "tb",
            "turn": t,
            "prompt_ids": ctx,
        })
    return contexts, {
        "n_turns": len(starts),
        "n_kept": len(contexts),
        "n_truncated": n_truncated,
        "n_skipped": n_skipped,
    }


def interleave(groups_a, groups_b):
    """P4: alternate conversation groups from two lazy iterables, draining
    whichever survives. Group = a TB conversation's contexts or a 1-element
    xLAM group; contexts within a group stay adjacent."""
    it_a, it_b = iter(groups_a), iter(groups_b)
    while True:
        moved = False
        for it in (it_a, it_b):
            try:
                group = next(it)
            except StopIteration:
                continue
            moved = True
            yield from group
        if not moved:
            return


def get_im_start_id(tokenizer=None) -> int:
    """The <|im_start|> token id from the tokenizer — never hardcoded."""
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    tid = tok.convert_tokens_to_ids(IM_START)
    if tid is None or tok.convert_ids_to_tokens(int(tid)) != IM_START:
        raise RuntimeError(f"{IM_START!r} is not a token in this tokenizer")
    return int(tid)


# ---------------------------------------------------------------------------
# Lazy source iterators (render from raw + frozen; P5)
# ---------------------------------------------------------------------------


def new_stats() -> dict:
    """Counters updated in place as the stream is consumed (P6). n_xlam /
    n_tb count CONSUMED contexts (stream level); n_tb_conv and the
    truncation/skip counters count what the source iterators SAW up to the
    point the stream stopped — a limit-stop mid-TB-group means the rest of
    that group was rendered but not consumed, and the counters say so
    honestly."""
    return {
        "n_contexts": 0,
        "n_xlam": 0,
        "n_xlam_skipped_over_cap": 0,
        "n_tb": 0,
        "n_tb_conv": 0,
        "n_tb_conv_no_contexts": 0,
        "n_truncated": 0,
        "n_skipped_over_cap": 0,
    }


def iter_xlam_groups(max_ctx: int, stats: dict, idx: list[int] | None = None):
    """Yield 1-element groups of xLAM contexts in pool-minus-carve order."""
    raw = _load_raw()
    for i in idx if idx is not None else xlam_stage2_idx(
        json.loads(XLAM_POOL_IDX.read_text()),
        json.loads(STAGE1_IDX.read_text()),
    ):
        ids = render_stage1_prompt(raw[i])
        if len(ids) > max_ctx:  # single-turn: nothing legal to drop (P3)
            stats["n_xlam_skipped_over_cap"] += 1
            continue
        yield [{
            "query_id": i,
            "source": "xlam",
            "turn": None,
            "prompt_ids": list(ids),
        }]


def iter_tb_groups(
    max_ctx: int,
    im_start_id: int,
    stats: dict,
    raw_indices: list[int] | None = None,
):
    """Yield one group (a conversation's turn contexts) per prefix entry.

    frozen/tb_prefix_idx.json already holds RAW ToolBench indices
    (freeze_splits dereferenced the clean-index positions when freezing),
    so no clean_index join is needed — every index converts directly."""
    from datasets import load_from_disk

    ds = load_from_disk(TB_RAW_DIR)["train"]
    indices = (
        raw_indices
        if raw_indices is not None
        else json.loads(TB_PREFIX_IDX.read_text())
    )
    for raw_idx in indices:
        res = convert_conversation(ds[raw_idx])
        if res is None:  # the frozen pool was built from conversion
            raise AssertionError(  # successes — a re-conversion failure = desync
                f"raw index {raw_idx} no longer converts — raw data differs "
                "from the pass that built frozen/tb_prefix_idx.json"
            )
        msgs, tools, meta = res
        rec = build_record(
            msgs, tools, query_id=raw_idx, final_answer_text=meta["final_answer"]
        )
        contexts, st = tb_contexts_from_record(rec, max_ctx, im_start_id)
        stats["n_tb_conv"] += 1
        stats["n_truncated"] += st["n_truncated"]
        stats["n_skipped_over_cap"] += st["n_skipped"]
        if not contexts:
            stats["n_tb_conv_no_contexts"] += 1
            continue  # no header asserts below; do not yield an empty group
        yield contexts


def iter_stage2_contexts(
    max_ctx: int = DEFAULT_MAX_CTX,
    limit: int | None = None,
    sources: str = DEFAULT_SOURCES,
    stats: dict | None = None,
    im_start_id: int | None = None,
):
    """The Stage-2 context stream (P4/P5): lazy, interleaved, capped.
    `limit` counts CONTEXTS (the trainer's --limit N = N draft samples).
    `stats` is updated in place (P6); pass None for a throwaway dict."""
    st = stats if stats is not None else {}
    for k, v in new_stats().items():
        st.setdefault(k, v)
    have = set(sources.replace(" ", "").split(","))
    if not have & {"xlam", "tb"}:
        raise ValueError(f"no valid source in {sources!r} (xlam and/or tb)")
    ga = (
        iter_xlam_groups(max_ctx, st)
        if "xlam" in have
        else iter(())
    )
    gb = (
        iter_tb_groups(
            max_ctx,
            im_start_id if im_start_id is not None else get_im_start_id(),
            st,
        )
        if "tb" in have
        else iter(())
    )
    n = 0
    for ctx in interleave(ga, gb):
        if limit is not None and n >= limit:
            return
        n += 1
        st["n_contexts"] = n
        st["n_" + ctx["source"]] += 1
        yield ctx


def build_stage2_contexts(
    limit: int | None = None,
    max_ctx: int = DEFAULT_MAX_CTX,
    sources: str = DEFAULT_SOURCES,
    stats: dict | None = None,
) -> list[dict]:
    """List wrapper over the stream (P5) — tests and the stats CLI."""
    return list(iter_stage2_contexts(max_ctx, limit, sources, stats))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _length_report(contexts: list[dict]) -> dict:
    def lens(src: str | None) -> list[int]:
        return [
            len(c["prompt_ids"])
            for c in contexts
            if src is None or c["source"] == src
        ]

    def dist(vals: list[int]) -> dict | None:
        if not vals:
            return None
        a = np.array(vals)
        return {
            "n": len(vals),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p95": float(np.percentile(a, 95)),
            "max": int(a.max()),
        }

    return {
        "all": dist(lens(None)),
        "xlam": dist(lens("xlam")),
        "tb": dist(lens("tb")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage-2 on-policy context builder (plan §2 Stage 2)"
    )
    ap.add_argument("cmd", choices=["stats", "preview"])
    ap.add_argument("--limit", type=int, default=200,
                    help="first N CONTEXTS in interleaved frozen order (P6)")
    ap.add_argument("--max-ctx", type=int, default=DEFAULT_MAX_CTX)
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--n", type=int, default=2, help="preview: how many contexts")
    ap.add_argument("--chars", type=int, default=260,
                    help="preview: tail characters to print per context")
    args = ap.parse_args()

    st: dict = {}
    ctxs = build_stage2_contexts(args.limit, args.max_ctx, args.sources, st)
    if args.cmd == "stats":
        st.pop("n_contexts", None)
        print(json.dumps({
            "n_contexts": len(ctxs),
            "lengths": _length_report(ctxs),
            **st,
        }, indent=2))
    else:
        tok = get_tokenizer()
        for i, c in enumerate(ctxs[: args.n]):
            text = tok.decode(c["prompt_ids"])
            tail = text[-args.chars:]
            assert text.endswith("<|im_start|>assistant\n"), "no header (P2)"
            print(json.dumps({
                "i": i,
                "query_id": c["query_id"],
                "source": c["source"],
                "turn": c["turn"],
                "n_ctx_tokens": len(c["prompt_ids"]),
                "tail": tail,
            }, indent=2))
        print(f"({len(ctxs)} contexts in the first --limit)")


if __name__ == "__main__":
    main()
