"""TB-prefix KD ablation: the Stage-1 recipe applied to ToolBench (plan §2
variant, added 2026-09-27 as the Stage-2 off-policy comparison).

Why this exists: the user's challenge — "why not SFT on ToolBench?" —
splits into a weak form (SFT toward gold ToolLLaMA text: wrong target,
acceptance is agreement with the frozen 14B, and gold-mismatch measured
70/21% even on xLAM) and a strong form (KD from the TARGET's OWN greedy
generations on TB contexts — a legitimate off-policy baseline for greedy
spec decoding, where committed prefixes are always target-greedy states).
This module builds the strong form so Stage-2 GKD has a real comparator:
same recipe as Stage-1, different context distribution. If off-policy KD
matches GKD's transfer gains, that is itself a finding (the on-policy
machinery is unnecessary for greedy decoding); if GKD wins, the transient-
state argument is validated. Either way the memo gets the comparison.

Pinned conventions (T-series; golden-tested in tests/test_build_tb_distill
.py; the generation/training sides reuse gen_stage1.py and kd_warmstart.py
UNCHANGED — only the prompt source and validation differ):

  T1 contexts = assistant-turn BOUNDARIES of the frozen TB prefix pool
     (frozen/tb_prefix_idx.json raw indices): for each conversation, a
     context is input_ids up to each assistant-turn start (the C6
     protocol's own turn-start helper, imported so the two sides cannot
     desync) — teacher-forced multi-turn sampling points, including the
     post-observation states that are the point of the TB mix. Same
     prefix-cut rule as build_stage2_contexts (apps-48's): full record
     kept, boundary enumerated by the trainer.
  T2 prompts are rendered fresh from raw via convert_conversation (the
     committed frozen raw indices — no clean_index join; same host-
     portability rule as build_stage2_contexts).
  T3 validation: the Stage-1 G4 rule CANNOT be transplanted verbatim —
     TB contexts are mixed prose/call turns (measured: only ~49% of turn-0
     emissions carry a wrapper at all, and thought-prose turns are the
     majority of tokens), so "no tool call in generation" is NOT a drop
     reason on TB. Instead: a TB generation is kept iff (a) it terminates
     (finish_reason stop, incl. EOS) or hits the cap cleanly, AND (b)
     every tool call it DOES contain is schema-valid against the
     conversation's own tool list (the G4 payload rules verbatim, incl.
     G8 dual-wrapper and comma-joined parallel calls). Prose-only
     generations are kept — they are the majority class of the target's
     real TB behavior. Hallucinated names / bad JSON still drop.
  T4 max-ctx: contexts left-truncate to --max-ctx (default 4096) with
     the same <|im_start|> boundary snap as onpolicy_gkd's _clip.
  T5 determinism: prompts in frozen tb_prefix_idx order, turns in
     conversation order; --limit N = first N contexts in that order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_prep.render import get_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = REPO_ROOT / "frozen"
TB_PREFIX_IDX = FROZEN / "tb_prefix_idx.json"

MAX_CTX_DEFAULT = 4096
IM_START = "<|im_start|>"


def tb_boundary_contexts(limit: int | None = None, max_ctx: int = MAX_CTX_DEFAULT) -> list[dict]:
    """T1/T2: every assistant-turn boundary of the frozen TB prefix pool
    as a prompt {"query_id": "<raw_idx>#t<turn>", "prompt_ids": [...]}.
    """
    from datasets import load_from_disk

    from src.analysis.eval_acceptance import _turn_starts_from_labels
    from src.data_prep.toolbench_clean import RAW_DIR, convert_conversation
    from src.data_prep.render import build_record

    raw_idx = json.loads(TB_PREFIX_IDX.read_text())
    tok = get_tokenizer()
    im_start_id = int(tok.convert_tokens_to_ids(IM_START))
    out: list[dict] = []
    for i in raw_idx:
        res = convert_conversation(load_from_disk(RAW_DIR)["train"][i])
        if res is None:
            raise AssertionError(
                f"raw index {i} no longer converts — raw data differs from "
                "the freeze pass (see build_stage2_contexts' same rule)"
            )
        msgs, tools, meta = res
        rec = build_record(msgs, tools, query_id=i,
                           final_answer_text=meta["final_answer"])
        starts = _turn_starts_from_labels(rec["labels"])
        for t, s in enumerate(starts):
            ids = rec["input_ids"][:s]
            if len(ids) > max_ctx:  # T4: left-truncate to a turn boundary
                ids = ids[-max_ctx:]
                if ids[0] != im_start_id:
                    for j, x in enumerate(ids):
                        if x == im_start_id:
                            ids = ids[j:]
                            break
            if not ids:
                continue
            out.append({
                "query_id": f"{i}#t{t}",
                "prompt_ids": ids,
                "n_tools": len(tools),
                "tools": tools,
            })
            if limit is not None and len(out) >= limit:
                return out  # exact-N prefix (T5): cut mid-conversation
    return out


def validate_tb_generation(text: str, tools: list[dict], tags) -> dict:
    """T3: keep prose, validate only the calls that exist. Reuses G4's
    payload validation (incl. G8 dual-wrapper + comma parallel calls)
    via validate_generation, but on TB a record with NO call is valid —
    the target's TB behavior is majority prose (measured 2026-09-27)."""
    from src.data_prep.build_distill_data import validate_generation

    v = validate_generation(text, tools, tags)
    problems = [p for p in v["problems"]
                if p != "no tool call in generation"]
    return {"valid": not problems, "n_calls": v["n_calls"],
            "problems": problems}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TB-prefix KD ablation: target-greedy datagen prompt "
                    "source + validation (generation via gen_stage1.py, "
                    "training via kd_warmstart.py — both unchanged)"
    )
    ap.add_argument("cmd", choices=["prompts", "stats"])
    ap.add_argument("--limit", type=int, default=None,
                    help="first N boundary contexts in frozen order (T5)")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX_DEFAULT)
    args = ap.parse_args()
    if args.cmd == "prompts":
        ctxs = tb_boundary_contexts(args.limit, args.max_ctx)
        print(json.dumps({
            "n_contexts": len(ctxs),
            "order": "frozen tb_prefix_idx.json order, turns in "
                     "conversation order (T5)",
            "sample_query_id": ctxs[0]["query_id"] if ctxs else None,
        }, indent=2))
    else:
        # stats: length distribution of the boundary contexts
        import numpy as np

        ctxs = tb_boundary_contexts(args.limit, args.max_ctx)
        lens = np.array([len(c["prompt_ids"]) for c in ctxs])
        print(json.dumps({
            "n": len(ctxs),
            "median": float(np.median(lens)),
            "p95": float(np.percentile(lens, 95)),
            "max": int(lens.max()),
        }, indent=2))


if __name__ == "__main__":
    main()
