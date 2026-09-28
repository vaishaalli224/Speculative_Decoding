"""Build the Stage-2 context pools on the GPU host (plan.md §2 Stage 2, §6.9).

Two pools, both derived ONLY from committed frozen indices + raw data (the
host never re-derives splits and never needs gitignored data/processed
artifacts — freeze_splits.verify's guarantee, extended to Stage 2):
  xLAM: training pool minus stage1 (52,794 contexts) — frozen/
        xlam_train_pool_idx.json minus frozen/stage1_idx.json (both
        committed), rendered like the train/eval records (build_messages +
        target template + regions)
  TB:   the 45,023 prefix conversations (tools disjoint from TB-500 by
        construction) — frozen/tb_prefix_idx.json holds RAW ToolBench
        indices (freeze_splits already dereferenced the clean-index
        positions when freezing), so no clean_index.json join and no
        clean+carve re-run is needed on the host: convert_conversation
        runs on raw directly.

Stage 2 consumes CONTEXTS. A context is a PROMPT PREFIX: input_ids up to
a sampling boundary, ending with the template's assistant header (the
header is the last thing before the boundary — find_assistant_turns
starts the loss span just after it, so input_ids[:n_prompt_tokens] ends
with `<|im_start|>assistant\n`, exactly the shape inference serves).
Records keep the FULL conversation in input_ids with n_prompt_tokens
marking the boundary (the trainer cuts; the gold assistant turns stay in
the pool for provenance). One record per xLAM example (single-turn: one
boundary). One record per TB conversation (multi-turn: the trainer
enumerates EVERY assistant-turn boundary as a context — post-observation
states are the point of the TB mix, plan §3.2/§9; a turn-0-only cut
would reduce TB to "xLAM with different prompts").

Pure CPU; runs on the host while the GPU does tau probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = REPO_ROOT / "frozen"


def xlam_stage2_idx(limit: int | None = None) -> list[int]:
    """Pool minus Stage-1 carve, from the COMMITTED frozen indices (host-
    portable: data/processed/xlam/train_idx.json is gitignored and absent
    on the GPU host). Sorted — a pure function of the two frozen files."""
    pool = json.loads((FROZEN / "xlam_train_pool_idx.json").read_text())
    stage1 = set(json.loads((FROZEN / "stage1_idx.json").read_text()))
    if not stage1.issubset(set(pool)):
        raise AssertionError(
            f"{len(stage1 - set(pool))} Stage-1 indices are outside the "
            "training pool — frozen/xlam_train_pool_idx.json and "
            "frozen/stage1_idx.json desynced; re-run freeze_splits"
        )
    idx = sorted(set(pool) - stage1)
    if limit:
        idx = idx[:limit]
    return idx


def build_xlam_stage2(limit: int | None = None) -> dict:
    """Render the xLAM Stage-2 context pool (pool minus stage1)."""
    from datasets import Dataset, load_from_disk

    from src.data_prep.xlam_prep import OUT_DIR, RAW_DIR, process_example

    idx = xlam_stage2_idx(limit)
    ds = load_from_disk(RAW_DIR)["train"]
    records = [process_example(ds[i]) for i in idx]
    out = OUT_DIR / "stage2_contexts"
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(out)
    return {
        "n": len(records),
        "source": "frozen/xlam_train_pool_idx.json minus frozen/stage1_idx.json",
        "out": str(out),
    }


def build_tb_prefixes(limit: int | None = None) -> dict:
    """Render the TB prefix pool straight from the committed frozen raw
    indices — no clean_index.json join, no clean+carve re-run (the frozen
    file already encodes which conversations passed)."""
    from datasets import Dataset, load_from_disk

    from src.data_prep.render import build_record
    from src.data_prep.toolbench_clean import RAW_DIR, convert_conversation

    raw_idx = json.loads((FROZEN / "tb_prefix_idx.json").read_text())
    if limit:
        raw_idx = raw_idx[:limit]
    ds = load_from_disk(RAW_DIR)["train"]
    records = []
    for i in raw_idx:
        res = convert_conversation(ds[i])
        if res is None:
            raise AssertionError(
                f"raw index {i} (frozen/tb_prefix_idx.json) no longer "
                "converts — raw data differs from the pass that froze the "
                "prefix pool"
            )
        msgs, tools, meta = res
        rec = build_record(
            msgs, tools, query_id=i, final_answer_text=meta["final_answer"]
        )
        records.append(rec)
    out_dir = REPO_ROOT / "data" / "processed" / "toolbench" / "prefixes"
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(out_dir)
    return {
        "n": len(records),
        "source": "frozen/tb_prefix_idx.json (raw indices)",
        "out": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", choices=["xlam", "tb"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.pool == "xlam":
        print(json.dumps(build_xlam_stage2(args.limit), indent=2))
    else:
        print(json.dumps(build_tb_prefixes(args.limit), indent=2))


if __name__ == "__main__":
    main()
