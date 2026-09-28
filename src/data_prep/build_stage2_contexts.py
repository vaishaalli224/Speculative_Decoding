"""Build the Stage-2 context pools on the GPU host (plan §2 Stage 2, §6.9).

Two pools, both from committed frozen indices (the host never re-derives
splits — freeze_splits.verify's guarantee):
  xLAM: training pool minus stage1 (52,794 contexts) — rendered like the
        train/eval records (build_messages + target template + regions)
  TB:   the 45,023 prefix conversations (tools disjoint from TB-500 by
        construction) — toolbench_clean build --split prefixes

Stage 2 consumes CONTEXTS: for xLAM that is the same full records the KD
pipeline builds (input_ids include the gold assistant turn — the trainer
samples from the PROMPT prefix only; a context record's n_prompt_tokens
marks the sampling boundary). For TB the prefixes split already carries
multi-turn records; the trainer likewise samples per assistant turn
boundary... Stage-2 v1 (today) samples from turn-0 contexts only
(n_prompt_tokens), which keeps the sampling loop simple and still covers
post-observation states via the TB mix's multi-turn record prefixes
that end at each assistant turn.

Pure CPU; runs on the host while the GPU does tau probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = REPO_ROOT / "frozen"


def build_xlam_stage2(limit: int | None = None) -> dict:
    """Render the xLAM Stage-2 context pool (pool minus stage1)."""
    from datasets import Dataset, load_from_disk

    from src.data_prep.render import get_tokenizer
    from src.data_prep.xlam_prep import (
        OUT_DIR, RAW_DIR, _load_split_idx, process_example,
    )

    pool = set(_load_split_idx()["train"])
    stage1 = set(json.loads((FROZEN / "stage1_idx.json").read_text()))
    idx = sorted(pool - stage1)
    if limit:
        idx = idx[:limit]
    ds = load_from_disk(RAW_DIR)["train"]
    records = [process_example(ds[i]) for i in idx]
    out = OUT_DIR / "stage2_contexts"
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(out)
    return {
        "n": len(records),
        "source": "xlam pool minus stage1 (frozen indices)",
        "out": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", choices=["xlam", "tb"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.pool == "xlam":
        print(json.dumps(build_xlam_stage2(args.limit), indent=2))
    else:
        from src.data_prep.toolbench_clean import build_split

        print(json.dumps(build_split("prefixes", args.limit), indent=2))


if __name__ == "__main__":
    main()
