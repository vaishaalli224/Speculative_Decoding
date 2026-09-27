#!/usr/bin/env python
"""§8.1 draft-choice rule, mechanically (plan §5 hour 3–3.5 decision point).

The rule is pre-registered (plan §8.1) so the GPU day makes no judgment
call: distill the draft with the higher UNTUNED wall-clock speedup vs. AR
at k=5, greedy, batch 1, on xLAM-eval; tie-break toward 0.5B (cheaper to
train/verify; more headroom). This script reads results/metrics.jsonl —
the append-only record every bench run appends to (B7) — finds the three
runs the rule needs, and prints the decision + the numbers behind it.

If either run is missing, it says exactly what to run (never guesses, never
falls back to a different metric).

Usage (GPU host, after run_baselines.sh):
  python scripts/pick_draft.py [--metrics results/metrics.jsonl] [--ar-tag ar]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# metric names from the run descriptors in bench_vllm.py's bench_and_record
AR_TAG = "ar"
DRAFT_TAGS = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": "0.5B",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "1.5B",
}


def load_metrics(path: str) -> list[dict]:
    lines = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    benches = [r for r in lines if r.get("kind") == "bench"]
    if not benches:
        raise SystemExit(f"no bench records in {path} — run scripts/run_baselines.sh first")
    # flatten k out of spec_config so find_run can match it like any field
    # (draft records carry it only inside spec_config; AR records have none)
    for r in benches:
        cfg = r.get("spec_config") or {}
        r["_k"] = cfg.get("num_speculative_tokens")
    return benches


def find_run(benches: list[dict], **match) -> dict:
    """The most recent bench record matching every (key, value) pair; None
    if absent. The baselines script appends in run order, so 'most recent'
    is deterministic even if a config was re-run."""
    for rec in reversed(benches):
        if all(rec.get(k) == v for k, v in match.items()):
            return rec
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.jsonl")
    args = ap.parse_args()

    benches = load_metrics(args.metrics)

    # the rule's three runs: AR + both drafts, all at k=5 greedy batch 1 on
    # xLAM-eval (run_baselines.sh's exact configurations). Exactness-mode
    # records carry tag="*_exactness" and runs=1 — exclude both so the rule
    # reads the 3-run medians, never a 50-prompt exactness side.
    ar = find_run(benches, spec_method="ar", batch=1, temperature=0.0,
                  per_turn=False, tag="ar", runs=3)
    if ar is None:
        raise SystemExit(
            "AR batch-1 greedy 3-run baseline not found in metrics.jsonl — "
            "run scripts/run_baselines.sh first"
        )
    ar_tok_s = ar["median_gen_tok_s"]

    per_draft: dict[str, dict] = {}
    for model, label in DRAFT_TAGS.items():
        rec = find_run(
            benches, spec_method="draft_model", method_detail=model,
            batch=1, temperature=0.0, per_turn=False,
            tag="draft_model", runs=3, _k=5,
        )
        if rec is None:
            raise SystemExit(
                f"untuned {label} k=5 greedy batch-1 run not found — "
                "run scripts/run_baselines.sh first"
            )
        per_draft[label] = {
            "model": model,
            "gen_tok_s": rec["median_gen_tok_s"],
            "speedup": rec["median_gen_tok_s"] / ar_tok_s,
        }

    # §8.1: higher untuned speedup wins; tie-break toward 0.5B
    a, b = per_draft["0.5B"], per_draft["1.5B"]
    if a["speedup"] > b["speedup"]:
        winner, reason = "0.5B", "higher untuned speedup"
    elif b["speedup"] > a["speedup"]:
        winner, reason = "1.5B", "higher untuned speedup"
    else:
        winner, reason = "0.5B", "tie — §8.1 tie-break toward 0.5B"

    out = {
        "rule": "plan §8.1: higher untuned speedup vs AR, k=5 greedy batch 1; "
                "tie-break toward 0.5B",
        "ar_gen_tok_s": ar_tok_s,
        "candidates": per_draft,
        "winner": winner,
        "winner_model": per_draft[winner]["model"],
        "reason": reason,
    }
    print(json.dumps(out, indent=2))
    out_path = Path(args.metrics).resolve().parent / "draft_decision.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\ndecision: distill {winner} ({per_draft[winner]['model']}) — "
          f"saved to {out_path}")


if __name__ == "__main__":
    main()
