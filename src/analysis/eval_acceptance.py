"""Acceptance-metrics evaluation from instrumented-loop event streams
(plan.md §4.1–4.3, §6.4).

The instrumented speculative-decoding loop (§6.5, src/serving/instrumented_
spec.py) emits one JSON object per *verification step*; this script turns
those events into every acceptance metric the plan requires. It is pure
numpy/pandas-free and torch-free, so it is built and tested before the loop
exists — the loop's only contract is the event schema below.

`load_records` (the shared frozen-record loader — the loop and the vLLM
bench both import it from here, so there is exactly one definition) accepts
both committed frozen/ parquets and HF save_to_disk dirs; the GPU host has
only the former (data/ is gitignored), so every records path must go
through it.

Event schema (one JSON object per verification step, JSONL):
  {
    "query_id": <id of the eval record>,
    "step": <0-based verification-step index within this prompt>,
    "draft_tokens": [t, ...],      # the k tokens the draft proposed
    "accept_mask": [b, ...],       # parallel bools: accepted at position n?
    "correction_token": t | null,  # target's token after the last accepted
                                    # one: present on reject AND on full
                                    # acceptance (the "bonus" token, §4.1)
    "eos": b                       # true if this step ended the generation
                                    # (correction_token == the stop token,
                                    # or draft proposed fewer than k tokens
                                    # because the draft emitted eos first)
  }

Counting conventions (pinned here; golden-tested in
tests/test_eval_acceptance.py — hand-computed cases):

  Verification stops at the first rejection (Leviathan et al.): a position
  n was *verified* in an event iff every position < n was accepted (or n
  is beyond the step's proposal length). Positions after a rejection got
  no verdict and must not count as "proposed but rejected".

  verified(n)  = #events that reached position n (all positions < n
                  accepted, and len(draft_tokens) > n)
  accepted(n)  = #events with accept_mask[n] True
  alpha        = sum over verified positions of accepted / verified
  alpha_n      = accepted(n) / verified(n)              # per-position §4.1
  tau          = mean over events of sum(accept_mask)   # accepted draft
                                                          tokens per step
  bonus_rate   = fraction of events with correction_token != null — the
                 target-emitted token after acceptance (rejection correction
                 or post-full-acceptance continuation), counted separately
                 from tau per plan §4.1 ("Report both conventions").
  emitted tokens per event = sum(accept_mask) + (1 if correction_token
                 else 0); the final eos correction token is counted as
                 emitted (it is part of the output) but prompts' totals are
                 reported both with and without it in the per-prompt records.

Region assignment (§4.3): the caller supplies, per query_id, the rendered
record's token-level region map (RLE from the data pipeline). Emitted
tokens (accepted + correction) are counted in the region of the output
position they occupy; rejected draft tokens are counted in the region of
the position they were *proposed* for (the plan's per-region α asks "how
often is the draft right in region r", which is a property of proposals).
Region codes as in src/data_prep/render.py: 0 context, 1 prose, 2
tool-call JSON, 3 tag, 4 final answer. The sub-cut of the tool-call region
into name vs. arguments is derived from the emitted/payload text when the
caller passes decode=True (the payload is {"name": ..., "arguments":...}
JSON, so the split is at the '"arguments"' key).

CLI:
  eval_acceptance.py events.jsonl --records frozen/xlam_eval.parquet [--k 5]
      -> prints the metrics summary; --out writes the full JSON report;
         --per-prompt writes one JSONL line per prompt.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data_prep.render import (
    REGION_CONTEXT,
    REGION_FINAL_ANSWER,
    REGION_PROSE,
    REGION_TAG,
    REGION_TOOL_CALL,
    unrle,
)

# sub-region codes for the name-vs-arguments sub-cut (§4.3 extra cut)
SUB_NAME = 10
SUB_ARGS = 11

_CI_BOOTSTRAP_N = 1000
_CI_SEED = 42


def load_records(path: str, limit: int | None = None) -> list[dict]:
    """Frozen-record loader shared by every consumer (the instrumented loop,
    the vLLM bench, this CLI): parquet (the committed frozen/ eval sets — the
    only form the GPU host has, since data/ is gitignored) or an HF
    save_to_disk dir; first N records in frozen order when `limit` is set."""
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    else:
        from datasets import load_from_disk

        rows = list(load_from_disk(path))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"no records in {path}")
    return rows


def load_events(path: str | Path) -> list[dict]:
    """Read an event JSONL; validate every event against the schema."""
    events = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            dt, am = e.get("draft_tokens"), e.get("accept_mask")
            if not isinstance(dt, list) or not isinstance(am, list):
                raise ValueError(f"event {ln}: draft_tokens/accept_mask must be lists")
            if len(dt) != len(am):
                raise ValueError(f"event {ln}: mask length != tokens length")
            if not all(isinstance(b, (bool, np.bool_)) for b in am):
                raise ValueError(f"event {ln}: accept_mask must be bools")
            if not dt:
                raise ValueError(f"event {ln}: empty draft_tokens")
            ct = e.get("correction_token", None)
            if ct is not None and not isinstance(ct, int):
                raise ValueError(f"event {ln}: correction_token must be int or null")
            events.append(e)
    if not events:
        raise ValueError("no events")
    return events


def events_by_prompt(events: list[dict]) -> dict:
    """Group events per prompt, preserving generation order.

    Sort key is (turn, step): per-turn streams restart step at 0 for each
    assistant turn, so step alone would scramble multi-turn records.
    """
    by: dict = defaultdict(list)
    for e in events:
        by[e["query_id"]].append(e)
    for qid in by:
        by[qid].sort(key=lambda e: (e.get("turn", 0), e["step"]))
    return by


def _event_stats(e: dict) -> dict:
    """Per-event counts: accepted (leading Trues), scored, bonus, emitted.

    verified/accepted = leading True entries in accept_mask (tau's
    numerator; a True after a False is a schema violation — the loop cannot
    accept past a rejection). scored = accepted + 1 if a rejection ended the
    step (the rejected position got a verdict too): alpha's denominator.
    bonus = 1 if the target emitted a correction token; emitted = accepted
    + bonus (the tokens that actually entered the output).
    """
    am = list(e["accept_mask"])
    verified = 0
    for b in am:
        if b:
            verified += 1
        else:
            break
    if sum(am) != verified:
        raise ValueError(
            f"event {e['query_id']}/{e['step']}: accepted token after a rejection"
        )
    # scored = positions the target actually judged: the accepted prefix
    # plus the rejected position (if any). alpha's denominator (a rejected
    # proposal IS a verdict); "verified"/"accepted" (leading Trues) is
    # tau's numerator.
    scored = verified + (1 if verified < len(am) else 0)
    return {
        "verified": verified,
        "accepted": verified,
        "scored": scored,
        "bonus": 1 if e.get("correction_token") is not None else 0,
        "emitted": verified + (1 if e.get("correction_token") is not None else 0),
    }


def flat_metrics(events: list[dict], k_max: int | None = None) -> dict:
    """Whole-run metrics: alpha, tau, bonus_rate, per-position alpha_n.

    k_max caps the per-position table (the loop's k, or the largest
    observed proposal length); events are unaffected.
    """
    per_event = [_event_stats(e) for e in events]
    n_events = len(events)

    # per-position scored/accepted: st["verified"] = leading Trues; a step
    # that rejected at position j scored positions 0..j (j rejected);
    # positions > j got no verdict. No rejection -> every position scored.
    pos_scored: list[int] = []
    pos_accepted: list[int] = []
    for e, st in zip(events, per_event):
        am = list(e["accept_mask"])
        for n in range(len(am)):
            while len(pos_scored) <= n:
                pos_scored.append(0)
                pos_accepted.append(0)
            if n <= st["verified"]:
                pos_scored[n] += 1
                if am[n]:
                    pos_accepted[n] += 1

    total_scored = sum(pos_scored)
    total_accepted = sum(pos_accepted)
    alpha = total_accepted / total_scored if total_scored else 0.0
    tau = float(np.mean([s["accepted"] for s in per_event])) if n_events else 0.0
    bonus_rate = float(np.mean([s["bonus"] for s in per_event])) if n_events else 0.0

    if k_max is not None:
        pos_scored = pos_scored[:k_max]
        pos_accepted = pos_accepted[:k_max]

    alpha_n = [a / v if v else None for a, v in zip(pos_accepted, pos_scored)]
    return {
        "n_events": n_events,
        "n_prompts": len({e["query_id"] for e in events}),
        "alpha": alpha,
        "tau": tau,
        "bonus_rate": bonus_rate,
        "alpha_n": alpha_n,
        "pos_scored": pos_scored,
        "pos_accepted": pos_accepted,
        "tokens_emitted": int(sum(s["emitted"] for s in per_event)),
    }


def assign_token_regions(
    events: list[dict],
    region_map: list[int],
    n_prompt_tokens: int,
    labels: list[int] | None = None,
) -> list[list[int]]:
    """Per event: region code per DRAFT PROPOSAL position + per emitted token.

    Returns a list (per event, in event order) of two parallel lists:
      proposal_regions[n]  — region of output position (cursor + n); used for
                             per-region alpha (a property of proposals)
      emitted_regions      — regions of the tokens that actually entered the
                             output this step: accepted draft tokens in
                             order, then the correction token if present.
                             Used for per-region tau/emitted shares.
    cursor = n_prompt_tokens + emitted-so-far (accepted + bonus tokens from
    all previous steps of this prompt). Positions beyond the record's region
    map (draft overran the ground-truth length) are REGION_PROSE — they are
    still valid proposal positions, but the ground truth says the model
    should have stopped there.

    Multi-turn records (ToolBench): the loop generates each assistant turn
    separately (its context ends where that turn begins). Events carry an
    optional "turn" field (0-based assistant-turn index); when present, the
    cursor resets to that turn's start for each new turn. Turn starts are
    derived from the record's labels: the first loss token of each
    assistant span (loss is contiguous within a turn — golden-tested).
    """
    cursor = n_prompt_tokens
    turn_starts: list[int] | None = None
    cur_turn: int | None = None
    out = []
    for e in events:
        if e.get("turn") is not None:
            if turn_starts is None:
                turn_starts = _turn_starts_from_labels(labels)
            t = e["turn"]
            if t != cur_turn:  # reset ONLY on a turn transition
                cur_turn = t
                if t < len(turn_starts):
                    cursor = turn_starts[t]
                else:
                    # turn index beyond the record's turns: the loop
                    # generated past the final <|im_end|>; anchor at the
                    # record end so all such proposals read as prose
                    cursor = len(region_map)
        proposal = []
        for n in range(len(e["draft_tokens"])):
            pos = cursor + n
            proposal.append(
                region_map[pos] if pos < len(region_map) else REGION_PROSE
            )
        emitted = [proposal[n] for n in range(_event_stats(e)["accepted"])]
        if e.get("correction_token") is not None:
            pos = cursor + len(emitted)
            emitted.append(
                region_map[pos] if pos < len(region_map) else REGION_PROSE
            )
        out.append((proposal, emitted))
        cursor += _event_stats(e)["emitted"]
    return out


def _turn_starts_from_labels(labels: list[int]) -> list[int]:
    """First token position of each assistant span (loss is contiguous
    within a turn; a span boundary is labels[i] != -100 after labels[i-1]
    == -100, plus position 0 if the record starts with a loss token)."""
    starts = [i for i in range(1, len(labels)) if labels[i] != -100 and labels[i - 1] == -100]
    if labels and labels[0] != -100:
        starts.insert(0, 0)
    return starts


def region_metrics(events: list[dict], per_event_regions) -> dict:
    """Per-region alpha (proposal-anchored) and tau/emitted shares."""
    acc: dict = defaultdict(lambda: [0, 0, 0, 0])  # region -> [verified, accepted, emitted, proposals]
    for e, (proposal, emitted) in zip(events, per_event_regions):
        st = _event_stats(e)
        for n in range(len(proposal)):
            r = proposal[n]
            if n <= st["verified"]:
                acc[r][0] += 1
                if e["accept_mask"][n]:
                    acc[r][1] += 1
            acc[r][3] += 1
        for r in emitted:
            acc[r][2] += 1
    out = {}
    for r in sorted(acc):
        verified, accepted, emitted, proposed = acc[r]
        out[str(r)] = {
            "alpha": accepted / verified if verified else None,
            "n_verified": verified,
            "n_accepted": accepted,
            "n_proposed": proposed,
            "n_emitted": emitted,
        }
    return out


def name_vs_arguments_cut(
    region_map: list[int],
    input_ids: list[int],
    decode_fn,
) -> list[int]:
    """Sub-region map (§4.3 extra cut) for the tool-call payload tokens.

    Within each contiguous region-2 span, decode the span text and split it
    at the '"arguments"' key: tokens whose decoded prefix ends at/before the
    cut are SUB_NAME (function-name side), the rest SUB_ARGS. The split uses
    prefix decoding (decode_fn(input_ids[i:t+1]) length = the token's end
    offset in the span text) — exact for BPE since decode concatenates
    pieces. Payloads without an '"arguments"' key (malformed) are all
    SUB_NAME. Non-payload regions pass through unchanged.
    """
    sub = list(region_map)
    i = 0
    while i < len(region_map):
        if region_map[i] != REGION_TOOL_CALL:
            i += 1
            continue
        j = i
        while j < len(region_map) and region_map[j] == REGION_TOOL_CALL:
            j += 1
        text = decode_fn(input_ids[i:j])
        cut = text.find('"arguments"')
        if cut < 0:
            for t in range(i, j):
                sub[t] = SUB_NAME
        else:
            for t in range(i, j):
                end = len(decode_fn(input_ids[i : t + 1]))
                sub[t] = SUB_NAME if end <= cut else SUB_ARGS
        i = j
    return sub


def _prompt_alpha(events: list[dict]) -> float | None:
    """alpha over one prompt's events; None if no scored positions.

    Denominator = scored positions (accepted prefix + rejection verdicts),
    matching flat_metrics' alpha convention: a rejected proposal is a
    verdict, not a vanishing one.
    """
    v = a = 0
    for e in events:
        st = _event_stats(e)
        v += st["scored"]
        a += st["accepted"]
    return a / v if v else None


def bootstrap_ci(
    events: list[dict],
    stat: str = "alpha",
    n_boot: int = _CI_BOOTSTRAP_N,
    seed: int = _CI_SEED,
) -> dict:
    """Bootstrap 95% CI over prompts (plan §4: CIs over prompts, not steps)."""
    by = events_by_prompt(events)
    qids = sorted(by)
    rng = np.random.default_rng(seed)
    stats = []
    for qid in qids:
        pe = by[qid]
        if stat == "alpha":
            s = _prompt_alpha(pe)
        elif stat == "tau":
            s = (
                float(np.mean([_event_stats(e)["accepted"] for e in pe]))
                if pe
                else None
            )
        else:
            raise ValueError(f"unknown stat {stat}")
        stats.append(s)
    stats = np.array([s if s is not None else 0.0 for s in stats])
    n = len(stats)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = stats[rng.integers(0, n, n)]
        boots[b] = sample.mean()
    return {
        "stat": stat,
        "mean": float(stats.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_prompts": n,
    }


def per_prompt_records(events: list[dict]) -> list[dict]:
    """One summary record per prompt (for the --per-prompt JSONL)."""
    by = events_by_prompt(events)
    out = []
    for qid in sorted(by):
        pe = by[qid]
        st = [_event_stats(e) for e in pe]
        out.append(
            {
                "query_id": qid,
                "n_steps": len(pe),
                "n_scored": sum(s["scored"] for s in st),
                "n_accepted": sum(s["accepted"] for s in st),
                "alpha": _prompt_alpha(pe),
                "tau": float(np.mean([s["accepted"] for s in st])),
                "n_emitted": sum(s["emitted"] for s in st),
            }
        )
    return out


def full_report(
    events: list[dict],
    records=None,
    k_max: int | None = None,
    decode_fn=None,
) -> dict:
    """Assemble the complete §4.1–4.3 report from events (+ optional region
    records: an iterable of dataset rows with input_ids/regions/
    n_prompt_tokens, keyed by query_id). Region splits require records;
    the name-vs-arguments sub-cut additionally needs a decode_fn (the CLI
    supplies the target tokenizer's decode)."""
    report: dict = {"flat": flat_metrics(events, k_max)}
    report["ci"] = {
        "alpha": bootstrap_ci(events, "alpha"),
        "tau": bootstrap_ci(events, "tau"),
    }
    report["per_prompt"] = per_prompt_records(events)
    if records is None:
        return report

    by_id = {r["query_id"]: r for r in records}
    by = events_by_prompt(events)
    acc: dict = defaultdict(lambda: [0, 0])  # region -> [verified, accepted]
    emitted_by_region: dict = defaultdict(int)
    sub_acc: dict = defaultdict(lambda: [0, 0])  # SUB_NAME/SUB_ARGS -> [v, a]
    for qid, pe in by.items():
        r = by_id[qid]
        rmap = unrle([list(x) for x in r["regions"]])
        sub_map = (
            name_vs_arguments_cut(rmap, r["input_ids"], decode_fn)
            if decode_fn is not None
            else None
        )
        per_ev = assign_token_regions(
            pe, rmap, r["n_prompt_tokens"], r["labels"]
        )
        # sub-cut: same proposal positions, remapped through the sub map
        sub_per_ev = None
        if sub_map is not None:
            sub_per_ev = assign_token_regions(
                pe, sub_map, r["n_prompt_tokens"], r["labels"]
            )
        for idx, (e, (prop, emit)) in enumerate(zip(pe, per_ev)):
            st = _event_stats(e)
            for n in range(len(prop)):
                if n <= st["verified"]:
                    acc[prop[n]][0] += 1
                    if e["accept_mask"][n]:
                        acc[prop[n]][1] += 1
            for reg in emit:
                emitted_by_region[reg] += 1
            if sub_per_ev is not None:
                sprop, _ = sub_per_ev[idx]
                for n in range(len(sprop)):
                    if n <= st["verified"]:
                        sub_acc[sprop[n]][0] += 1
                        if e["accept_mask"][n]:
                            sub_acc[sprop[n]][1] += 1
    report["region_alpha"] = {
        str(k): (v[1] / v[0] if v[0] else None) for k, v in sorted(acc.items())
    }
    report["region_n_verified"] = {str(k): v[0] for k, v in sorted(acc.items())}
    report["region_n_emitted"] = {
        str(k): v for k, v in sorted(emitted_by_region.items())
    }
    if decode_fn is not None:
        report["subcut_alpha"] = {
            str(k): (v[1] / v[0] if v[0] else None) for k, v in sorted(sub_acc.items())
        }
        report["subcut_n_verified"] = {str(k): v[0] for k, v in sorted(sub_acc.items())}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("events", help="event JSONL from the instrumented loop")
    ap.add_argument("--records", default=None, help="frozen parquet or rendered "
        "dataset dir with input_ids/regions/n_prompt_tokens, for region splits")
    ap.add_argument("--k", type=int, default=None, help="draft length cap")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    ap.add_argument("--per-prompt", dest="per_prompt_out", default=None,
        help="write per-prompt JSONL here")
    ap.add_argument("--subcut", action="store_true",
        help="also compute the name-vs-arguments sub-cut (needs the target "
        "tokenizer — downloads config on first use)")
    args = ap.parse_args()

    events = load_events(args.events)
    records = None
    if args.records:
        records = load_records(args.records)
    decode_fn = None
    if args.subcut:
        from src.data_prep.render import get_tokenizer

        decode_fn = get_tokenizer().decode
    rep = full_report(events, records, k_max=args.k, decode_fn=decode_fn)
    summary = {
        k: rep[k]
        for k in (
            "flat", "ci", "region_alpha", "region_n_verified",
            "subcut_alpha", "subcut_n_verified",
        )
        if k in rep
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
    if args.per_prompt_out:
        p = Path(args.per_prompt_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for r in rep["per_prompt"]:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
