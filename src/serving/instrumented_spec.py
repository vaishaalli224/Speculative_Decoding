"""Instrumented speculative-decoding loop (plan.md §6.5).

A *proposer* (src/serving/proposers.py — an HF draft or an n-gram matcher)
proposes up to k tokens greedily; the frozen target verifies greedily
(Leviathan et al. rejection sampling at temperature 0); one JSON event
per verification step is emitted in exactly the schema
`src/analysis/eval_acceptance.py` consumes. This loop is the measurement
instrument for α, τ, per-position α_n and region splits (§4.1–4.3) on a
small prompt subset; wall-clock numbers always come from vLLM (§6.6).

The core is a pure-Python state machine over token-id lists — torch-free
— so every convention below is golden-tested without torch; the HF
adapters at the bottom of this file are thin, separately-tested wrappers.

Models are placed via --device (cuda:0 on the GPU host; the default "auto"
keeps CPU/MPS behavior for the local §4.4 checks unchanged).

Pinned conventions (event semantics; golden-tested in
tests/test_instrumented_spec.py):

  C1 rejection at position j: accept_mask = [T...T, F] (never a True
     after a False — the analyzer raises on that), correction_token =
     the target's greedy prediction at proposal position j. eos iff
     the correction == EOS.
  C2 full acceptance of a proposal that ends with the draft's EOS (may
     be shorter than k): mask all-True, correction_token = None — no
     bonus past EOS. eos = true.
  C3 full acceptance of an EOS-free proposal: mask all-True,
     correction_token = the bonus (the target's prediction after the
     last proposal token) if budget headroom remains, else None (C4).
     eos iff the emitted bonus == EOS.
  C4 token budget (max_new_tokens per assistant turn): the proposal is
     capped at the remaining budget BEFORE proposing; a fully-accepted
     proposal takes its bonus only if >= 1 token of headroom remains.
     The loop therefore emits at most max_new_tokens tokens — outputs
     are directly comparable to HF generate(max_new_tokens=N), which
     the §4.4 exactness test requires. A budget stop sets eos=true and
     step_type="budget".
  C5 stop token: EOS is the tokenizer's eos_token (<|im_end|>, 151645,
     read from the tokenizer, never hard-coded). Other special ids are
     ordinary emitted tokens.
  C6 multi-turn records (TB-500): each assistant turn is generated
     separately, teacher-forced on the ground-truth context — context
     for turn t is input_ids[:turn_starts[t]], turn starts derived
     from the record's labels by the analyzer's own helper (imported,
     so the two sides can never desync). Events carry "turn" (0-based
     assistant-turn index) only when the record has >1 assistant turn;
     "step" restarts at 0 each turn. Proposal regions thus never fall
     in the context region. Between turns the adapters are *synced*
     forward onto the ground-truth span (crop the generated tokens,
     forward the truth) — no per-turn re-prefill.
  C7 proposer contract: proposals are 1..cap tokens where
     cap = min(k, budget_left). Empty proposals are banned — an n-gram
     proposer with no match emits a sacrificial PAD token instead
     (Qwen pad = <|endoftext|>), keeping step semantics uniform across
     proposer types. Exactness is unaffected: acceptance only ever
     emits the target's own argmax.

State synchronization (the classic off-by-one bug source): each adapter
keeps its own view of the sequence, which may run past the canonical
(committed) prefix with speculative tokens. After every round, and at
every teacher-forced turn transition, the loop calls
`adapter.sync(pre_len, emitted)` with the canonical (prefix_len,
emitted) pair; the generic sync keeps whatever speculative prefix
matches `emitted` and forwards only the remainder (at most one token
after a rejection — the correction — plus the inter-turn truth span for
turn transitions). `next_pred` (the argmax after the view's last token)
is tracked with the position it was computed at, so any crop that
invalidates it is detected and repaired with a single-token re-forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.analysis.eval_acceptance import _turn_starts_from_labels, load_records

STEP_REJECT = "reject"
STEP_FULL_BONUS = "full_bonus"
STEP_FULL_EOS = "full_eos"
STEP_BUDGET = "budget"


class Proposer(Protocol):
    """The draft side of the loop (see src/serving/proposers.py)."""

    def propose(self, k: int) -> list[int]: ...

    def reset(self, tokens: list[int]) -> None: ...

    def sync(self, pre_len: int, emitted: list[int]) -> None: ...


class Verifier(Protocol):
    """The target side: greedy predictions over the current sequence."""

    def predict(self, tokens: list[int]) -> list[int]:
        """Append `tokens` (one forward), return len(tokens)+1 argmaxes:
        preds[j] = the model's greedy choice at the state
        canonical + tokens[0..j-1] (the token that would follow position
        j-1 of the proposal), plus one extra entry — the choice after
        the last proposal token (the bonus)."""
        ...  # pragma: no cover

    def reset(self, tokens: list[int]) -> None: ...

    def sync(self, pre_len: int, emitted: list[int]) -> None: ...


@dataclass
class LoopResult:
    """Everything one assistant-turn generation produces."""

    events: list[dict]
    output_ids: list[int]  # emitted tokens (accepted + corrections/bonus)

    @property
    def n_tokens(self) -> int:
        return len(self.output_ids)


def verify_round(
    proposal: list[int],
    preds: list[int],
    budget_left: int,
    eos_id: int,
) -> tuple[list[bool], int | None, str, bool]:
    """Score one round from the verifier's predictions; pure function.

    `preds` has len(proposal)+1 entries (the last is the bonus, § C3).
    Returns (accept_mask, correction_token, step_type, eos).
    """
    n = len(proposal)
    if len(preds) != n + 1:
        raise AssertionError("verifier predictions length != proposal+1")

    for j in range(n):
        if proposal[j] != preds[j]:  # first rejection at position j (C1)
            return [True] * j + [False], preds[j], STEP_REJECT, preds[j] == eos_id

    mask = [True] * n  # full acceptance
    if proposal[-1] == eos_id:  # C2: the draft's EOS was accepted
        return mask, None, STEP_FULL_EOS, True
    if budget_left > n:  # C4: bonus only with >= 1 token of headroom
        bonus = preds[n]
        if bonus == eos_id:  # C3: the bonus ends the turn
            return mask, bonus, STEP_FULL_EOS, True
        return mask, bonus, STEP_FULL_BONUS, False
    return mask, None, STEP_BUDGET, True  # C4: budget exhausted, no bonus


def run_rounds(
    proposer: Proposer,
    verifier: Verifier,
    prompt_len: int,
    eos_id: int,
    k: int,
    max_new_tokens: int,
    query_id,
    turn: int | None = None,
    max_steps: int | None = None,
) -> LoopResult:
    """Round loop for one assistant turn; adapters must already be
    aligned to the turn's context. Every round emits >= 1 token, so the
    budget alone terminates the loop; `max_steps` is a safety cap."""
    if k < 1:
        raise ValueError("k must be >= 1")
    events: list[dict] = []
    output: list[int] = []
    step = 0
    while len(output) < max_new_tokens:
        if max_steps is not None and step >= max_steps:
            break
        budget_left = max_new_tokens - len(output)
        cap = min(k, budget_left)  # C4 caps the proposal first
        proposal = proposer.propose(cap)
        if not 1 <= len(proposal) <= cap:
            raise AssertionError(
                f"proposer returned {len(proposal)} tokens, contract is 1..{cap}"
            )
        preds = verifier.predict(proposal)
        mask, correction, stype, eos = verify_round(
            proposal, preds, budget_left, eos_id
        )
        n_acc = mask.count(True)
        emitted = [proposal[i] for i in range(n_acc)]
        if correction is not None:
            emitted.append(correction)
        if not emitted:
            raise AssertionError("round emitted nothing — proposer/verifier bug")
        # The event carries only the SCORED prefix of the proposal
        # (positions 0..n_acc, plus the rejection verdict at n_acc):
        # draft_tokens and accept_mask must be parallel per the analyzer's
        # schema, and post-rejection positions got no verdict — the
        # analyzer's convention is that they do not appear at all.
        scored_len = n_acc + (1 if correction is not None and stype == STEP_REJECT else 0)
        event = {
            "query_id": query_id,
            "step": step,
            "draft_tokens": list(proposal[:scored_len]),
            "accept_mask": mask,
            "correction_token": correction,
            "eos": bool(eos),
            "step_type": stype,
        }
        if turn is not None:
            event["turn"] = turn
        events.append(event)

        pre_len = prompt_len + len(output)  # canonical length before this round
        proposer.sync(pre_len, emitted)
        verifier.sync(pre_len, emitted)
        output.extend(emitted)
        if eos:
            break
        step += 1
    return LoopResult(events=events, output_ids=output)


def generate_turn(
    proposer: Proposer,
    verifier: Verifier,
    prompt: list[int],
    eos_id: int,
    k: int,
    max_new_tokens: int,
    query_id,
    turn: int | None = None,
    max_steps: int | None = None,
) -> LoopResult:
    """Reset both sides to `prompt`, then run_rounds (single-turn path)."""
    proposer.reset(list(prompt))
    verifier.reset(list(prompt))
    return run_rounds(
        proposer, verifier, len(prompt), eos_id, k, max_new_tokens,
        query_id, turn, max_steps,
    )


def generate_record(
    record: dict,
    proposer: Proposer,
    verifier: Verifier,
    eos_id: int,
    k: int,
    max_new_tokens: int,
    max_steps: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate every assistant turn of a rendered record (C6).

    Returns (events, outputs): events in stream order; outputs one entry
    per turn with {"turn", "output_ids", "n_tokens"} (turn = None for
    single-turn records). The proposer must be fresh for this record; the
    verifier is reused across records (reset for turn 0). Turn
    transitions sync both adapters onto the ground-truth span instead of
    re-prefilling (C6).
    """
    ids = record["input_ids"]
    starts = _turn_starts_from_labels(record["labels"])
    if not starts:
        return [], []
    events: list[dict] = []
    outputs: list[dict] = []
    multi = len(starts) > 1
    prev_start: int | None = None
    for t, s in enumerate(starts):
        if prev_start is None:
            proposer.reset(ids[:s])
            verifier.reset(ids[:s])
        else:  # teacher-forced turn transition (C6)
            proposer.sync(prev_start, ids[prev_start:s])
            verifier.sync(prev_start, ids[prev_start:s])
        res = run_rounds(
            proposer, verifier, s, eos_id, k, max_new_tokens,
            record["query_id"], t if multi else None, max_steps,
        )
        events.extend(res.events)
        outputs.append({
            "turn": t if multi else None,
            "output_ids": res.output_ids,
            "n_tokens": res.n_tokens,
        })
        prev_start = s
    return events, outputs


def get_eos_id(tokenizer=None) -> int:
    """EOS per C5: the tokenizer's eos_token (<|im_end|> for Qwen2.5)."""
    if tokenizer is None:
        from src.data_prep.render import get_tokenizer

        tokenizer = get_tokenizer()
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no eos_token_id")
    return int(tokenizer.eos_token_id)


# ---------------------------------------------------------------------------
# HF adapters — the only torch-importing code, built lazily by the CLI
# ---------------------------------------------------------------------------


class _HFAdapter:
    """KV-cache adapter shared by the verifier and the draft.

    State: `seq` = the adapter's full view (canonical prefix + possibly a
    speculative tail); `_cache` = DynamicCache with exactly len(seq)
    positions; `_np` = the argmax computed at sequence position `_np_pos`.
    `_np` is valid for the current view iff _np_pos == len(seq) - 1.
    """

    def __init__(self, model):
        import torch  # noqa: F401 — fail loudly if torch missing

        self.model = model
        self.model.eval()
        self.seq: list[int] = []
        self._cache = None
        self._np: int | None = None
        self._np_pos: int = -1

    # -- primitives ---------------------------------------------------------

    def _forward(self, fwd: list[int]):
        """Forward `fwd`, appending to the cache; return logits rows."""
        import torch

        with torch.no_grad():
            out = self.model(
                input_ids=torch.tensor([fwd], dtype=torch.long).to(self.model.device),
                past_key_values=self._cache,
                use_cache=True,
            )
        self.seq.extend(fwd)
        if fwd:
            self._np = int(out.logits[0, -1].argmax())
            self._np_pos = len(self.seq) - 1
        return out.logits[0]

    def _crop(self, n_kept: int) -> None:
        if n_kept > len(self.seq):
            raise AssertionError("crop beyond current view length")
        self.seq = self.seq[:n_kept]
        self._cache.crop(n_kept)

    def _np_valid(self) -> bool:
        return self._np is not None and self._np_pos == len(self.seq) - 1

    def _realign(self, tokens: list[int]):
        """Repair a stale `_np`: re-forward the view's last token (saved
        BEFORE cropping — reading seq[-1] after the crop would fetch the
        wrong token and double-append it to the cache) followed by
        `tokens`. Returns the logits rows for the forwarded span."""
        last = self.seq[-1]
        self._crop(len(self.seq) - 1)
        return self._forward([last] + list(tokens))

    # -- protocol implementation --------------------------------------------

    def reset(self, tokens: list[int]) -> None:
        from transformers import DynamicCache

        self.seq = []
        self._cache = DynamicCache()
        self._np = None
        self._np_pos = -1
        self._forward(list(tokens))

    def sync(self, pre_len: int, emitted: list[int]) -> None:
        """Canonical is now view-prefix[:pre_len] + emitted. Keep any
        speculative prefix that matches `emitted` (already in the KV
        cache), crop the rest, and forward only the remainder — at most
        one correction token after a round, or the truth span at a turn
        transition."""
        keep = pre_len
        for i, tok in enumerate(emitted):
            if pre_len + i < len(self.seq) and self.seq[pre_len + i] == tok:
                keep += 1
            else:
                break
        self._crop(keep)
        rest = emitted[keep - pre_len:]
        if rest:
            self._forward(rest)
        elif keep != self._np_pos + 1 and keep > 0:
            # cropped below the next_pred position without re-forcing:
            # _np is now stale. It will be repaired lazily by the next
            # predict/propose (single-token re-forward).
            pass


class HFVerifier(_HFAdapter):
    """Target side; satisfies the Verifier protocol (see predict())."""

    def predict(self, tokens: list[int]) -> list[int]:
        n = len(tokens)
        if self._np_valid():
            held = self._np  # capture BEFORE _forward clobbers it
            rows = self._forward(list(tokens))  # rows at positions m..m+n-1
            row_arg = [int(r.argmax()) for r in rows]
            # preds[0] is the held argmax for proposal position 0; row j
            # predicts tokens[j+1], so preds[j] = row_arg[j-1]; the last
            # row is the bonus (== the adapter's new _np, set by _forward)
            preds = [held] + row_arg[:-1]
            return preds + [row_arg[-1]]
        # stale _np (post-crop): re-forward the last canonical token
        # together with the proposal; row j sits at position m+j-1.
        rows = self._realign(tokens)
        row_arg = [int(r.argmax()) for r in rows]
        return row_arg[:n] + [row_arg[n]]


class HFDraftProposer(_HFAdapter):
    """Draft side; satisfies the Proposer protocol.

    Greedy autoregressive chaining of up to k tokens; stops early when
    the draft proposes EOS (C2). `eos_id` is the draft's EOS (the
    instruct models use <|im_end|> — same id as the target's).
    """

    def __init__(self, model, eos_id: int):
        super().__init__(model)
        self.eos_id = eos_id

    def propose(self, k: int) -> list[int]:
        if not self._np_valid():  # repair a stale _np with one token
            self._realign([])
        out: list[int] = []
        pred = self._np
        for _ in range(k):
            out.append(pred)
            if pred == self.eos_id:
                break
            self._forward([pred])
            pred = self._np
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_model(model_id: str, dtype, device: str = "auto"):
    """HF model load with device placement ("auto" keeps from_pretrained's
    default — CPU/MPS on the dev host; the GPU host passes cuda:0)."""
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype
    )
    if device != "auto":
        model = model.to(device)
    return model


def build_proposer(kind: str, draft_model: str | None, eos_id: int, dtype, ngram_min: int = 2, ngram_max: int = 5, device: str = "auto"):
    """Return the proposer instance. Draft proposers are stateful and
    reused across records (their cache syncs like the verifier's); n-gram
    proposers are cheap and built fresh per record by the caller."""
    if kind == "draft":
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(
            draft_model, torch_dtype=dtype
        )
        if device != "auto":
            model = model.to(device)
        return HFDraftProposer(model, eos_id)
    if kind == "ngram":
        from src.serving.proposers import NgramProposer

        return NgramProposer(ngram_min, ngram_max)
    raise ValueError(f"unknown proposer kind {kind!r}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Instrumented speculative-decoding loop (plan §6.5)"
    )
    ap.add_argument("records", help="frozen parquet or save_to_disk dir")
    ap.add_argument("--proposer", choices=["draft", "ngram"], required=True)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
        help="first N records in frozen order (deterministic subset)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--ngram-min", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--dtype", default=None, help="e.g. float32 (local "
        "exactness) or bfloat16 (H100 measurement); default float32")
    ap.add_argument("--device", default="auto",
        help="model placement: cuda:0 on the GPU host; 'auto' keeps the "
        "from_pretrained default (CPU/MPS) for the local §4.4 checks")
    ap.add_argument("--out", required=True, help="events JSONL")
    ap.add_argument("--outputs-out", default=None,
        help="per-turn generated ids JSONL (§4.4 exactness diff)")
    ap.add_argument("--meta-out", default=None, help="run metadata JSON")
    args = ap.parse_args()

    import torch

    dtype = getattr(torch, args.dtype) if args.dtype else torch.float32
    import transformers  # noqa: F401 — presence check, mirroring VLLMEngine

    eos_id = get_eos_id()
    target = load_model(args.target_model, dtype, args.device)
    verifier = HFVerifier(target)

    records = load_records(args.records, args.limit)
    if args.proposer == "draft":
        proposer = build_proposer(
            "draft", args.draft_model, eos_id, dtype, device=args.device
        )
        factory = lambda: proposer
    else:
        factory = lambda: build_proposer(
            "ngram", None, eos_id, dtype, args.ngram_min, args.ngram_max
        )

    all_events: list[dict] = []
    all_outputs: list[dict] = []
    for rec in records:
        events, outputs = generate_record(
            rec, factory(), verifier, eos_id, args.k,
            args.max_new_tokens, args.max_steps,
        )
        all_events.extend(events)
        for o in outputs:
            entry = {"query_id": rec["query_id"], **o}
            all_outputs.append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in all_events:
            f.write(json.dumps(e) + "\n")

    if args.outputs_out:
        po = Path(args.outputs_out)
        po.parent.mkdir(parents=True, exist_ok=True)
        with open(po, "w") as f:
            for o in all_outputs:
                f.write(json.dumps(o) + "\n")

    if args.meta_out:
        import subprocess

        meta = {
            "proposer": args.proposer,
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "k": args.k,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "dtype": args.dtype or "float32",
            "device": args.device,
            "n_records": len(records),
            "n_events": len(all_events),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip() or None,
        }
        mp = Path(args.meta_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta, indent=2))

    print(
        f"wrote {len(all_events)} events, {len(all_outputs)} turn outputs "
        f"({len(records)} records) -> {out}"
    )


if __name__ == "__main__":
    main()
