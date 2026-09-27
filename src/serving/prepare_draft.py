"""Pad a Qwen2.5 Coder draft's embedding vocab to the target's (plan §1.2).

vLLM 0.30.0's SpeculativeConfig hard-fails when draft and target
`config.vocab_size` differ ("Target and draft model should have the same
vocabulary size", pydantic ValidationError) — hit by the day-0 smoke test
on the rented H100, 2026-09-27. The Coder drafts (0.5B/1.5B) pad their
embeddings to 151,936; the 14B target pads to 152,064. The *real*
tokenizer vocab is 151,665 in every Qwen2.5 member (plan §1.2), so ids in
[151,936, 152,064) are pure embedding padding: never produced by the
tokenizer, never present in any rendered record. Appending 128 zero rows
to the draft's embedding (and lm_head, when untied) therefore cannot
change any real-token logit.

Residual, stated honestly: a zero lm_head row yields logit 0 for a
padding id, so at a state where every real logit is < 0 the padded draft's
argmax could pick a padding id — the target then rejects it. Cost: one
rejection (tiny τ hit), never correctness (only target argmax tokens are
ever emitted; the §4.4 exactness gate is unaffected). Zero is still the
right init: any fixed nonzero row is a hidden-state-dependent logit and
can be arbitrarily large; 0 is the only bounded choice.

Parity gate (runs at prepare time, fails loudly): greedy generation on a
frozen prompt must be token-identical before vs after padding — the real
rows are bit-identical, so any argmax drift means an init/resize bug.

CLI (GPU host; setup_day0.sh runs it for both drafts):
  python -m src.serving.prepare_draft \
      --draft Qwen/Qwen2.5-Coder-0.5B-Instruct \
      --out drafts/coder-0.5b-padded
  # --target defaults to the frozen target; its vocab_size is read from
  # AutoConfig (config.json only — no weights downloaded).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_TARGET = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_PROMPT_RECORDS = "frozen/xlam_eval.parquet"
PARITY_PROMPT_TOKENS = 64  # CPU-cheap: parity, not quality
PARITY_NEW_TOKENS = 8


def target_vocab_size(target_id: str = DEFAULT_TARGET) -> int:
    """config.json vocab_size of the target (no weights touched)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(target_id)
    vocab = int(cfg.vocab_size)
    if vocab < 151_665:
        raise AssertionError(
            f"target {target_id} vocab_size {vocab} < real tokenizer vocab "
            "151,665 — wrong model or changed tokenizer"
        )
    return vocab


def parity_prompt(records_path: str = DEFAULT_PROMPT_RECORDS) -> list[int]:
    """First frozen eval record's prompt (truncated) — same file on both
    hosts, deterministic across runs."""
    from src.analysis.eval_acceptance import load_records

    rec = load_records(records_path, limit=1)[0]
    return list(rec["input_ids"][:PARITY_PROMPT_TOKENS])


def _greedy(model, prompt: list[int], eos_id: int) -> list[int]:
    import torch

    with torch.no_grad():
        out = model.generate(
            torch.tensor([prompt]),
            max_new_tokens=PARITY_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    return out[0].tolist()[len(prompt):]


def pad_draft(
    draft_id: str,
    out_dir: str,
    target_id: str = DEFAULT_TARGET,
    records_path: str = DEFAULT_PROMPT_RECORDS,
    dtype_name: str = "bfloat16",
    family_guard: bool = True,
) -> dict:
    """Resize the draft's embedding to the target's padded vocab_size with
    zero-init new rows; assert greedy parity; save model + tokenizer.

    family_guard=False is for the tiny-model tests only (their test
    model is intentionally off-family); production callers never pass it."""
    import torch
    import transformers

    dtype = getattr(torch, dtype_name)
    new_vocab = target_vocab_size(target_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(draft_id, torch_dtype=dtype)
    tok = transformers.AutoTokenizer.from_pretrained(draft_id)
    eos_id = int(tok.eos_token_id)
    old_vocab = int(model.config.vocab_size)
    if old_vocab > new_vocab:
        raise SystemExit(
            f"draft {draft_id} vocab {old_vocab} > target {new_vocab} — "
            "padding cannot shrink; check model ids"
        )
    if family_guard and old_vocab < 151_665:
        raise SystemExit(
            f"draft {draft_id} vocab {old_vocab} < real tokenizer vocab — "
            "wrong model family"
        )

    prompt = parity_prompt(records_path)
    before = _greedy(model, prompt, eos_id)

    if old_vocab != new_vocab:
        try:  # transformers 5.x: mean_resizing samples new rows from a
            # fitted distribution — we zero them anyway; skip the fit
            model.resize_token_embeddings(new_vocab, mean_resizing=False)
        except TypeError:  # older signature without the kwarg
            model.resize_token_embeddings(new_vocab)
        d_new = new_vocab - old_vocab
        model.get_input_embeddings().weight.data[old_vocab:] = 0
        if not getattr(model.config, "tie_word_embeddings", False):
            out_emb = model.get_output_embeddings()
            if out_emb is not None:
                out_emb.weight.data[old_vocab:] = 0
    else:
        d_new = 0

    after = _greedy(model, prompt, eos_id)
    if before != after:
        raise SystemExit(
            f"parity FAILED for {draft_id}: {before} != {after} — padding "
            "changed real-token greedy output; do not use this artifact"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {
        "draft": draft_id,
        "target": target_id,
        "old_vocab": old_vocab,
        "new_vocab": new_vocab,
        "rows_added": d_new,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "parity_prompt_tokens": len(prompt),
        "parity_new_tokens": PARITY_NEW_TOKENS,
        "parity_ok": True,
        "dtype": dtype_name,
    }
    (out / "pad_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--records", default=DEFAULT_PROMPT_RECORDS)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    meta = pad_draft(args.draft, args.out, args.target, args.records, args.dtype)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
