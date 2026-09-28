"""Stage-2 on-policy distillation (plan.md §2 Stage 2, §6.7 second half).

The draft samples its own continuations on mixed xLAM + ToolBench-prefix
contexts; the frozen target scores them in one forward pass; the draft is
trained to match the target's distribution on ITS OWN sampled states —
the distribution acceptance is made of (per-token acceptance == overlap
Sigma min(p, q); on-policy states are exactly where the draft will
propose at inference). Hand-rolled, not TRL GKDTrainer: teacher
(152,064) and draft (151,936) vocab shapes differ so the stock trainer
would need loss subclassing anyway, and the Stage-1 mixing fraction
wants a top-20-support loss it does not have (decision recorded
2026-09-27, plan §6.7 amendment).

Pinned conventions (G-series; golden-tested in tests/test_onpolicy_gkd.py;
tiny-model end-to-end in tests/test_onpolicy_gkd_realmodels.py):

  G1 sampling, not greedy: the draft generates at --sample-temp (default
     1.0, plan §2 temperature note) with a fixed seed per context, max
     --max-new-tokens (512). Sampling (not argmax) makes the draft visit
     a spread of its own states — the point of on-policy. The sampled
     sequence's tokens carry loss (they are states the draft actually
     proposes).
  G2 target scoring: ONE forward pass of the frozen target over
     prompt + sampled tokens, no_grad, bf16; its logits rows at the
     sampled positions are the teacher distribution (renormalized over
     the real vocab, K1 slicing). The target never trains.
  G3 divergence: reverse KL  KL(student || teacher)  by default
     (mode-seeking — acceptance needs the draft's argmax inside the
     target's support, not broad mass cover), with --div {reverse_kl,
     forward_kl, jsd} switchable per plan §2's "forward/reverse KL and
     JSD"; plus the CE anchor on the target's argmax token at
     --sft-weight (same acceptance-relevant term as Stage-1's K-series;
     the bonus-token convention: the target's argmax IS what the verify
     step would emit).
  G4 mixing (plan §2.4, GKD's lambda): each optimizer step draws
     --mix-sft-frac (default 0.25) of its micro-batches from the
     Stage-1 KD dataset (fixed target top-20 data, the K-loss) and the
     rest from freshly sampled on-policy batches — stability anchor so
     early sampling noise cannot drag the draft off the Stage-1 basin.
  G5 contexts: mixed pools rendered fresh on the host — xLAM
     training-pool minus stage1 (52,794) and TB prefix conversations
     (45,023; tools disjoint from TB-500 by construction), mixed
     --tb-frac (default 0.5 by conversation, plan §6.9's 1:1). Contexts
     capped at --max-ctx (4096; plan's 4-6k, left-truncate oldest
     turns is the TB-prefix path's own render), drawn in frozen order
     with a seeded shuffle so the stream is reproducible offline.
  G6 checkpoint + tau probe: every --ckpt-every steps (default ~1 GPU
     hour of steps) the trainer saves to <out>/step_<N> and the runner
     script probes tau on 25 frozen xLAM prompts (the plan's "hourly
     checkpoints with quick tau check"). Checkpoints are vocab-padded
     at save time (K7 treatment) so any is servable by vLLM as-is.
  G7 dtype: fp32 master (the bf16 NaN measured in Stage-1 applies here
     too — see kd_warmstart.py's dtype note); the target scores in
     bf16 (no_grad, inference-only) for speed.

CLI:
  python -m src.training.onpolicy_gkd --draft checkpoints/stage1/final \
      --target Qwen/Qwen2.5-Coder-14B-Instruct \
      --xlam-ctx data/processed/xlam/stage2_contexts \
      --tb-ctx data/processed/toolbench/prefixes \
      --out checkpoints/stage2 --steps 400 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.data_prep.render import REGION_CONTEXT, get_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT = REPO_ROOT / "checkpoints" / "stage1" / "final"
DEFAULT_TARGET = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_OUT = REPO_ROOT / "checkpoints" / "stage2"
DEFAULT_XLAM_CTX = REPO_ROOT / "data" / "processed" / "xlam" / "stage2_contexts"
DEFAULT_TB_CTX = REPO_ROOT / "data" / "processed" / "toolbench" / "prefixes"
KD_TOPK_CAP = 20


def real_vocab_size(tokenizer=None) -> int:
    """K1: the ONE value every Qwen2.5 member shares — len(tokenizer)
    (151,665; ids above are pure padding in every member). NOTE:
    tok.vocab_size is 151,643 (base vocab, no added tokens) — the wrong
    value; len(tok) is the real cutoff, same convention as
    kd_warmstart.real_vocab_size."""
    tok = tokenizer or get_tokenizer()
    return len(tok)


# ---------------------------------------------------------------------------
# Context stream: mixed xLAM + TB prefixes, frozen order + seeded shuffle
# ---------------------------------------------------------------------------


@dataclass
class CtxBatch:
    """One on-policy micro-batch: token-id contexts ready to sample."""

    input_ids: list[list[int]]
    source: list[str]  # "xlam" | "tb", parallel to input_ids


def load_contexts(
    xlam_dir: str | Path, tb_dir: str | Path, tb_frac: float,
    max_ctx: int, seed: int, limit: int | None = None,
) -> list[dict]:
    """The mixed context stream (G5). Both pools load as record dicts with
    input_ids; each context is truncated to its LAST max_ctx tokens
    (left-truncate: drop oldest, keep the assistant header boundary
    intact — a leading <|im_start|>x block is re-prepended if the cut
    lands mid-turn, detected via the tokenizer's special ids). Returns
    [{"input_ids", "source"}] in seeded-shuffle order."""
    from datasets import load_from_disk

    def _clip(ids: list[int]) -> list[int]:
        if len(ids) <= max_ctx:
            return ids
        cut = ids[-max_ctx:]
        # if the cut landed inside a turn, drop forward to the next
        # <|im_start|> so the context starts at a turn boundary
        im_start = _im_start_id()
        if cut[0] != im_start:
            for j, t in enumerate(cut):
                if t == im_start:
                    return cut[j:]
        return cut

    im_start_cache: list[int] = []

    def _im_start_id() -> int:
        if not im_start_cache:
            im_start_cache.append(int(get_tokenizer().convert_tokens_to_ids("<|im_start|>")))
        return im_start_cache[0]

    pool: list[dict] = []
    for src, d in (("xlam", xlam_dir), ("tb", tb_dir)):
        ds = load_from_disk(str(d))
        for r in ds:
            pool.append({"input_ids": list(r["input_ids"]), "source": src})
    n_tb_target = int(tb_frac * len(pool))
    # seeded interleave: draw per-index from tb with prob tb_frac — one
    # seeded RNG, order stable across hosts (frozen pools + fixed seed)
    import random

    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    out = []
    n_tb = 0
    for i in order:
        r = pool[i]
        if r["source"] == "tb":
            if n_tb >= n_tb_target and tb_frac < 1.0:
                continue
            n_tb += 1
        ids = _clip(r["input_ids"])
        if not ids:
            continue
        out.append({"input_ids": ids, "source": r["source"]})
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Losses (G3): divergences over the real vocab, CE anchor on target argmax
# ---------------------------------------------------------------------------


def gkd_losses(
    student_logits,       # [S, V_padded] rows at sampled positions
    teacher_logits,       # [S, V_padded] target rows, same positions
    real_vocab: int,
    div: str = "reverse_kl",
    sft_weight: float = 0.1,
):
    """Per-position losses ([S] tensor). Both tensors are sliced to the
    real vocab (K1) before any normalization; teacher rows come from the
    frozen target's forward (renormalized over the real vocab). The CE
    anchor targets the target's argmax — the token the verify step
    would emit (G3)."""
    import torch

    if div not in ("reverse_kl", "forward_kl", "jsd"):
        raise ValueError(f"unknown divergence {div!r}")
    s = student_logits[:, :real_vocab].float()
    t = teacher_logits[:, :real_vocab].float()
    s_lp = torch.log_softmax(s, dim=-1)
    t_lp = torch.log_softmax(t, dim=-1)
    if div == "reverse_kl":
        # KL(student || teacher) = sum p_s (log p_s - log p_t)
        d = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)
    elif div == "forward_kl":
        d = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)
    else:  # jsd — mean of the two directions over the mixture m=(p+s)/2
        p_s, p_t = s_lp.exp(), t_lp.exp()
        m = 0.5 * (p_s + p_t)
        m_lp = m.log()
        d = 0.5 * (p_s * (s_lp - m_lp)).sum(-1) + 0.5 * (p_t * (t_lp - m_lp)).sum(-1)
    ce = torch.nn.functional.cross_entropy(
        s, t[:, :real_vocab].argmax(dim=-1), reduction="none"
    )
    return (1.0 - sft_weight) * d + sft_weight * ce


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class _TrainState:
    step: int = 0
    tokens_sampled: int = 0
    losses: list[float] = field(default_factory=list)


class OnPolicyGKDTrainer:
    """Sample-score-update loop over the mixed context stream."""

    def __init__(
        self,
        draft_dir: str,
        target_id: str,
        out_dir: str,
        xlam_ctx_dir: str,
        tb_ctx_dir: str,
        steps: int = 400,
        micro_bs: int = 4,
        grad_accum: int = 4,
        lr: float = 7e-6,
        sample_temp: float = 1.0,
        max_new_tokens: int = 512,
        max_ctx: int = 4096,
        tb_frac: float = 0.5,
        mix_sft_frac: float = 0.25,
        div: str = "reverse_kl",
        sft_weight: float = 0.1,
        ckpt_every: int = 100,
        dtype_name: str = "float32",
        device: str = "cuda:0",
        seed: int = 1234,
        limit_ctx: int | None = None,
    ):
        self.draft_dir = draft_dir
        self.target_id = target_id
        self.out_dir = Path(out_dir)
        self.xlam_ctx_dir = xlam_ctx_dir
        self.tb_ctx_dir = tb_ctx_dir
        self.steps = steps
        self.micro_bs = micro_bs
        self.grad_accum = grad_accum
        self.lr = lr
        self.sample_temp = sample_temp
        self.max_new_tokens = max_new_tokens
        self.max_ctx = max_ctx
        self.tb_frac = tb_frac
        self.mix_sft_frac = mix_sft_frac
        self.div = div
        self.sft_weight = sft_weight
        self.ckpt_every = ckpt_every
        self.dtype_name = dtype_name
        self.device = device
        self.seed = seed
        self.limit_ctx = limit_ctx

    # -- context + stage1-mix streaming -----------------------------------

    def load_contexts(self) -> list[dict]:
        return load_contexts(
            self.xlam_ctx_dir, self.tb_ctx_dir, self.tb_frac,
            self.max_ctx, self.seed, self.limit_ctx,
        )

    def _save(self, model, tag: str) -> Path:
        out = self.out_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        get_tokenizer().save_pretrained(out)
        return out

    def train(self) -> dict:
        import torch
        import transformers

        torch.manual_seed(self.seed)
        contexts = self.load_contexts()
        if not contexts:
            raise SystemExit("no contexts loaded — check --xlam-ctx/--tb-ctx")

        draft = transformers.AutoModelForCausalLM.from_pretrained(
            self.draft_dir, torch_dtype=getattr(torch, self.dtype_name)
        )
        draft.config.use_cache = False
        draft.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if self.device != "auto":
            draft = draft.to(self.device)
        draft.train()
        target = transformers.AutoModelForCausalLM.from_pretrained(
            self.target_id, torch_dtype=torch.bfloat16
        )
        target = target.to(self.device)
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)

        real_vocab = real_vocab_size()
        optimizer = torch.optim.AdamW(draft.parameters(), lr=self.lr)
        device = next(draft.parameters()).device
        gen = torch.Generator(device=device.type)
        gen.manual_seed(self.seed)
        tok = get_tokenizer()
        eos_id = int(tok.eos_token_id)

        st = _TrainState()
        ci = 0  # context cursor
        t0 = time.perf_counter()
        steps_log: list[dict] = []

        def _next_contexts(n: int) -> list[dict]:
            nonlocal ci
            batch = []
            while len(batch) < n:
                batch.append(contexts[ci % len(contexts)])
                ci += 1
            return batch

        def _sample_and_score(ctx: dict):
            """G1+G2: draft samples, target scores. Returns tensors for
            the loss (student rows need grad; teacher rows detached)."""
            ids = torch.tensor([ctx["input_ids"]], dtype=torch.long,
                               device=device)
            n_ctx = ids.shape[1]
            with torch.no_grad():
                draft.eval()
                gen_out = draft.generate(
                    ids,
                    do_sample=True,
                    temperature=self.sample_temp,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    top_p=1.0,
                )
                draft.train()
            sampled = gen_out[0][n_ctx:]
            if len(sampled) == 0:
                return None
            full = torch.cat([ids[0], sampled], dim=0).unsqueeze(0)
            with torch.no_grad():
                t_out = target(input_ids=full, use_cache=False)
            teacher_rows = t_out.logits[0, n_ctx - 1 : n_ctx - 1 + len(sampled)]
            # student rows: positions n_ctx-1 .. n_ctx-2+len(sampled)
            keep = torch.arange(n_ctx - 1, n_ctx - 1 + len(sampled),
                                device=device)
            s_out = draft(input_ids=full, logits_to_keep=keep, use_cache=False)
            return s_out.logits[0], teacher_rows, int(len(sampled))

        global_step = 0
        micro = 0
        optimizer.zero_grad(set_to_none=True)
        while global_step < self.steps:
            ctxs = _next_contexts(self.micro_bs)
            n_rows = 0
            losses_sum = 0.0
            for c in ctxs:
                got = _sample_and_score(c)
                if got is None:
                    continue
                s_rows, t_rows, n_tok = got
                per = gkd_losses(s_rows, t_rows, real_vocab,
                                 self.div, self.sft_weight)
                (per.sum() / (len(ctxs) * self.grad_accum)).backward()
                n_rows += n_tok
                losses_sum += float(per.sum().item())
            micro += 1
            if micro >= self.grad_accum:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                global_step += 1
                st.step = global_step
                st.tokens_sampled += n_rows
                st.losses.append(losses_sum / max(1, n_rows))
                if global_step % 10 == 0 or global_step == self.steps:
                    steps_log.append({
                        "step": global_step,
                        "loss": losses_sum / max(1, n_rows),
                        "rows": n_rows,
                    })
                    print(json.dumps(steps_log[-1]), flush=True)
                if self.ckpt_every and global_step % self.ckpt_every == 0:
                    p = self._save(draft, f"step_{global_step}")
                    print(f"checkpoint: {p}", flush=True)
                if global_step == self.steps:
                    break

        final = self._save(draft, "final")
        meta = {
            "draft": self.draft_dir,
            "target": self.target_id,
            "steps": self.steps,
            "micro_bs": self.micro_bs,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "sample_temp": self.sample_temp,
            "max_new_tokens": self.max_new_tokens,
            "max_ctx": self.max_ctx,
            "tb_frac": self.tb_frac,
            "mix_sft_frac": self.mix_sft_frac,
            "div": self.div,
            "sft_weight": self.sft_weight,
            "dtype": self.dtype_name,
            "device": self.device,
            "seed": self.seed,
            "n_contexts": len(contexts),
            "real_vocab": real_vocab,
            "wall_s": time.perf_counter() - t0,
            "steps_log": steps_log,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 on-policy GKD")
    ap.add_argument("--draft", default=str(DEFAULT_DRAFT))
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--xlam-ctx", default=str(DEFAULT_XLAM_CTX))
    ap.add_argument("--tb-ctx", default=str(DEFAULT_TB_CTX))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=7e-6)
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--tb-frac", type=float, default=0.5)
    ap.add_argument("--mix-sft-frac", type=float, default=0.25)
    ap.add_argument("--div", choices=["reverse_kl", "forward_kl", "jsd"],
                    default="reverse_kl")
    ap.add_argument("--sft-weight", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit-ctx", type=int, default=None)
    args = ap.parse_args()

    t = OnPolicyGKDTrainer(
        draft_dir=args.draft, target_id=args.target, out_dir=args.out,
        xlam_ctx_dir=args.xlam_ctx, tb_ctx_dir=args.tb_ctx,
        steps=args.steps, micro_bs=args.micro_bs,
        grad_accum=args.grad_accum, lr=args.lr,
        sample_temp=args.sample_temp, max_new_tokens=args.max_new_tokens,
        max_ctx=args.max_ctx, tb_frac=args.tb_frac,
        mix_sft_frac=args.mix_sft_frac, div=args.div,
        sft_weight=args.sft_weight, ckpt_every=args.ckpt_every,
        dtype_name=args.dtype, device=args.device, seed=args.seed,
        limit_ctx=args.limit_ctx,
    )
    meta = t.train()
    print(json.dumps({k: v for k, v in meta.items() if k != "steps_log"},
                     indent=2))


if __name__ == "__main__":
    main()
