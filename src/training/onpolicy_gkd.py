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
     1.0, plan §2 temperature note), max --max-new-tokens (512), with a
     per-context seeded generator (reseeded from (seed, micro, sample)
     before each draw — reproducible given weights) and a K1 logits
     processor masking ids >= the real vocab to -inf: the draft's padding
     rows [151,665, vocab) are untrained random init, occasionally
     reachable, and a sampled padding id would become a loss-carrying
     'state' the target can never accept. Sampling (not argmax) makes the
     draft visit a spread of its own states — the point of on-policy.
     The sampled sequence's tokens carry loss (they are states the draft
     actually proposes); the sampled EOS carries loss too (the draft must
     keep learning to stop).
  G2 target scoring: ONE forward pass of the frozen target over
     prompt + sampled tokens, no_grad, bf16, with the same SCATTERED
     logits_to_keep rows as the student (K3: positions len(ctx)-1 ..
     len(ctx)+n-2 — the rows predicting exactly the sampled tokens; the
     teacher materializes only n rows, not ctx x 152k). Its rows are the
     teacher distribution (renormalized over the real vocab, K1
     slicing). The target never trains.
  G3 divergence: reverse KL  KL(student || teacher)  by default
     (mode-seeking — acceptance needs the draft's argmax inside the
     target's support, not broad mass cover), with --div {reverse_kl,
     forward_kl, jsd} switchable per plan §2's "forward/reverse KL and
     JSD"; plus the CE anchor on the target's argmax token at
     --sft-weight (same acceptance-relevant term as Stage-1's K-series;
     the bonus-token convention: the target's argmax IS what the verify
     step would emit).
  G4 mixing (plan §2.4, GKD's lambda): every K-th micro-batch (K =
     round(1/--mix-sft-frac), min 2; default 0.25 -> every 4th) is a
     Stage-1 KD pack instead of an on-policy batch, run through
     kd_warmstart's own pack_records/batch_tensors/kd_row_losses path —
     K1-K7 apply to that stream unchanged, with --s1-sft-weight as its
     CE anchor. Deterministic slot plan (a pure function of the run
     length), packs cycle if fewer than needed; 0 disables. The Stage-1
     stream is the stability anchor so early sampling noise cannot drag
     the draft off the Stage-1 basin.
  G5 contexts: mixed pools (build_stage2_contexts.py) — xLAM
     training-pool minus stage1 (52,794) and TB prefix conversations
     (45,023; tools disjoint from TB-500 by construction). A CONTEXT is
     the PROMPT PREFIX ending at a sampling boundary with the template's
     fresh assistant header: input_ids[:n_prompt_tokens] for single-turn
     xLAM, ONE PER ASSISTANT-TURN START for multi-turn TB (post-
     observation states are the point of the TB mix, plan §3.2/§9 — a
     turn-0-only cut would reduce TB to "xLAM with different prompts").
     Pools mixed --tb-frac (default 0.5 by conversation, plan §6.9's
     1:1) with a seeded shuffle so the stream is reproducible offline;
     prefixes over --max-ctx (4096, plan's 4-6k) are left-truncated to
     the last max_ctx tokens, then dropped forward to the next
     <|im_start|> turn boundary when the cut lands mid-turn.
  G6 checkpoint + tau probe: every --ckpt-every steps (default ~1 GPU
     hour of steps) the trainer saves to <out>/step_<N>, probes tau on
     --tau-limit frozen xLAM-eval records (the instrumented loop's own
     HFDraftProposer/HFVerifier around the in-process models — the
     draft proposes, the frozen TARGET verifies), and writes
     tau_<step>.json (the plan's "hourly checkpoints with quick tau
     check"). Every save — checkpoints and final — is vocab-padded at
     save time (K7 treatment) so any is servable by vLLM as-is. The
     samples JSONL (<out>/samples.jsonl) is the run's on-policy record:
     one line per sampled context (source, lengths) — on-policy data is
     a function of the weights at sampling time, so unlike Stage 1 it
     cannot be re-rendered offline.
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
DEFAULT_S1_DATA = REPO_ROOT / "data" / "processed" / "stage1_kd"
DEFAULT_TAU_RECORDS = REPO_ROOT / "frozen" / "xlam_eval.parquet"
S1_PACK_MAX_LEN = 8192  # the Stage-1 mixing stream keeps Stage 1's pack cap
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
    input_ids; each context is the record's PROMPT PREFIX — input_ids up to
    the sampling boundary (n_prompt_tokens for xLAM; for TB, one context
    per assistant-turn start: post-observation states are the point of
    the TB mix, plan §3.2/§9 — a turn-0-only cut would reduce TB to
    "xLAM with different prompts"), ending with the template's fresh
    assistant header. Prefixes longer than max_ctx are left-truncated to
    their LAST max_ctx tokens, then dropped forward to the next
    <|im_start|> when the cut lands mid-turn so the context always
    starts at a turn boundary (the tool-schema system block may drop —
    that is the accepted cost of the cap, same as §2.6's trim). Returns
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

    def _prefixes(r: dict, source: str) -> list[list[int]]:
        """The sampling boundaries of one record (G5). Labeled records
        (the real pools): one context per assistant-turn start for TB
        (post-observation states are the point of the TB mix), the
        n_prompt_tokens boundary for xLAM. Bare records (input_ids only,
        e.g. hand-built test pools): the whole input_ids is the context —
        the record IS a prompt."""
        ids = list(r["input_ids"])
        if "labels" in r:
            if source == "xlam" and "n_prompt_tokens" in r:
                return [ids[: int(r["n_prompt_tokens"])]]
            from src.analysis.eval_acceptance import _turn_starts_from_labels

            starts = _turn_starts_from_labels(list(r["labels"]))
            return [ids[:s] for s in starts]
        if "n_prompt_tokens" in r:
            return [ids[: int(r["n_prompt_tokens"])]]
        return [ids]  # bare prompt record: the whole thing is context

    pool: list[dict] = []
    for src, d in (("xlam", xlam_dir), ("tb", tb_dir)):
        ds = load_from_disk(str(d))
        for r in ds:
            for pre in _prefixes(r, src):
                pool.append({"input_ids": pre, "source": src})
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
        s1_data: str = str(DEFAULT_S1_DATA),
        s1_sft_weight: float = 0.1,
        div: str = "reverse_kl",
        sft_weight: float = 0.1,
        ckpt_every: int = 100,
        tau_records: str = str(DEFAULT_TAU_RECORDS),
        tau_limit: int = 25,
        tau_k: int = 5,
        tau_max_new: int = 128,
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
        self.s1_data = s1_data
        self.s1_sft_weight = s1_sft_weight
        self.div = div
        self.sft_weight = sft_weight
        self.ckpt_every = ckpt_every
        self.tau_records = tau_records
        self.tau_limit = tau_limit
        self.tau_k = tau_k
        self.tau_max_new = tau_max_new
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

    def _mix_plan(self, n_micro: int) -> list[bool]:
        """G4's deterministic slot plan: True = a Stage-1 pack slot.
        Every K-th micro-batch is s1 (K = round(1/f)); f >= 0.5 is refused
        (K = 1 would mean no on-policy batches at all — the mixing is a
        stabilizing MINORITY by construction); the plan is a pure
        function of (n_micro, mix_sft_frac)."""
        if self.mix_sft_frac < 0 or self.mix_sft_frac >= 0.5:
            raise ValueError(
                f"mix_sft_frac must be in [0, 0.5), got "
                f"{self.mix_sft_frac} — the Stage-1 stream is a stabilizing "
                "minority, not the majority"
            )
        if self.mix_sft_frac == 0:
            return [False] * n_micro
        K = max(2, round(1.0 / self.mix_sft_frac))
        return [(i + 1) % K == 0 for i in range(n_micro)]

    def _save(self, model, tag: str, pad: bool = True) -> Path:
        """Save an HF dir (G6). pad=True applies prepare_draft's zero-row
        treatment so any checkpoint is servable by vLLM as-is (K7). The
        padding happens on a RELOADED COPY, never the live model —
        resize_token_embeddings mid-run would mutate the training weights
        (and desync the optimizer's per-param buffers), which measured as
        a hard backward shape error. The live model's numerics are
        unaffected either way: the loss/sampler never touch the padding
        rows (K1)."""
        import transformers

        out = self.out_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        if pad:
            from src.serving.prepare_draft import target_vocab_size

            new_vocab = target_vocab_size()
            if int(model.config.vocab_size) != new_vocab:
                import shutil

                tmp = self.out_dir / f"_pad_tmp_{tag}"
                tmp.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(tmp)
                m2 = transformers.AutoModelForCausalLM.from_pretrained(
                    tmp, torch_dtype=next(
                        p.dtype for p in model.parameters()
                    )
                )
                old_vocab = int(m2.config.vocab_size)
                try:
                    m2.resize_token_embeddings(new_vocab, mean_resizing=False)
                except TypeError:  # older signature
                    m2.resize_token_embeddings(new_vocab)
                m2.get_input_embeddings().weight.data[old_vocab:] = 0
                if not getattr(m2.config, "tie_word_embeddings", False):
                    out_emb = m2.get_output_embeddings()
                    if out_emb is not None:
                        out_emb.weight.data[old_vocab:] = 0
                m2.save_pretrained(out)
                del m2
                shutil.rmtree(tmp)
                get_tokenizer().save_pretrained(out)
                return out
        model.save_pretrained(out)
        get_tokenizer().save_pretrained(out)
        return out

    def _tau_probe(self, draft, target, tok) -> dict:
        """G6's quick tau check: the instrumented loop's own adapters around
        the IN-PROCESS models over the frozen xLAM-eval head — HFDraftProposer
        on the current draft, HFVerifier on the frozen TARGET (the draft
        verifying itself would make alpha trivially 1) — greedy, k=tau_k,
        scored by the analyzer's flat_metrics. Runs under eval + no_grad so
        grad-ckpt stays inactive; the adapters pass use_cache=True per
        call."""
        import torch
        from src.serving.instrumented_spec import (
            HFDraftProposer,
            HFVerifier,
            generate_record,
        )
        from src.analysis.eval_acceptance import flat_metrics, load_records

        records = load_records(self.tau_records, self.tau_limit)
        eos = int(tok.eos_token_id)
        was_training = draft.training
        draft.eval()
        events: list[dict] = []
        with torch.no_grad():
            proposer = HFDraftProposer(draft, eos)
            verifier = HFVerifier(target)
            for rec in records:
                ev, _ = generate_record(
                    rec, proposer, verifier, eos,
                    self.tau_k, self.tau_max_new,
                )
                events.extend(ev)
        if was_training:
            draft.train()
        return flat_metrics(events, self.tau_k)

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
        tok = get_tokenizer()
        eos_id = int(tok.eos_token_id)

        st = _TrainState()
        ci = 0  # context cursor
        t0 = time.perf_counter()
        steps_log: list[dict] = []
        ckpt_log: list[dict] = []

        def _next_contexts(n: int) -> list[dict]:
            nonlocal ci
            batch = []
            while len(batch) < n:
                batch.append(contexts[ci % len(contexts)])
                ci += 1
            return batch

        def _sample_and_score(ctx: dict, ctx_seed: int):
            """G1+G2: draft samples, target scores. Returns tensors for
            the loss (student rows need grad; teacher rows detached).
            Sampling runs with the per-context seed and a K1 logits
            processor (padding ids masked to -inf — a padding id must
            never become a loss-carrying sampled 'state')."""
            import torch

            ids = torch.tensor([ctx["input_ids"]], dtype=torch.long,
                               device=device)
            n_ctx = ids.shape[1]

            def _mask_padding(input_ids, scores):
                # K1 at sampling time: the draft's untrained padding rows
                # [151,665, vocab) must never be drawn. LogitsProcessor
                # signature is (input_ids, scores) — scores is arg 2.
                return scores.index_fill(
                    1,
                    torch.arange(real_vocab, scores.shape[1],
                                 device=scores.device),
                    float("-inf"),
                )

            with torch.no_grad():
                draft.eval()
                # per-context determinism: transformers 5.17's generate()
                # takes no generator kwarg — seed the global RNG (the
                # sampling draws are the only consumer between seed and
                # draw; reproducible given weights)
                torch.manual_seed(ctx_seed)
                gen_out = draft.generate(
                    ids,
                    do_sample=True,
                    temperature=self.sample_temp,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    top_k=0,          # full distribution over the real vocab
                    top_p=1.0,
                    logits_processor=[_mask_padding],
                )
                draft.train()
            sampled = gen_out[0][n_ctx:]
            if len(sampled) == 0:
                return None
            full = torch.cat([ids[0], sampled], dim=0).unsqueeze(0)
            keep = torch.arange(n_ctx - 1, n_ctx - 1 + len(sampled),
                                device=device)
            with torch.no_grad():  # G2: the SAME scattered keep — and the
                t_out = target(  # target materializes only the n loss rows
                    input_ids=full, logits_to_keep=keep, use_cache=False
                )
            teacher_rows = t_out.logits[0]
            s_out = draft(input_ids=full, logits_to_keep=keep, use_cache=False)
            return s_out.logits[0], teacher_rows, int(len(sampled))

        # -- G4: the Stage-1 fixed stream (kd_warmstart's own pack path) ----
        s1_packs: list = []
        k_pad = 0
        s1_stats = {"n_s1_batches": 0, "n_s1_rows": 0}
        if self.mix_sft_frac > 0:
            if not Path(self.s1_data).exists():
                raise SystemExit(
                    f"--mix-sft-frac {self.mix_sft_frac} > 0 but "
                    f"{self.s1_data} does not exist — assemble the Stage-1 "
                    "KD dataset first (build_distill_data assemble) or "
                    "pass --mix-sft-frac 0"
                )
            from src.training.kd_warmstart import (
                batch_tensors as s1_batch_tensors,
                kd_row_losses,
                pack_records,
                topk_pad,
            )
            from src.analysis.eval_acceptance import load_records

            s1_records = load_records(self.s1_data)
            s1_packs, _ = pack_records(s1_records, max_len=S1_PACK_MAX_LEN)
            if not s1_packs:
                raise SystemExit(f"no usable packs in {self.s1_data}")
            k_pad = topk_pad(s1_packs)

        global_step = 0
        micro = 0
        n_micro_total = self.steps * self.grad_accum
        mix_plan = self._mix_plan(n_micro_total)
        s1_i = 0
        n_samples = 0
        samples_path = self.out_dir / "samples.jsonl"
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        samples_f = open(samples_path, "w")

        optimizer.zero_grad(set_to_none=True)
        while global_step < self.steps:
            ctxs = _next_contexts(self.micro_bs)
            n_rows = 0
            losses_sum = 0.0
            is_s1 = mix_plan[micro] if micro < len(mix_plan) else False
            if is_s1 and s1_packs:
                # G4: one Stage-1 KD pack through the Stage-1 loss (K1-K7
                # apply to this stream unchanged)
                pack = s1_packs[s1_i % len(s1_packs)]
                s1_i += 1
                t_ids, t_lps, valid, labels, keep = s1_batch_tensors(
                    pack, k_pad, device
                )
                pids = torch.tensor([pack.input_ids], dtype=torch.long,
                                    device=device)
                pos = torch.tensor([pack.position_ids], dtype=torch.long,
                                   device=device)
                out = draft(input_ids=pids, position_ids=pos,
                            logits_to_keep=keep, use_cache=False)
                per = kd_row_losses(
                    out.logits[0], t_ids, t_lps, valid, labels,
                    real_vocab, self.s1_sft_weight,
                )
                n_tok = len(pack.loss_positions)
                (per.sum() / self.grad_accum).backward()
                losses_sum += float(per.sum().item())
                n_rows += n_tok
                s1_stats["n_s1_batches"] += 1
                s1_stats["n_s1_rows"] += n_tok
            else:
                for c in ctxs:
                    ctx_seed = (self.seed * 1_000_003 + micro * 31
                                + n_samples) % 2**63
                    got = _sample_and_score(c, ctx_seed)
                    if got is None:
                        continue
                    s_rows, t_rows, n_tok = got
                    per_c = gkd_losses(s_rows, t_rows, real_vocab,
                                       self.div, self.sft_weight)
                    (per_c.sum() / (len(ctxs) * self.grad_accum)).backward()
                    n_rows += n_tok
                    losses_sum += float(per_c.sum().item())
                    samples_f.write(json.dumps({
                        "micro": micro, "source": c.get("source"),
                        "n_ctx_tokens": len(c["input_ids"]),
                        "n_sampled": n_tok,
                    }) + "\n")
                    n_samples += 1
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
                        "is_s1": is_s1,
                    })
                    print(json.dumps(steps_log[-1]), flush=True)
                if self.ckpt_every and global_step % self.ckpt_every == 0:
                    p = self._save(draft, f"step_{global_step}")
                    tau = self._tau_probe(draft, target, tok)
                    (self.out_dir / f"tau_{global_step}.json").write_text(
                        json.dumps(tau, indent=2)
                    )
                    print(f"checkpoint: {p} (tau={tau['tau']:.4f})", flush=True)
                    ckpt_log.append({
                        "step": global_step, "ckpt": str(p),
                        "tau": tau["tau"], "alpha": tau["alpha"],
                    })
                if global_step == self.steps:
                    break

        final = self._save(draft, "final")
        final_tau = self._tau_probe(draft, target, tok)
        (self.out_dir / "tau_final.json").write_text(
            json.dumps(final_tau, indent=2)
        )
        samples_f.close()
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
            "s1_data": self.s1_data,
            "s1_sft_weight": self.s1_sft_weight,
            "s1_stats": s1_stats,
            "div": self.div,
            "sft_weight": self.sft_weight,
            "dtype": self.dtype_name,
            "device": self.device,
            "seed": self.seed,
            "n_contexts": len(contexts),
            "samples_out": str(samples_path),
            "n_samples": n_samples,
            "real_vocab": real_vocab,
            "wall_s": time.perf_counter() - t0,
            "checkpoints": ckpt_log,
            "final_tau": final_tau,
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
    ap.add_argument("--s1-data", default=str(DEFAULT_S1_DATA),
                    help="Stage-1 assembled KD dataset (the G4 mixing stream)")
    ap.add_argument("--div", choices=["reverse_kl", "forward_kl", "jsd"],
                    default="reverse_kl")
    ap.add_argument("--sft-weight", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--tau-records", default=str(DEFAULT_TAU_RECORDS))
    ap.add_argument("--tau-limit", type=int, default=25)
    ap.add_argument("--tau-k", type=int, default=5)
    ap.add_argument("--tau-max-new", type=int, default=128)
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
        mix_sft_frac=args.mix_sft_frac, s1_data=args.s1_data, div=args.div,
        sft_weight=args.sft_weight, ckpt_every=args.ckpt_every,
        tau_records=args.tau_records, tau_limit=args.tau_limit,
        tau_k=args.tau_k, tau_max_new=args.tau_max_new,
        dtype_name=args.dtype, device=args.device, seed=args.seed,
        limit_ctx=args.limit_ctx,
    )
    meta = t.train()
    print(json.dumps({k: v for k, v in meta.items() if k != "steps_log"},
                     indent=2))


if __name__ == "__main__":
    main()
