"""Stage-1 supervised KD warm-start (plan.md §2 Stage 1, §5 hours 4.5-5.5).

Full fine-tunes the fixed 0.5B draft (Qwen2.5-Coder-0.5B-Instruct, §8.1)
against the frozen target's greedy top-k distributions stored in the KD
dataset built by src/data_prep/build_distill_data.py (its G1-G7 docstring
is the data-side contract; this file is the loss-side one). Loss lives on
the target's emitted tokens only (the dataset's labels, incl. its EOS);
the divergence is a top-k forward KL with the teacher distribution
renormalized over its top-k support, plus an optional CE anchor on the
teacher's argmax (the acceptance-relevant term: the draft is used greedy,
so argmax agreement IS the metric).

Pinned conventions (golden-tested in tests/test_kd_warmstart.py; the
tiny-model end-to-end runs in tests/test_kd_warmstart_realmodels.py):

  K1 vocab slicing (plan §1.2 / §9 "slice to 151,665 in the loss"): both
     student and teacher tensors are sliced to the real tokenizer vocab
     before any softmax/normalization, so the padded rows (ids >=
     151,665 — the tokenizer can never emit them) contribute neither
     probability mass nor gradients. REAL_VOCAB is read from the
     *tokenizer* (the one value shared by every Qwen2.5 member), never
     from either model's config.vocab_size (151,936 draft / 152,064
     target — they differ by design; both are pure padding).
  K2 teacher renormalization: the stored top-k logprobs are the target's
     full-vocab distribution restricted to its top k; they are
     renormalized (logsumexp over the k entries) to form the KD target —
     the standard top-k KD estimator. Positions whose logprob row is null
     (the withheld-EOS case, G2) fall back to a plain CE on the emitted
     token: no teacher distribution, but the token itself is ground truth.
  K3 loss masking + the shift: a label at sequence position p is trained
     from the logits row at position p-1 (logits_to_keep receives p-1 —
     scattered positions, verified to match the corresponding full rows).
     Only labels != -100 contribute (the dataset's labels cover exactly
     the target's generation). A record's first token can never carry a
     label (every record has a rendered prompt, so its first loss
     position is >= 1) — the packer asserts this, because a label at
     record position 0 would be predicted from the previous packed
     record's last token (or from nothing).
  K4 packing (plan §2 "seq len 8192 with packing"): records are
     concatenated greedily in frozen order into packs of <= max_len
     tokens (never split — a split record's tail would attend to a
     stranger's context); each pack carries restarted position_ids
     (0..len-1 per record). attention_mask is NOT passed: transformers
     5.17 auto-detects the packed format from position_ids
     (masking_utils.find_packed_sequence_indices) and builds the correct
     block-causal mask for every attention implementation. Verified
     locally 2026-09-27: packed rows reproduce solo forwards
     token-identically under both sdpa and eager, and the "obvious"
     hand-built 4D mask alternative is silently WRONG under eager (bool
     4D masks hit a different convention there) — which is exactly why
     this module relies on the library's packing path instead.
  K5 memory: only loss rows go through the lm_head (scattered
     logits_to_keep), so peak logits memory scales with loss tokens,
     not with max_len x 152k; activations are bounded by gradient
     checkpointing (--grad-ckpt, default on) and micro-batching.
  K6 determinism: no shuffling — dataset order is the assembled KD
     dataset's frozen order (the generation JSONL's order, G6
     everywhere); packing is a pure function of that order + max_len,
     so the packed stream is reproducible offline from committed
     artifacts alone. Seeds only affect init/dropout-free numerics.
  K7 full FT, not LoRA (plan §2): the trained artifact is a plain HF
     model dir loadable by vLLM as a draft. Saved to <out>/final with
     the target's tokenizer (the repo renders everything with it).
     --save-vocab-padded additionally resizes the saved draft to the
     target's padded vocab_size with zero rows AFTER training (the
     prepare_draft treatment, so serving can use it directly); training
     numerics are unaffected either way because K1 never touches the
     padding.

Loss shape, per loss token at teacher rows: (1-w)*KL + w*CE with
w = --sft-weight (default 0.1; 0 = the plan's literal pure-KD loss). The
KL is computed in fp32 (the whole game is tail mass; bf16 logprobs lose
it). Padded teacher columns (ragged rows widened to a rectangle) carry
p=0 mass and are masked out of the sum — 0 * -inf is NaN, not 0.

CLI (thin; every behavior is an importable, tested function):
  python -m src.training.kd_warmstart --data data/processed/stage1_kd \
      --draft Qwen/Qwen2.5-Coder-0.5B-Instruct --epochs 2 \
      --dtype bfloat16 --device cuda:0 --out checkpoints/stage1
  Local toy dry run: --limit 32 --max-len 512 --epochs 1 --dtype float32
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.analysis.eval_acceptance import load_records
from src.data_prep.render import get_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DRAFT = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "stage1_kd"
DEFAULT_OUT = REPO_ROOT / "checkpoints" / "stage1"

DEFAULT_SFT_WEIGHT = 0.1  # CE anchor; --sft-weight 0 = pure top-k KL
KD_TOPK_CAP = 20          # plan §2: top-20 logprobs everywhere


# ---------------------------------------------------------------------------
# Real-vocab guard (K1)
# ---------------------------------------------------------------------------


def real_vocab_size(tokenizer=None) -> int:
    """The one true vocab: len(tokenizer) — ids above it are pure padding.

    Both the draft (151,936) and the target (152,064) pad their embeddings
    past this value; the padding differs BY DESIGN (plan §1.2), which is
    exactly why the loss must not use either config.vocab_size.
    """
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    n = int(len(tok))
    if n < 151_665:  # the entire Qwen2.5 family shares one tokenizer
        raise AssertionError(
            f"tokenizer vocab {n} < 151,665 — wrong tokenizer family; the "
            "vocab-slicing loss (K1) must run over the real vocab"
        )
    return n


# ---------------------------------------------------------------------------
# Packing (K4, K6)
# ---------------------------------------------------------------------------


@dataclass
class Pack:
    """One packed training row: concatenated records, restarted positions."""

    input_ids: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    # label positions in the PACKED row (token ids live in input_ids there);
    # parallel lists, appended in lockstep, strictly increasing (K6)
    loss_positions: list[int] = field(default_factory=list)
    teacher_token_ids: list[list[int] | None] = field(default_factory=list)
    teacher_logprobs: list[list[float] | None] = field(default_factory=list)
    record_ids: list = field(default_factory=list)  # query_id per record
    n_records: int = 0

    def __len__(self) -> int:
        return len(self.input_ids)


def _teacher_rows(rec: dict) -> tuple[list[list[int] | None], list[list[float] | None]]:
    """Per-label teacher rows for one record.

    The KD dataset stores logprobs only for the GENERATED tokens; the
    labels cover exactly those tokens (G3/G5), so teacher row j belongs
    to the j-th label. Any length mismatch is a hard error — it means the
    assembled record violates the G2/G3 contract and must be rebuilt.
    """
    n_labels = sum(1 for l in rec["labels"] if l != -100)
    ids_rows = list(rec["gen_logprob_token_ids"])
    val_rows = list(rec["gen_logprob_values"])
    if len(ids_rows) != n_labels or len(val_rows) != n_labels:
        raise AssertionError(
            f"record {rec.get('query_id')}: {n_labels} labels but "
            f"{len(ids_rows)}/{len(val_rows)} teacher rows — the assembled "
            "dataset's G2/G3 contract is violated; rebuild with "
            "build_distill_data assemble"
        )
    return ids_rows, val_rows


def pack_records(records: list[dict], max_len: int) -> tuple[list[Pack], int]:
    """Greedy packing in frozen record order (K4, K6); returns (packs,
    n_skipped_too_long). Records longer than max_len are skipped and
    counted — never split: a split record's tail would carry labels whose
    predictor rows attend to a stranger's record (block-causal masking
    makes that wrong, not just ugly)."""
    packs: list[Pack] = []
    cur = Pack()
    n_skipped = 0
    for rec in records:
        ids = list(rec["input_ids"])
        if len(ids) > max_len:
            n_skipped += 1
            continue
        if cur.n_records and len(cur) + len(ids) > max_len:
            packs.append(cur)
            cur = Pack()
        rel_labels = [i for i, l in enumerate(rec["labels"]) if l != -100]
        if rel_labels and min(rel_labels) == 0:
            raise AssertionError(  # K3: the label's predictor would be the
                f"record {rec.get('query_id')} has a label at position 0 — "  # previous pack record
                "its logits row does not exist inside the record (K3)"
            )
        off = len(cur)
        t_ids, t_vals = _teacher_rows(rec)
        cur.input_ids.extend(ids)
        cur.position_ids.extend(range(len(ids)))
        cur.loss_positions.extend(off + i for i in rel_labels)
        cur.teacher_token_ids.extend(t_ids)
        cur.teacher_logprobs.extend(t_vals)
        cur.record_ids.append(rec.get("query_id"))
        cur.n_records += 1
    if cur.n_records:
        packs.append(cur)
    return packs, n_skipped


def pack_stats(packs: list[Pack], n_records: int, n_skipped: int) -> dict:
    n_labels = sum(len(p.loss_positions) for p in packs)
    return {
        "n_records": n_records,
        "n_packs": len(packs),
        "n_skipped_too_long": n_skipped,
        "records_per_pack_median": (
            float(np.median([p.n_records for p in packs])) if packs else None
        ),
        "pack_len_median": (
            float(np.median([len(p) for p in packs])) if packs else None
        ),
        "pack_len_max": max((len(p) for p in packs), default=None),
        "n_label_tokens": n_labels,
        "label_share": round(n_labels / sum(len(p) for p in packs), 4)
        if packs
        else None,
    }


# ---------------------------------------------------------------------------
# Loss (K1, K2) — torch imported lazily; packing above stays torch-free
# ---------------------------------------------------------------------------


def kd_row_losses(
    student_logits,          # [P, V_padded] raw logits rows at p-1 positions
    teacher_ids,             # [P, k] teacher top-k ids, rectangular
    teacher_logprobs,        # [P, k] parallel logprobs; -inf on pad columns
    row_valid,               # [P] bool: a teacher distribution exists
    labels,                  # [P] the emitted token ids (CE anchor)
    real_vocab: int,
    sft_weight: float,
):
    """Per-row losses (a [P] tensor, un-reduced) — see the module docstring
    for the exact form. Rows with row_valid=False contribute CE only
    (K2); rows with a teacher contribute (1-w)*KL + w*CE."""
    import torch

    P, V = student_logits.shape
    if V < real_vocab:
        raise AssertionError(
            f"student vocab {V} < real vocab {real_vocab} — the draft cannot "
            "have a smaller vocab than the tokenizer (K1)"
        )
    if int(labels.max()) >= real_vocab or int(labels.min()) < 0:
        raise AssertionError(  # same K1 logic, token side: the CE anchor
            # target must live in the real vocab too
            "label token outside real vocab — bad KD data (K1)"
        )
    if bool(row_valid.any()) and int(teacher_ids[row_valid].max()) >= real_vocab:
        # The target's own top-k must live in the real vocab. A padding id
        # here would mean the data pipeline stored ids the tokenizer can't
        # emit — clamp()ing it onto some real token would silently corrupt
        # the loss, so fail loudly instead (the assembled-data contract
        # keeps this unreachable; empirically the target's top-20 never
        # contains padding ids on tool-call states, verified 2026-09-27).
        raise AssertionError(
            "teacher row contains a padding id (>= real vocab) — bad KD data"
        )
    sl = student_logits[:, :real_vocab].float()          # K1 slice
    logp = torch.log_softmax(sl, dim=-1)                 # renorm over real vocab
    ce = torch.nn.functional.cross_entropy(sl, labels, reduction="none")

    kl = torch.zeros_like(ce)
    if bool(row_valid.any()):
        idx = row_valid.nonzero(as_tuple=True)[0]
        t_lps = teacher_logprobs[idx].float()
        finite = torch.isfinite(t_lps)                   # pad columns: p = 0
        t_lps = t_lps - torch.logsumexp(  # K2 renorm over the top-k support
            torch.where(finite, t_lps, torch.full_like(t_lps, float("-inf"))),
            dim=-1, keepdim=True,
        )
        p_t = torch.exp(t_lps)
        s_lps = torch.gather(  # student logp at the teacher's support
            logp[idx], 1, teacher_ids[idx].long()
        )
        terms = p_t * (t_lps - s_lps)
        # 0 * -inf is NaN on pad columns — mask, never rely on p_t == 0
        terms = torch.where(finite, terms, torch.zeros_like(terms))
        kl[idx] = terms.sum(dim=-1)

    has_teacher = row_valid.to(ce.dtype)
    return (1.0 - sft_weight) * has_teacher * kl + \
        (1.0 - has_teacher + sft_weight * has_teacher) * ce


def batch_tensors(pack: Pack, k_pad: int, device):
    """Pack -> (teacher_ids, teacher_logprobs, row_valid, labels, keep) on
    `device`; keep = the logits rows to gather (label positions - 1, K3)."""
    import torch

    n = len(pack.loss_positions)
    t_ids = torch.zeros((n, k_pad), dtype=torch.long)
    t_lps = torch.full((n, k_pad), float("-inf"), dtype=torch.float32)
    valid = torch.zeros(n, dtype=torch.bool)
    labels = torch.zeros(n, dtype=torch.long)
    for i, (ids_row, vals_row) in enumerate(
        zip(pack.teacher_token_ids, pack.teacher_logprobs)
    ):
        labels[i] = pack.input_ids[pack.loss_positions[i]]
        if ids_row is None or not ids_row:
            continue  # K2: no teacher row -> CE fallback
        k = min(len(ids_row), k_pad)
        t_ids[i, :k] = torch.tensor(ids_row[:k], dtype=torch.long)
        t_lps[i, :k] = torch.tensor(vals_row[:k], dtype=torch.float32)
        valid[i] = True
    keep = torch.tensor(sorted(p - 1 for p in pack.loss_positions),
                         dtype=torch.long)
    return (t_ids.to(device), t_lps.to(device), valid.to(device),
            labels.to(device), keep.to(device))


def topk_pad(packs: list[Pack]) -> int:
    """Rectangle width for teacher rows: max observed row length, capped at
    the plan's top-20 (rows arrive sorted desc, so a cap keeps the top-k)."""
    m = 1
    for p in packs:
        for row in p.teacher_token_ids:
            if row:
                m = max(m, min(len(row), KD_TOPK_CAP))
    return m


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class KDWarmstartTrainer:
    """Full-FT trainer over the packed KD dataset. Every knob is a ctor arg
    so tests drive it at toy scale; the CLI is a thin wrapper."""

    def __init__(
        self,
        draft_id: str,
        data_dir: str,
        out_dir: str,
        max_len: int = 8192,
        micro_bs: int = 1,
        grad_accum: int = 8,
        lr: float = 1e-5,
        epochs: int = 2,
        sft_weight: float = DEFAULT_SFT_WEIGHT,
        grad_ckpt: bool = True,
        dtype_name: str = "float32",
        device: str = "auto",
        limit: int | None = None,
        seed: int = 1234,
        save_vocab_padded: bool = False,
        log_every: int = 10,
    ):
        self.draft_id = draft_id
        self.data_dir = data_dir
        self.out_dir = Path(out_dir)
        self.max_len = max_len
        self.micro_bs = micro_bs
        self.grad_accum = grad_accum
        self.lr = lr
        self.epochs = epochs
        self.sft_weight = sft_weight
        self.grad_ckpt = grad_ckpt
        self.dtype_name = dtype_name
        self.device = device
        self.limit = limit
        self.seed = seed
        self.save_vocab_padded = save_vocab_padded
        self.log_every = log_every

    # -- data ---------------------------------------------------------------

    def load_packs(self) -> tuple[list[Pack], dict]:
        records = load_records(self.data_dir, self.limit)
        packs, n_skipped = pack_records(records, self.max_len)
        return packs, pack_stats(packs, len(records), n_skipped)

    # -- training -----------------------------------------------------------

    def _save(self, model) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        final = self.out_dir / "final"
        if self.save_vocab_padded:  # K7: prepare_draft's treatment, post hoc
            from src.serving.prepare_draft import target_vocab_size

            new_vocab = target_vocab_size()
            old_vocab = int(model.config.vocab_size)
            if old_vocab != new_vocab:
                try:
                    model.resize_token_embeddings(new_vocab, mean_resizing=False)
                except TypeError:  # older signature
                    model.resize_token_embeddings(new_vocab)
                model.get_input_embeddings().weight.data[old_vocab:] = 0
                if not getattr(model.config, "tie_word_embeddings", False):
                    out_emb = model.get_output_embeddings()
                    if out_emb is not None:
                        out_emb.weight.data[old_vocab:] = 0
        model.save_pretrained(final)
        get_tokenizer().save_pretrained(final)

    def train(self) -> dict:
        import torch
        import transformers

        packs, stats = self.load_packs()
        if not packs:
            raise SystemExit("no packs to train on — empty or all-too-long data")
        k_pad = topk_pad(packs)

        torch.manual_seed(self.seed)
        dtype = getattr(torch, self.dtype_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.draft_id, torch_dtype=dtype
        )
        model.config.use_cache = False
        if self.grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if self.device != "auto":
            model = model.to(self.device)
        model.train()
        self.real_vocab = real_vocab_size()
        model_vocab = int(model.config.vocab_size)
        if model_vocab < self.real_vocab:
            raise AssertionError(
                f"draft vocab {model_vocab} < real vocab {self.real_vocab} (K1)"
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        n_micro = math.ceil(len(packs) / self.micro_bs)
        total_steps = math.ceil(n_micro / self.grad_accum) * self.epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps)
        )

        device = next(model.parameters()).device
        t0 = time.perf_counter()
        step_log: list[dict] = []
        global_step = 0
        n_label_tokens = 0
        for epoch in range(self.epochs):
            accum = 0
            win_loss_sum, win_rows = 0.0, 0
            for bstart in range(0, len(packs), self.micro_bs):
                batch_packs = packs[bstart : bstart + self.micro_bs]
                rows_in_batch = sum(len(p.loss_positions) for p in batch_packs)
                if rows_in_batch == 0:
                    continue
                for pack in batch_packs:  # micro-batch = a few packs, no padding
                    if not pack.loss_positions:
                        continue
                    t_ids, t_lps, valid, labels, keep = batch_tensors(
                        pack, k_pad, device
                    )
                    ids = torch.tensor([pack.input_ids], dtype=torch.long,
                                       device=device)
                    pos = torch.tensor([pack.position_ids], dtype=torch.long,
                                       device=device)
                    out = model(input_ids=ids, position_ids=pos,
                                logits_to_keep=keep, use_cache=False)
                    rows = kd_row_losses(out.logits[0], t_ids, t_lps, valid,
                                         labels, self.real_vocab, self.sft_weight)
                    # token-weighted: divide by this micro-batch's rows so a
                    # window of uneven micro-batches keeps one-token-one-vote
                    # within it; grad_accum on top handles the window
                    (rows.sum() / (rows_in_batch * self.grad_accum)).backward()
                    win_loss_sum += float(rows.sum().item())
                    win_rows += len(pack.loss_positions)
                accum += 1
                if accum == self.grad_accum:
                    self._step(optimizer, sched)
                    accum = 0
                    global_step += 1
                    n_label_tokens += win_rows
                    if global_step % self.log_every == 0 or global_step == total_steps:
                        step_log.append({
                            "step": global_step,
                            "epoch": epoch,
                            "loss": round(win_loss_sum / max(1, win_rows), 5),
                            "lr": sched.get_last_lr()[0],
                        })
                    win_loss_sum, win_rows = 0.0, 0
            if accum:  # flush a partial window at the epoch boundary. The
                # flushed micro-batches were pre-divided by the full
                # grad_accum, so rescale their cached grads up by
                # grad_accum/accum — one-token-one-vote must hold here too.
                if accum < self.grad_accum:
                    scale = self.grad_accum / accum
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                self._step(optimizer, sched)
                global_step += 1
                n_label_tokens += win_rows
                step_log.append({
                    "step": global_step, "epoch": epoch,
                    "loss": round(win_loss_sum / max(1, win_rows), 5),
                    "lr": sched.get_last_lr()[0],
                })
                win_loss_sum, win_rows = 0.0, 0
        wall = time.perf_counter() - t0

        model.eval()
        self._save(model)

        meta = {
            "draft": self.draft_id,
            "data_dir": str(self.data_dir),
            "out_dir": str(self.out_dir),
            "epochs": self.epochs,
            "max_len": self.max_len,
            "micro_bs": self.micro_bs,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "sft_weight": self.sft_weight,
            "grad_ckpt": self.grad_ckpt,
            "dtype": self.dtype_name,
            "device": str(device),
            "seed": self.seed,
            "real_vocab": self.real_vocab,
            "model_vocab": model_vocab,
            "save_vocab_padded": self.save_vocab_padded,
            "teacher_topk_pad": k_pad,
            "pack_stats": stats,
            "n_global_steps": global_step,
            "n_label_tokens": n_label_tokens,
            "wall_s": round(wall, 2),
            "steps_log": step_log,
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    @staticmethod
    def _step(optimizer, sched) -> None:
        optimizer.step()
        sched.step()
        optimizer.zero_grad(set_to_none=True)


def _git_commit() -> str | None:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage-1 supervised KD warm-start (plan §2 Stage 1)"
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="assembled KD dataset dir (build_distill_data assemble)")
    ap.add_argument("--draft", default=DEFAULT_DRAFT)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-len", type=int, default=8192,
                    help="pack length cap in tokens (plan §2: 8192)")
    ap.add_argument("--micro-bs", type=int, default=1,
                    help="packs per micro-batch (no padding; looped)")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="micro-batches per optimizer step — plan §2's global "
                         "batch ~64 is in RECORDS, so set this to ~64 / "
                         "records_per_pack_median from train_meta.json")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sft-weight", type=float, default=DEFAULT_SFT_WEIGHT,
                    help="CE anchor on the teacher's argmax (0 = the plan's "
                         "literal pure-KL loss)")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction,
                    default=True, help="gradient checkpointing (K5)")
    ap.add_argument("--dtype", default="float32",
                    help="float32 locally; bfloat16 on the H100")
    ap.add_argument("--device", default="auto",
                    help="cuda:0 on the GPU host; 'auto' keeps CPU/MPS")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N records in frozen order (dry runs)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-vocab-padded", action="store_true",
                    help="resize the saved draft to the target's padded vocab "
                         "with zero rows after training (K7)")
    ap.add_argument("--metrics-out", default=None,
                    help="append one summary line to results/metrics.jsonl")
    args = ap.parse_args()

    trainer = KDWarmstartTrainer(
        draft_id=args.draft,
        data_dir=args.data,
        out_dir=args.out,
        max_len=args.max_len,
        micro_bs=args.micro_bs,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        sft_weight=args.sft_weight,
        grad_ckpt=args.grad_ckpt,
        dtype_name=args.dtype,
        device=args.device,
        limit=args.limit,
        seed=args.seed,
        save_vocab_padded=args.save_vocab_padded,
    )
    meta = trainer.train()

    if args.metrics_out:
        from src.serving.bench_vllm import append_metrics

        append_metrics(args.metrics_out, {
            "kind": "stage1_kd_warmstart",
            **{k: v for k, v in meta.items() if k != "steps_log"},
        })
    print(json.dumps({k: v for k, v in meta.items() if k != "steps_log"},
                     indent=2))


if __name__ == "__main__":
    main()
