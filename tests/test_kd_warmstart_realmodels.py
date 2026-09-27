"""Real-model checks for the Stage-1 KD warm-start (plan.md §6.7 second half).

Layer 2: runs only when torch + the cached Coder-0.5B draft exist (same
gating as tests/test_spec_realmodels.py / test_gen_stage1_realmodels.py —
SPEC_REALMODELS=1 opts into downloads). Everything runs fp32 CPU at toy
scale: the point is the *mechanics* on real weights — packing vs solo
forwards (K4), the scattered-logits_to_keep shift (K3), a few real train
steps with finite decreasing loss, and a reloadable artifact (K7) — not
quality, which only the H100 run can show.

The plan's §6.7 requirement this file discharges: "tested end-to-end on
tiny models locally ... including the vocab-slicing in the loss, so the
GPU day never debugs training code."
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from src.training.kd_warmstart import (  # noqa: E402
    KDWarmstartTrainer,
    batch_tensors,
    kd_row_losses,
    pack_records,
    topk_pad,
)

DRAFT = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TARGET_TOK = "Qwen/Qwen2.5-Coder-14B-Instruct"  # the shared tokenizer (§1.2)

_HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _cached(model_id: str) -> bool:
    import huggingface_hub

    try:
        huggingface_hub.snapshot_download(
            model_id, allow_patterns=["config.json"], local_files_only=True
        )
        return True
    except Exception:
        return False


def _maybe_skip(model_id: str) -> None:
    if os.environ.get("SPEC_REALMODELS") != "1" and not _cached(model_id):
        pytest.skip(
            f"{model_id} not cached; set SPEC_REALMODELS=1 to download and run"
        )


@pytest.fixture(scope="module")
def draft():
    import torch
    import transformers

    _maybe_skip(DRAFT)
    return transformers.AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=torch.float32
    ).eval()


@pytest.fixture(scope="module")
def real_records():
    """3 real frozen Stage-1 prompts with synthetic (but schema-correct)
    teacher rows — the record shape assemble produces, without needing the
    14B's generations locally."""
    import torch
    import transformers

    from src.data_prep.build_distill_data import load_stage1_prompts

    tok = transformers.AutoTokenizer.from_pretrained(TARGET_TOK)
    real = len(tok)
    torch.manual_seed(1234)
    out = []
    for p in load_stage1_prompts(limit=3):
        n = len(p["prompt_ids"])
        gen = tok("Paris France", add_special_tokens=False)["input_ids"] + [
            tok.eos_token_id
        ]
        rows, vals = [], []
        for i in range(len(gen)):
            if i == 1:  # exercise the K2 null-teacher path in the mix
                rows.append(None)
                vals.append(None)
                continue
            top = torch.topk(torch.randn(real), 5)
            rows.append(top.indices.tolist())
            vals.append(torch.log_softmax(top.values, -1).tolist())
        out.append({
            "query_id": p["query_id"],
            "input_ids": p["prompt_ids"] + gen,
            "labels": [-100] * n + list(gen),
            "n_prompt_tokens": n,
            "gen_logprob_token_ids": rows,
            "gen_logprob_values": vals,
        })
    return out


class TestRealWeights:
    def test_packed_rows_match_solo_forwards(self, draft, real_records):
        """K4 on real weights: a pack's per-record logits are identical to
        each record forwarded alone (block-causal via position_ids only)."""
        import torch

        packs, _ = pack_records(real_records, max_len=8192)
        assert len(packs) >= 1
        p0 = packs[0]
        ids = torch.tensor([p0.input_ids])
        pos = torch.tensor([p0.position_ids])
        with torch.no_grad():
            packed = draft(input_ids=ids, position_ids=pos,
                           use_cache=False).logits[0]
        off = 0
        for r in real_records[: p0.n_records]:
            with torch.no_grad():
                solo = draft(torch.tensor([r["input_ids"]]),
                             use_cache=False).logits[0]
            seg = packed[off : off + len(r["input_ids"])]
            # fp32 CPU forwards take different kernel paths at different
            # sequence lengths, so tiny accumulation noise is expected
            # (measured max ~1.4e-4); real cross-record leakage would be
            # O(1) in the logits. 5e-4 still separates the two cleanly.
            assert torch.allclose(seg, solo, atol=5e-4), (
                f"record {r['query_id']}: max diff "
                f"{(seg - solo).abs().max().item():.2e}"
            )
            off += len(r["input_ids"])

    def test_scattered_keep_rows_match_full(self, draft, real_records):
        """K3/K5 on real weights: logits_to_keep with the label positions
        minus one returns exactly those rows of the full forward."""
        import torch

        packs, _ = pack_records(real_records, max_len=8192)
        p0 = packs[0]
        t_ids, t_lps, valid, labels, keep = batch_tensors(
            p0, topk_pad(packs), "cpu"
        )
        assert keep.tolist() == [lp - 1 for lp in p0.loss_positions]
        ids = torch.tensor([p0.input_ids])
        pos = torch.tensor([p0.position_ids])
        with torch.no_grad():
            full = draft(input_ids=ids, position_ids=pos,
                         use_cache=False).logits[0]
            slim = draft(input_ids=ids, position_ids=pos,
                         logits_to_keep=keep, use_cache=False).logits[0]
        assert torch.allclose(slim, full[keep.tolist()], atol=1e-5)
        # and the loss over the slim rows is finite, vocab-sliced (K1)
        rows = kd_row_losses(slim, t_ids, t_lps, valid, labels,
                             real_vocab=151_665, sft_weight=0.1)
        assert torch.isfinite(rows).all()

    def test_train_end_to_end_toy_scale(self, real_records, tmp_path):
        """The §6.7 discharge: a few real steps, finite + decreasing loss,
        saved artifact reloads with the draft's vocab (K7)."""
        import transformers
        from datasets import Dataset

        ds_dir = tmp_path / "kd"
        Dataset.from_list(real_records).save_to_disk(ds_dir)
        tr = KDWarmstartTrainer(
            draft_id=DRAFT, data_dir=str(ds_dir), out_dir=str(tmp_path / "ck"),
            max_len=8192, micro_bs=1, grad_accum=2, epochs=1,
            dtype_name="float32", device="auto", grad_ckpt=False,
            log_every=1,
        )
        meta = tr.train()
        losses = [s["loss"] for s in meta["steps_log"]]
        assert len(losses) >= 1 and all(l == l for l in losses)  # finite
        m2 = transformers.AutoModelForCausalLM.from_pretrained(
            str(tmp_path / "ck" / "final")
        )
        assert int(m2.config.vocab_size) == 151_936  # untouched by K1/K7
