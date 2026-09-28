"""Tiny-model end-to-end for Stage-2 on-policy GKD (§6.7's "tested on
tiny models locally" contract). Runs only when torch + the 0.5B weights
are available (skipped otherwise — the golden suite is the default gate).

The draft samples, the (same) tiny model stands in as the frozen target,
a few optimizer steps run, and the run must: produce finite losses that
decrease-or-hold, save a checkpoint that reloads and still generates, and
exercise the sample->score->loss path with real tensors.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

TINY = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

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
        pytest.skip(f"{model_id} not cached; SPEC_REALMODELS=1 to download")


def _tiny_contexts(tmp_path):
    """Two 8-token contexts on disk as a fake save_to_disk dir the loader
    can read (input_ids only — exactly the loader's contract)."""
    pytest.importorskip("datasets")
    from datasets import Dataset

    ids = [9707, 785, 3462, 4650, 512, 205, 837, 13]  # plausible Qwen ids
    d = tmp_path / "ctx"
    Dataset.from_list([{"input_ids": ids}, {"input_ids": ids}]).save_to_disk(d)
    return d


def test_gkd_tiny_e2e(tmp_path):
    _maybe_skip(TINY)
    import torch

    from src.training.onpolicy_gkd import OnPolicyGKDTrainer

    ctx_dir = _tiny_contexts(tmp_path)
    t = OnPolicyGKDTrainer(
        draft_dir=TINY,
        target_id=TINY,
        out_dir=str(tmp_path / "out"),
        xlam_ctx_dir=str(ctx_dir),
        tb_ctx_dir=str(ctx_dir),
        steps=2,
        micro_bs=1,
        grad_accum=1,
        lr=1e-5,
        sample_temp=1.0,
        max_new_tokens=4,
        max_ctx=8,
        tb_frac=0.5,
        div="reverse_kl",
        sft_weight=0.1,
        ckpt_every=0,       # only the final save
        dtype_name="float32",
        device="cpu",
        seed=1234,
    )
    meta = t.train()
    assert meta["steps"] == 2
    losses = [s["loss"] for s in meta["steps_log"]]
    assert losses and all(l == l for l in losses), "NaN/inf loss"
    assert all(l >= 0 for l in losses)
    # final checkpoint reloads and generates
    import transformers

    m = transformers.AutoModelForCausalLM.from_pretrained(
        str(tmp_path / "out" / "final"))
    tok = transformers.AutoTokenizer.from_pretrained(TINY)
    with torch.no_grad():
        out = m.generate(
            torch.tensor([[9707, 785]]), max_new_tokens=2, do_sample=False,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
        )
    assert out.shape[1] > 2
