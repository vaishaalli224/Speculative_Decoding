"""§4.4 real-model checks for the instrumented loop (plan.md §6.5).

Layer 2: runs only when torch + the tiny Qwen models are available
(skipped otherwise — the golden Layer-1 suite in test_instrumented_spec.py
is the default gate). Two checks, both CPU/fp32 for determinism:

  self-consistency: draft = target (same 0.5B model in both roles),
  greedy — every proposal token must be accepted, α = 1.0, τ = k.
  exactness: draft = Qwen2.5-0.5B-Instruct (non-Coder, same tokenizer
  family per plan §1.2 — different weights, so rejections actually
  happen and the crop path executes), target = Qwen2.5-Coder-0.5B —
  the loop's full output must equal plain HF greedy generate() on the
  same context, token-for-token.

These are the plan's explicit "sanity check of the harness before the
GPU day"; they also guard the HF adapters' KV-cache sync.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from src.serving.instrumented_spec import (
    HFVerifier,
    HFDraftProposer,
    generate_turn,
    get_eos_id,
)

TINY = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TINY_NONCODER = "Qwen/Qwen2.5-0.5B-Instruct"  # different weights, same tokenizer

# Skip unless the user opted in (downloads ~1-2 GB) or models are cached.
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
def target_model():
    _maybe_skip(TINY)
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(
        TINY, torch_dtype="float32"
    )


@pytest.fixture(scope="module")
def draft_noncoder():
    _maybe_skip(TINY_NONCODER)
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(
        TINY_NONCODER, torch_dtype="float32"
    )


def _prompt(record: dict, n_prompt: int) -> list[int]:
    return record["input_ids"][:n_prompt]


def tests_dir() -> "Path":
    from pathlib import Path

    return Path(__file__).parent


PROMPTS = None  # filled lazily; two short prompts keep CPU runtime sane


def _get_prompts():
    global PROMPTS
    if PROMPTS is None:
        import pyarrow.parquet as pq

        t = pq.read_table(tests_dir().parent / "frozen" / "xlam_eval.parquet")
        PROMPTS = [
            t.column("input_ids")[i].as_py()[: t.column("n_prompt_tokens")[i].as_py()]
            for i in (0, 1)
        ]
    return PROMPTS


class TestSelfConsistency:
    def test_draft_equals_target_all_accepted(self, target_model):
        eos_id = get_eos_id()
        verifier = HFVerifier(target_model)
        proposer = HFDraftProposer(target_model, eos_id)
        for prompt in _get_prompts():
            res = generate_turn(
                proposer, verifier, prompt, eos_id, k=5,
                max_new_tokens=32, query_id="self",
            )
            for e in res.events:
                assert all(e["accept_mask"]), (
                    f"rejection with draft == target: step {e['step']}"
                )
                # every non-terminal round carries the bonus; a None
                # correction is only legal on the terminal event (draft
                # proposed EOS, or the budget stop)
                if not e["eos"]:
                    assert e["correction_token"] is not None
            # α=1, τ=k across the run: assert via the analyzer. τ is the
            # per-round mean, and the FINAL round is budget-capped (its
            # proposal is shorter than k), so τ = k exactly on every
            # non-terminal round and slightly below k overall — assert
            # per-round instead of the mean.
            from src.analysis.eval_acceptance import flat_metrics

            m = flat_metrics(res.events, k_max=5)
            assert m["alpha"] == 1.0
            for e in res.events[:-1]:  # every non-terminal round: all k accepted
                assert sum(e["accept_mask"]) == 5
            # terminal round: budget-capped proposal, all accepted
            assert sum(res.events[-1]["accept_mask"]) == len(
                res.events[-1]["draft_tokens"]
            )
            assert len(res.output_ids) == 32


class TestExactness:
    def test_loop_equals_plain_greedy(self, target_model, draft_noncoder):
        """The §4.4 exactness claim at loop level: output ids identical
        to plain autoregressive greedy decoding of the target."""
        import torch
        import transformers

        eos_id = get_eos_id()
        verifier = HFVerifier(target_model)
        proposer = HFDraftProposer(draft_noncoder, eos_id)
        tok = transformers.AutoTokenizer.from_pretrained(TINY)
        for prompt in _get_prompts():
            res = generate_turn(
                proposer, verifier, prompt, eos_id, k=5,
                max_new_tokens=48, query_id="exact",
            )
            # rejections must actually occur (else the crop path is dead
            # code in this test)
            assert any(
                not all(e["accept_mask"]) for e in res.events
            ), "draft agreed everywhere — exactness test not exercising rejection"
            with torch.no_grad():
                out = target_model.generate(
                    torch.tensor([prompt]),
                    max_new_tokens=48,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                )
            plain = out[0].tolist()[len(prompt):]
            assert res.output_ids[: len(plain)] == plain
