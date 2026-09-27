"""Real-model checks for the Stage-1 generation harness (plan.md §6.7).

Layer 2: runs only when torch + the tiny Qwen models are available
(skipped otherwise — the Layer-1 suite in test_gen_stage1.py is the
default gate), mirroring tests/test_spec_realmodels.py's conventions.

HFGenEngine (fp32 CPU) plays the generation path end-to-end on real
frozen Stage-1 prompts with Qwen2.5-Coder-0.5B-Instruct as a stand-in
target (same tokenizer family as the 14B target, plan §1.2):

  - the JSONL that comes out loads through build_distill_data.load_gens
    and assembles into a KD dataset whose labels/regions/logprobs are
    internally consistent (the exact artifacts the GPU day will produce
    with the 14B target, one tokenizer family over);
  - E2: a "stop" generation's output_ids end with the tokenizer's EOS
    and its logprobs row covers that position;
  - greedy generation is what HF's plain generate() produces — the
    engine's token path adds nothing (the logprob path is separate).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from src.data_prep.build_distill_data import (  # noqa: E402
    assemble,
    load_gens,
)
from src.serving.gen_stage1 import (  # noqa: E402
    HFGenEngine,
    normalize_output,
    write_generations,
)

STANDIN = "Qwen/Qwen2.5-Coder-0.5B-Instruct"  # cached for test_spec_realmodels

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
def engine():
    _maybe_skip(STANDIN)
    return HFGenEngine(STANDIN, logprobs=5, dtype=None, device="auto")


@pytest.fixture(scope="module")
def prompts():
    from src.data_prep.build_distill_data import load_stage1_prompts

    return load_stage1_prompts(limit=2)


@pytest.fixture(scope="module")
def gen_jsonl(tmp_path_factory, engine, prompts):
    outs = [normalize_output(o, engine.eos_id)
            for o in engine.generate(
                [p["prompt_ids"] for p in prompts],
                {"max_tokens": 48, "temperature": 0.0, "seed": 1234},
            )]
    p = tmp_path_factory.mktemp("stage1") / "gen.jsonl"
    write_generations(str(p), prompts, outs)
    return p


class TestEndToEnd:
    def test_stop_generation_ends_with_eos_covered_by_logprobs(
        self, engine, gen_jsonl
    ):
        gens = load_gens(str(gen_jsonl))
        assert gens, "no generations"
        for g in gens:
            assert len(g["logprobs"]) == len(g["output_ids"])
            if g["finish_reason"] == "stop":
                assert g["output_ids"][-1] == engine.eos_id  # E2
                assert g["logprobs"][-1]  # EOS position has entries

    def test_greedy_tokens_match_plain_hf_generate(self, engine, prompts):
        import torch

        for p, expected in zip(
            prompts,
            engine.generate(
                [p["prompt_ids"] for p in prompts],
                {"max_tokens": 16, "temperature": 0.0, "seed": 1234},
            ),
        ):
            inp = torch.tensor([p["prompt_ids"]], dtype=torch.long,
                               device=engine.model.device)
            with torch.no_grad():
                ref = engine.model.generate(
                    inp, max_new_tokens=16, do_sample=False,
                    temperature=None,
                )
            ref_ids = ref[0][inp.shape[1]:].tolist()
            # same tokens up to the EOS the harness appends on "stop"
            n = len(expected["token_ids"])
            stop = expected["finish_reason"] == "stop"
            assert expected["token_ids"][: n - (1 if stop else 0)] == ref_ids

    def test_assembles_into_consistent_kd_dataset(self, gen_jsonl):
        gens = load_gens(str(gen_jsonl))
        records, stats = assemble(gens, limit=2)
        assert stats["n_stage1"] == 2
        assert stats["roundtrip_exact_all"]
        for rec in records:
            n = rec["n_prompt_tokens"]
            assert rec["labels"][:n] == [-100] * n
            assert rec["labels"][n:] == rec["gen_ids"]
            # every kept record has valid calls per its own schemas (G4)
            assert rec["n_tool_calls"] >= 1
            for row_ids, row_vals in zip(
                rec["gen_logprob_token_ids"], rec["gen_logprob_values"]
            ):
                if row_ids is None:
                    assert row_vals is None
                else:
                    assert len(row_ids) == len(row_vals)
                    # G2: sorted desc — the first entry is the argmax, which
                    # for greedy IS the emitted token
                    assert row_vals == sorted(row_vals, reverse=True)
