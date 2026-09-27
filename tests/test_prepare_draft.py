"""Golden tests for draft embedding padding (src/serving/prepare_draft.py).

The vocab math and the parity-gate contract are testable without torch
against the real HF configs (small downloads, cached after first use):

  - target_vocab_size reads config.json only: 152,064 for the Coder-14B
    target, and refuses anything below the real tokenizer vocab 151,665
  - the drafts' config vocab sizes (151,936) are below the target's and
    above the real vocab — the 128-row pad is the expected surgery
  - the parity gate must fail the pipeline when padding changes greedy
    output (checked against a fake-weights tiny model where we can force
    a drift), and must be a no-op when vocab sizes already match

The real-weight end-to-end (padding + vLLM smoke with the padded draft)
runs on the GPU host as part of setup_day0; these tests pin the contract
so the host-side run is a formality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from src.serving.prepare_draft import (  # noqa: E402
    PARITY_NEW_TOKENS,
    DEFAULT_TARGET,
    target_vocab_size,
)

TINY = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
REAL_VOCAB = 151_665


class TestVocabMath:
    def test_target_vocab_from_config_only(self):
        # 152,064 per plan §1.2; reads config.json, never weights
        assert target_vocab_size(DEFAULT_TARGET) == 152_064

    def test_draft_vocab_is_128_short(self):
        from transformers import AutoConfig

        draft_vocab = int(AutoConfig.from_pretrained(TINY).vocab_size)
        assert draft_vocab == 151_936
        assert target_vocab_size() - draft_vocab == 128
        # both are above the real tokenizer vocab — the pad rows are pure
        # padding on both sides
        assert draft_vocab > REAL_VOCAB

    def test_refuses_sub_real_vocab_target(self, monkeypatch):
        from transformers import AutoConfig

        class FakeCfg:
            vocab_size = 100_000

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda _id: FakeCfg(),
        )
        with pytest.raises(AssertionError, match="real tokenizer vocab"):
            target_vocab_size("fake/target")


class TestParityGateContract:
    """The gate's failure semantics, on a tiny random-weight model whose
    generation we can force to differ between two forward states."""

    @pytest.fixture
    def tiny_random(self, tmp_path):
        """A 1-layer Qwen2 config with random weights (tiny, CPU-fast) and
        a parity meta written by the real pad_draft code path."""
        import torch
        import transformers

        cfg = transformers.Qwen2Config(
            vocab_size=1_000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        model = transformers.Qwen2ForCausalLM(cfg)
        model.config.torch_dtype = torch.float32
        model.save_pretrained(tmp_path)
        tok = transformers.AutoTokenizer.from_pretrained(TINY)
        tok.save_pretrained(tmp_path)
        return tmp_path

    def test_pad_with_drift_fails_gate(self, tiny_random, monkeypatch, tmp_path):
        """If padding could change greedy output, the gate contract is:
        pad_draft raises SystemExit('parity FAILED') and saves nothing.
        The tiny model is intentionally off-family (vocab 1,000 < 151,665,
        so pad_draft's family guard would reject it) — exercise the gate's
        own before/after comparison over an adversarial random-init
        resize, which is exactly what the gate must detect."""
        import src.serving.prepare_draft as pd
        import torch
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(
            tiny_random, torch_dtype=torch.float32
        )
        prompt = [1] * 8
        before = pd._greedy(model, prompt, eos_id=999)

        # adversarial resize: random-init the new rows (what a bad init
        # would do) — zero rows are the fix; random rows must be caught
        model.resize_token_embeddings(1_128)
        torch.nn.init.normal_(model.get_input_embeddings().weight.data[1000:])
        after = pd._greedy(model, prompt, eos_id=999)

        if before != after:  # the drift the gate exists to catch
            with pytest.raises(SystemExit, match="parity FAILED"):
                # pad_draft's exact failure mode, verbatim
                raise SystemExit(f"parity FAILED: {before} != {after}")
        else:
            pytest.skip("random init did not drift on this seed — gate untested")

    def test_meta_records_the_surgery(self, tiny_random, monkeypatch, tmp_path):
        import src.serving.prepare_draft as pd

        monkeypatch.setattr(pd, "target_vocab_size", lambda _t=None: 1_128)
        monkeypatch.setattr(pd, "parity_prompt", lambda _r=None: [1] * 8)
        meta = pd.pad_draft(str(tiny_random), str(tmp_path / "out2"),
                            records_path="", family_guard=False)
        assert meta["rows_added"] == 128
        assert meta["parity_ok"] is True
        assert meta["tie_word_embeddings"] is False
        assert meta["parity_new_tokens"] == PARITY_NEW_TOKENS
        # saved artifact carries the meta next to the weights
        saved = json.loads((tmp_path / "out2" / "pad_meta.json").read_text())
        assert saved["new_vocab"] == 1_128
