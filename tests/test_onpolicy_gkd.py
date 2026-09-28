"""Golden tests for Stage-2 on-policy GKD (src/training/onpolicy_gkd.py).

Torch-dependent pieces (the losses) run under importorskip; the context
stream and mixing logic are torch-free. The tiny-model end-to-end lives
in tests/test_onpolicy_gkd_realmodels.py.
"""

from __future__ import annotations

import pytest

import src.training.onpolicy_gkd as g


class TestGKDLosses:
    """G3: divergences over the real vocab + CE anchor on target argmax."""

    def test_zero_divergence_for_identical_distributions(self):
        torch = pytest.importorskip("torch")
        x = torch.randn(4, 600)
        l = g.gkd_losses(x.clone().requires_grad_(True), x, 500,
                         "reverse_kl", sft_weight=0.0)
        assert float(l.abs().max()) < 1e-4

    def test_vocab_sliced_before_normalization(self):
        # a huge logit OUTSIDE the real vocab must not affect the loss:
        # padding carries neither mass nor gradient (K1)
        torch = pytest.importorskip("torch")
        base = torch.randn(3, 600)
        t = base.clone()
        s1 = base.clone(); s2 = base.clone()
        s2[0, 550] = 1e4  # id 550 >= real_vocab 500: pure padding
        l1 = g.gkd_losses(s1.requires_grad_(True), t, 500, "forward_kl", 0.0)
        l2 = g.gkd_losses(s2.requires_grad_(True), t, 500, "forward_kl", 0.0)
        assert torch.allclose(l1, l2)

    def test_reverse_kl_finite_and_grad_finite(self):
        torch = pytest.importorskip("torch")
        s = torch.randn(5, 600, requires_grad=True)
        t = torch.randn(5, 600)
        for div in ("reverse_kl", "forward_kl", "jsd"):
            l = g.gkd_losses(s, t, 500, div, 0.1)
            assert l.shape == (5,)
            assert bool(torch.isfinite(l).all())
            l.sum().backward(retain_graph=True)
            assert bool(torch.isfinite(s.grad).all())
            s.grad = None

    def test_unknown_divergence_rejected(self):
        torch = pytest.importorskip("torch")
        s = torch.randn(2, 10)
        with pytest.raises(ValueError, match="unknown divergence"):
            g.gkd_losses(s, s, 10, "tvd", 0.0)

    def test_sft_weight_blends_divergence_and_ce(self):
        # w=0 -> pure divergence; w=1 -> pure CE on the target's argmax
        torch = pytest.importorskip("torch")
        s = torch.randn(2, 50, requires_grad=True)
        t = torch.randn(2, 50)
        l0 = g.gkd_losses(s, t, 50, "forward_kl", 0.0)
        l1 = g.gkd_losses(s, t, 50, "forward_kl", 1.0)
        ce = torch.nn.functional.cross_entropy(
            s[:, :50], t[:, :50].argmax(-1), reduction="none")
        assert torch.allclose(l1, ce, atol=1e-5)
        assert not torch.allclose(l0, ce, atol=1e-5)


class TestRealVocab:
    def test_len_tokenizer_not_base_vocab(self):
        # 151,665 (len(tok)), NOT tok.vocab_size's 151,643 (K1 note)
        v = g.real_vocab_size()
        assert v == 151_665


class TestContextStream:
    """G5: mixed pools, seeded order, max-ctx left-truncation."""

    def test_clip_to_max_ctx(self, tmp_path, monkeypatch):
        # a 20-token context clipped to 10 keeps the LAST 10
        ids = list(range(100, 120))
        got = g.load_contexts  # noqa: F841  (indirect: _clip is nested)
        # exercise via a minimal pool: monkeypatch load_from_disk
        pytest.importorskip("datasets")

        import datasets as hfds

        monkeypatch.setattr(
            "src.training.onpolicy_gkd.load_from_disk" if hasattr(g, "load_from_disk")
            else "datasets.load_from_disk",
            lambda d: _FakeDS(ids),
        )
        # the module imports load_from_disk inside the function from
        # datasets — patch it there
        monkeypatch.setattr(hfds, "load_from_disk", lambda d: _FakeDS([ids]),
                            raising=False)

        ctxs = g.load_contexts(tmp_path, tmp_path, tb_frac=0.0,
                               max_ctx=10, seed=0)
        assert len(ctxs) == 1
        assert ctxs[0]["input_ids"] == ids[-10:]

    def test_mixed_sources_present(self, tmp_path, monkeypatch):
        pytest.importorskip("datasets")
        import datasets as hfds

        xlam_ids = [list(range(200, 230))]
        tb_ids = [list(range(300, 340))]
        dirs = {"x": _FakeDS(xlam_ids), "t": _FakeDS(tb_ids)}

        def fake_load(d):
            return dirs["x" if str(d).endswith("xlam") else "t"]

        monkeypatch.setattr(hfds, "load_from_disk", fake_load, raising=False)
        ctxs = g.load_contexts("dirx_xlam", "dirt", tb_frac=0.5,
                                max_ctx=4096, seed=0)
        sources = {c["source"] for c in ctxs}
        assert sources == {"xlam", "tb"}


class _FakeDS:
    """Minimal load_from_disk stand-in: iterable of record dicts. Rows are
    either bare id lists (wrapped as {"input_ids": rows}) or full record
    dicts (passed through untouched)."""

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        for r in self.rows:
            if isinstance(r, dict):
                yield r
            else:
                yield {"input_ids": r}


class TestPrefixBoundaries:
    """G5's boundary semantics: a CONTEXT is the prompt prefix ending at
    a sampling boundary — never the full conversation (the draft must not
    sample a continuation of a finished conversation's gold assistant
    turns)."""

    @pytest.fixture
    def patched_loader(self, monkeypatch):
        pytest.importorskip("datasets")
        import datasets as hfds

        def _patch(rows_by_dir):
            def fake_load(d):
                key = "x" if "xlam" in str(d) else "t"
                return _FakeDS(rows_by_dir[key])

            monkeypatch.setattr(hfds, "load_from_disk", fake_load, raising=False)
            return fake_load

        return _patch

    def test_xlam_context_is_prompt_prefix_not_full_record(
        self, patched_loader
    ):
        # a full record (gold assistant turn included) yields ONLY the
        # prompt part as its context, ending at n_prompt_tokens
        ids = list(range(100, 120))
        rec = {"input_ids": ids, "n_prompt_tokens": 8, "labels": [-100] * 8
               + list(range(108, 120))}
        patched_loader({"x": [rec], "t": []})
        ctxs = g.load_contexts("d_xlam", "d_tb", tb_frac=0.0,
                               max_ctx=4096, seed=0)
        assert len(ctxs) == 1
        assert ctxs[0]["input_ids"] == ids[:8]  # NOT the full 20

    def test_tb_yields_one_context_per_assistant_turn(self, patched_loader):
        # 2 labeled assistant spans -> 2 contexts; the turn-1 context
        # contains the turn-0 tool-response region (post-observation)
        ids = list(range(200, 260))
        labels = [-100] * 40
        labels[10] = 210; labels[11] = 211          # assistant span 1
        labels[30] = 230; labels[31] = 231          # assistant span 2
        rec = {"input_ids": ids, "labels": labels}
        patched_loader({"x": [], "t": [rec]})
        ctxs = g.load_contexts("d_xlam", "d_tb", tb_frac=1.0,
                               max_ctx=4096, seed=0)
        assert len(ctxs) == 2
        starts = sorted(len(c["input_ids"]) for c in ctxs)
        assert starts == [10, 30]
        # the later context includes the region between the spans
        long_ctx = max(ctxs, key=lambda c: len(c["input_ids"]))
        assert 220 in long_ctx["input_ids"]

    def test_bare_record_whole_ids_is_context(self, patched_loader):
        # a labels-less, n_prompt_tokens-less record IS a prompt
        ids = list(range(300, 340))
        patched_loader({"x": [ids], "t": []})
        ctxs = g.load_contexts("d_xlam", "d_tb", tb_frac=0.0,
                               max_ctx=4096, seed=0)
        assert ctxs[0]["input_ids"] == ids


class TestMixPlan:
    """G4's deterministic Stage-1 slot plan."""

    def test_zero_mix_all_onpolicy(self):
        t = g.OnPolicyGKDTrainer(
            draft_dir="d", target_id="t", out_dir="o", xlam_ctx_dir="x",
            tb_ctx_dir="b", mix_sft_frac=0.0,
        )
        assert t._mix_plan(5) == [False] * 5

    def test_quarter_mix_every_fourth_slot(self):
        t = g.OnPolicyGKDTrainer(
            draft_dir="d", target_id="t", out_dir="o", xlam_ctx_dir="x",
            tb_ctx_dir="b", mix_sft_frac=0.25,
        )
        plan = t._mix_plan(8)
        assert plan == [False, False, False, True, False, False, False,
                        True]
        assert sum(plan) == 2  # realized share 1/K = 1/4

    def test_high_mix_clamped_to_every_second(self):
        # f=0.4 -> K = max(2, round(2.5)) = 2 via banker's rounding
        t = g.OnPolicyGKDTrainer(
            draft_dir="d", target_id="t", out_dir="o", xlam_ctx_dir="x",
            tb_ctx_dir="b", mix_sft_frac=0.4,
        )
        assert t._mix_plan(4) == [False, True, False, True]

    def test_bad_mix_refused(self):
        t = g.OnPolicyGKDTrainer(
            draft_dir="d", target_id="t", out_dir="o", xlam_ctx_dir="x",
            tb_ctx_dir="b", mix_sft_frac=0.5,
        )
        with pytest.raises(ValueError):
            t._mix_plan(4)
