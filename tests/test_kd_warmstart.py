"""Golden tests for the Stage-1 KD warm-start (plan.md §6.7 second half).

Layer 1 (torch-free): packing + stats against hand-computed cases, the
K1/K2/K3 guards, and the CLI surface. The loss math itself is torch-bound;
its hand-computed cases live here too, gated on torch presence (the loss
module imports torch lazily, so the packing layer stays importable
torch-free — same two-tier pattern as test_gen_stage1.py vs
test_gen_stage1_realmodels.py).

The real-model end-to-end (cached Coder-0.5B, a few steps, the K4
packed-vs-solo property on real weights) is
tests/test_kd_warmstart_realmodels.py.
"""

from __future__ import annotations

import json

import pytest

from src.training.kd_warmstart import (
    DEFAULT_SFT_WEIGHT,
    KD_TOPK_CAP,
    Pack,
    batch_tensors,
    kd_row_losses,
    pack_records,
    pack_stats,
    topk_pad,
)


def rec(qid, ids, labels, t_ids=None, t_vals=None, n_prompt=None):
    """A minimal KD record — same fields the assembled dataset carries."""
    n_labels = sum(1 for l in labels if l != -100)
    if n_prompt is None:
        first = next((i for i, l in enumerate(labels) if l != -100), None)
        n_prompt = first if first is not None else len(ids)
    return {
        "query_id": qid,
        "input_ids": ids,
        "labels": labels,
        "n_prompt_tokens": n_prompt,
        "gen_logprob_token_ids": t_ids if t_ids is not None else [[9]] * n_labels,
        "gen_logprob_values": t_vals if t_vals is not None
        else [[-0.5]] * n_labels,
    }


class TestPacking:
    def test_single_record_becomes_one_pack(self):
        r = rec(1, [10, 11, 12, 13], [-100, -100, -100, 13])
        packs, skipped = pack_records([r], max_len=16)
        assert skipped == 0
        assert len(packs) == 1
        p = packs[0]
        assert p.input_ids == [10, 11, 12, 13]
        assert p.position_ids == [0, 1, 2, 3]
        assert p.loss_positions == [3]
        assert p.record_ids == [1]

    def test_two_records_share_a_pack(self):
        r1 = rec(1, [10, 11, 12, 13, 14], [-100] * 3 + [13, 14],
                 t_ids=[[13, 99], [14, 98]], t_vals=[[-0.1, -2.0], [-0.2, -1.5]])
        r2 = rec(2, [20, 21, 22, 23], [-100] * 3 + [23],
                 t_ids=[[23, 97]], t_vals=[[-0.3, -1.7]])
        packs, skipped = pack_records([r1, r2], max_len=16)
        assert skipped == 0 and len(packs) == 1
        p = packs[0]
        assert p.input_ids == [10, 11, 12, 13, 14, 20, 21, 22, 23]
        # K4: positions restart per record
        assert p.position_ids == [0, 1, 2, 3, 4, 0, 1, 2, 3]
        # loss positions are in PACKED coordinates, strictly increasing
        assert p.loss_positions == [3, 4, 8]
        assert p.teacher_token_ids == [[13, 99], [14, 98], [23, 97]]
        assert p.record_ids == [1, 2]

    def test_records_never_split_across_packs(self):
        # r1 needs 5 slots; with max_len=7 the 4-token r2 cannot fit after
        # it -> separate packs, never a split
        r1 = rec(1, [1, 2, 3, 4, 5], [-100, -100, -100, 4, 5])
        r2 = rec(2, [6, 7, 8, 9], [-100, -100, -100, 9])
        packs, skipped = pack_records([r1, r2], max_len=7)
        assert skipped == 0
        assert [len(p) for p in packs] == [5, 4]
        assert packs[0].record_ids == [1] and packs[1].record_ids == [2]

    def test_too_long_records_skipped_not_split(self):
        r = rec(3, list(range(10)), [-100] * 10)
        packs, skipped = pack_records([r], max_len=8)
        assert packs == [] and skipped == 1

    def test_label_at_position_zero_refused(self):
        # K3: the label's predictor row would be the previous record's tail
        # (or nothing) — a hard error, not a silent wrong loss
        bad = rec(4, [5, 6], [5, -100], t_ids=[[5]], t_vals=[[-0.1]])
        with pytest.raises(AssertionError, match="K3"):
            pack_records([bad], max_len=8)

    def test_teacher_row_count_mismatch_refused(self):
        # G2/G3 contract: one teacher row per label; a mismatch is a data
        # bug the trainer must not paper over
        bad = rec(5, [1, 2, 3], [-100, -100, 3], t_ids=[[3], [3]],
                  t_vals=[[-0.1], [-0.2]])
        with pytest.raises(AssertionError, match="teacher rows"):
            pack_records([bad], max_len=8)

    def test_frozen_order_preserved(self):
        # K6: packing is a pure function of record order — same input,
        # same packs, every time
        rs = [rec(i, list(range(100 + i, 104 + i)), [-100] * 3 + [103 + i])
              for i in range(5)]
        a, _ = pack_records(rs, max_len=12)
        b, _ = pack_records(rs, max_len=12)
        assert a == b

    def test_pack_stats(self):
        rs = [rec(1, [1, 2, 3, 4], [-100] * 3 + [4]),
              rec(2, [5, 6, 7, 8], [-100] * 3 + [8])]
        packs, skipped = pack_records(rs, max_len=32)
        st = pack_stats(packs, n_records=2, n_skipped=0)
        assert st["n_records"] == 2
        assert st["n_packs"] == 1
        assert st["n_label_tokens"] == 2
        assert st["label_share"] == 0.25
        assert st["pack_len_max"] == 8

    def test_topk_pad_caps_at_plan_top20(self):
        r = rec(1, [1, 2, 3], [-100, -100, 3],
               t_ids=[list(range(30))], t_vals=[[-0.1] * 30])
        packs, _ = pack_records([r], max_len=8)
        assert topk_pad(packs) == KD_TOPK_CAP
        r2 = rec(2, [1, 2, 3], [-100, -100, 3],
                 t_ids=[[1, 2]], t_vals=[[-0.1, -0.2]])
        packs2, _ = pack_records([r2], max_len=8)
        assert topk_pad(packs2) == 2

    def test_empty_teacher_rows_mean_ce_fallback(self):
        # K2: a None row (or an empty list) -> row_valid False -> CE only
        r = rec(1, [1, 2, 3], [-100, -100, 3], t_ids=[None], t_vals=[None])
        packs, _ = pack_records([r], max_len=8)
        t_ids, t_lps, valid, labels, keep = batch_tensors(packs[0], 4, "cpu")
        import torch  # noqa: F401 — needed for the tensor returns below

        assert valid.tolist() == [False]
        assert keep.tolist() == [1]  # label at packed pos 2 -> row 2-1
        assert labels.tolist() == [3]


class TestLossMath:
    """Hand-computed cases for the (1-w)*KL + w*CE form, K1/K2."""

    @classmethod
    def setup_class(cls):
        pytest.importorskip("torch")
        import torch

        cls.torch = torch

    def _row(self, student, t_ids, t_lps, valid, label, real, w):
        t = self.torch
        logits = t.tensor([student])
        rows = kd_row_losses(
            logits, t.tensor([t_ids]), t.tensor([t_lps]),
            t.tensor([valid]), t.tensor([label]), real, sft_weight=w,
        )
        return float(rows[0].item())

    def test_exact_match_is_zero_kl(self):
        # student == renormalized teacher on the top-k support -> KL ~ 0
        t = self.torch
        V, REAL = 100, 90
        s = t.full((V,), -20.0)
        s[3] = t.log(t.tensor(0.6))
        s[7] = t.log(t.tensor(0.3))
        # renormalized over {3,7}: p = (2/3, 1/3)
        loss = self._row(s.tolist(), [3, 7],
                         [t.log(t.tensor(0.6)).item(),
                          t.log(t.tensor(0.3)).item()],
                         True, 3, REAL, w=0.0)
        assert abs(loss) < 1e-5

    def test_ce_only_when_no_teacher(self):
        # K2 fallback: row_valid False -> plain CE
        t = self.torch
        V, REAL = 100, 90
        s = t.zeros(V)
        s[42] = 5.0
        label = 42
        ce_expected = -t.log(t.tensor(1.0))  # softmax puts ~1 on 42
        loss = self._row(s.tolist(), [0, 0],
                         [float("-inf"), float("-inf")],
                         False, label, REAL, w=0.0)
        assert loss == pytest.approx(ce_expected, abs=1e-4)

    def test_sft_weight_mixes_kl_and_ce(self):
        # w=1 -> pure CE on the teacher's argmax (first stored row entry,
        # G2 sort order); w between -> the linear mix
        t = self.torch
        V, REAL = 100, 90
        s = t.zeros(V)
        s[3] = 2.0
        ce = self._row(s.tolist(), [3, 7], [-0.1, -2.0], True, 3, REAL, w=1.0)
        kl0 = self._row(s.tolist(), [3, 7], [-0.1, -2.0], True, 3, REAL, w=0.0)
        mix = self._row(s.tolist(), [3, 7], [-0.1, -2.0], True, 3, REAL, w=0.3)
        assert mix == pytest.approx(0.7 * kl0 + 0.3 * ce, rel=1e-4)

    def test_vocab_slice_excludes_padding_mass(self):
        # K1: student mass on ids >= real_vocab must not be reachable —
        # the loss with pad mass vs identical logits without it must be
        # EQUAL (the slice renormalizes it away), which is the exact
        # property "padding contributes neither mass nor gradients"
        t = self.torch
        REAL = 90
        s = t.full((100,), -1.0)
        s[3] = 2.0
        s[95] = 50.0  # huge logit in the PADDING region
        with_pad = self._row(s.tolist(), [3, 7], [-0.1, -2.0], True, 3, REAL, w=0.5)
        s2 = s.clone()
        s2[95] = -1.0
        without_pad = self._row(s2.tolist(), [3, 7], [-0.1, -2.0], True, 3, REAL, w=0.5)
        assert with_pad == pytest.approx(without_pad, rel=1e-4)

    def test_padding_id_in_teacher_row_refused(self):
        t = self.torch
        s = t.zeros(100)
        with pytest.raises(AssertionError, match="padding id"):
            kd_row_losses(
                t.tensor([s]), t.tensor([[95, 7]]),
                t.tensor([[-0.1, -1.0]]), t.tensor([True]),
                t.tensor([3]), 90, sft_weight=0.1,
            )

    def test_label_outside_real_vocab_refused(self):
        t = self.torch
        s = t.zeros(100)
        with pytest.raises(AssertionError, match="label"):
            kd_row_losses(
                t.tensor([s]), t.tensor([[3, 7]]),
                t.tensor([[-0.1, -1.0]]), t.tensor([True]),
                t.tensor([95]), 90, sft_weight=0.1,
            )

    def test_batch_with_mixed_valid_rows(self):
        # one row with a teacher, one without; both must be finite and the
        # valid row must carry the KL term (finite everywhere, no NaN from
        # 0 * -inf on padded columns)
        t = self.torch
        t.manual_seed(0)
        logits = t.randn(2, 100)
        rows = kd_row_losses(
            logits, t.tensor([[3, 7], [0, 0]]),
            t.tensor([[-0.2, -1.0], [float("-inf"), float("-inf")]]),
            t.tensor([True, False]), t.tensor([3, 10]), 90,
            sft_weight=DEFAULT_SFT_WEIGHT,
        )
        assert t.isfinite(rows).all()
        assert rows.shape == (2,)


class TestCLI:
    def test_module_has_pinned_defaults(self):
        from src.training import kd_warmstart as k

        assert k.DEFAULT_DRAFT == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        assert k.DEFAULT_SFT_WEIGHT == 0.1
        assert k.KD_TOPK_CAP == 20

    def test_trainer_ctor_accepts_all_knobs(self):
        from src.training.kd_warmstart import KDWarmstartTrainer

        tr = KDWarmstartTrainer(
            draft_id="d", data_dir="x", out_dir="y", max_len=512,
            micro_bs=2, grad_accum=3, lr=1e-5, epochs=1,
            sft_weight=0.2, grad_ckpt=False, dtype_name="float32",
            device="cpu", limit=4, seed=7, save_vocab_padded=True,
        )
        assert tr.max_len == 512 and tr.grad_accum == 3

    def test_real_vocab_guard_rejects_small_tokenizer(self):
        from src.training.kd_warmstart import real_vocab_size

        class FakeTok:
            def __len__(self):
                return 100

        with pytest.raises(AssertionError, match="wrong tokenizer family"):
            real_vocab_size(FakeTok())
