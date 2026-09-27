"""Golden tests for the instrumented speculative-decoding loop (§6.5).

Layer 1 (this file) is torch-free: scripted fake adapters pin every
convention C1–C7 against hand-computed events, torture the sync/crop
bookkeeping, and round-trip loop output through eval_acceptance (the
consumer that defined the schema). Real-model §4.4 checks (self-
consistency and exactness) live in test_spec_realmodels.py, skipped
unless torch + the tiny models are available.

The scripted verifier below mirrors the HFVerifier's two predict paths
(held-argmax vs post-crop realign) so the fakes exercise the same
arithmetic the HF adapters will run.
"""

from __future__ import annotations

import json

import pytest

from src.analysis.eval_acceptance import flat_metrics, full_report
from src.serving.instrumented_spec import (
    STEP_BUDGET,
    STEP_FULL_BONUS,
    STEP_FULL_EOS,
    STEP_REJECT,
    generate_record,
    generate_turn,
    verify_round,
    run_rounds,
    LoopResult,
)
from src.serving.proposers import NgramProposer

EOS = 151645  # <|im_end|> (the CLI reads it from the tokenizer; pinned here)
PAD = 151643


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedVerifier:
    """Verifier driven by an explicit per-call prediction script.

    `script` is a list of prediction lists, consumed one per predict()
    call (a full list per round, in round order). Each entry must have
    len(proposal)+1 entries. Also self-checks the sync contract: the
    canonical sequence after each round must equal prompt + all emitted
    tokens so far (checked in sync()), and predict() must never be
    called with the adapter out of sync (no stale speculative tail).
    """

    def __init__(self, script: list[list[int]]):
        self.script = list(script)
        self.i = 0
        self.canonical: list[int] = []
        self.n_predict_calls = 0
        self.n_realigns = 0

    def reset(self, tokens: list[int]) -> None:
        self.canonical = list(tokens)
        # a reset mid-script means the loop restarted a turn without
        # finishing it — only legal at the very start
        assert self.i in (0, len(self.script)), "reset mid-script"

    def predict(self, tokens: list[int]) -> list[int]:
        self.n_predict_calls += 1
        preds = self.script[self.i]
        self.i += 1
        assert len(preds) == len(tokens) + 1, (
            f"script entry {self.i-1} has {len(preds)} preds, "
            f"expected {len(tokens)+1}"
        )
        return list(preds)

    def sync(self, pre_len: int, emitted: list[int]) -> None:
        # canonical must be exactly pre_len long before this round's emit
        assert len(self.canonical) == pre_len, (
            f"desync: canonical len {len(self.canonical)} != pre_len {pre_len}"
        )
        self.canonical.extend(emitted)


class TableVerifier:
    """Deterministic 'target' defined by a lookup of accepted tokens.

    T_ACCEPT: the set of tokens the target always accepts (its greedy
    choice at any position); everything else is rejected with correction
    token T_CORR. The bonus after a full accept is T_NEXT. Used for
    long/torture runs where scripting every round by hand is impractical.
    """

    T_ACCEPT = {10, 11, 12, 13, 14, 15}
    T_CORR = 99
    T_NEXT = 20

    def __init__(self):
        self.canonical: list[int] = []

    def reset(self, tokens: list[int]) -> None:
        self.canonical = list(tokens)

    def predict(self, tokens: list[int]) -> list[int]:
        preds = []
        for t in tokens:
            preds.append(t if t in self.T_ACCEPT else self.T_CORR)
        preds.append(self.T_NEXT)  # bonus
        return preds

    def sync(self, pre_len: int, emitted: list[int]) -> None:
        # mirrors the real adapters' contract: crop to the canonical
        # prefix length, then extend with what actually got emitted
        # (round sync: emitted follows the prefix; turn-transition
        # sync: emitted REPLACES previously generated tokens)
        self.canonical = self.canonical[:pre_len]
        self.canonical.extend(emitted)


class ScriptedProposer:
    """Proposer driven by an explicit per-call proposal script."""

    def __init__(self, script: list[list[int]]):
        self.script = list(script)
        self.i = 0
        self.canonical: list[int] = []
        self.n_propose_calls = 0

    def reset(self, tokens: list[int]) -> None:
        self.canonical = list(tokens)

    def propose(self, k: int) -> list[int]:
        self.n_propose_calls += 1
        proposal = self.script[self.i]
        self.i += 1
        assert 1 <= len(proposal) <= k, (
            f"script entry {self.i-1} len {len(proposal)} violates 1..{k}"
        )
        return list(proposal)

    def sync(self, pre_len: int, emitted: list[int]) -> None:
        assert len(self.canonical) == pre_len
        self.canonical.extend(emitted)


class ChainProposer:
    """Deterministic proposer: proposes tokens cycling through a chain
    sequence; stops early (length < k) when it would propose EOS."""

    def __init__(self, chain: list[int], eos_id: int = EOS):
        self.chain = list(chain)
        self.eos_id = eos_id
        self.canonical: list[int] = []
        self.pos = 0

    def reset(self, tokens: list[int]) -> None:
        self.canonical = list(tokens)
        self.pos = 0

    def propose(self, k: int) -> list[int]:
        out = []
        while len(out) < k and self.pos < len(self.chain):
            t = self.chain[self.pos]
            out.append(t)
            self.pos += 1
            if t == self.eos_id:
                break
        if not out:  # chain exhausted: pad forever (contract: non-empty)
            out = [PAD]
        return out

    def sync(self, pre_len: int, emitted: list[int]) -> None:
        # round syncs: canonical == pre_len (intra-turn). Turn-transition
        # syncs: canonical can be longer (the previous turn's generated
        # tokens, replaced by the truth span) — so crop, then extend.
        self.canonical = self.canonical[:pre_len]
        self.canonical.extend(emitted)
        self.pos = pre_len  # next proposals continue after the canon prefix


# ---------------------------------------------------------------------------
# verify_round (pure function)
# ---------------------------------------------------------------------------


class TestVerifyRound:
    def test_c1_rejection_first_position(self):
        mask, corr, stype, eos = verify_round([10, 11], [99, 99, 99], 5, EOS)
        assert mask == [False]
        assert corr == 99
        assert stype == STEP_REJECT
        assert eos is False

    def test_c1_rejection_mid_position(self):
        # accepted, accepted, rejected at j=2
        mask, corr, stype, eos = verify_round(
            [10, 11, 12], [10, 11, 55, 99], 5, EOS
        )
        assert mask == [True, True, False]
        assert corr == 55
        assert stype == STEP_REJECT
        assert eos is False

    def test_c1_rejection_correction_is_eos(self):
        mask, corr, stype, eos = verify_round([10], [EOS, 99], 5, EOS)
        assert mask == [False]
        assert corr == EOS
        assert eos is True

    def test_c2_full_accept_with_draft_eos(self):
        # proposal ends with EOS -> no bonus even with budget
        mask, corr, stype, eos = verify_round([10, EOS], [10, EOS, 99], 5, EOS)
        assert mask == [True, True]
        assert corr is None
        assert stype == STEP_FULL_EOS
        assert eos is True

    def test_c3_full_accept_bonus(self):
        mask, corr, stype, eos = verify_round([10, 11], [10, 11, 20], 5, EOS)
        assert mask == [True, True]
        assert corr == 20
        assert stype == STEP_FULL_BONUS
        assert eos is False

    def test_c3_full_accept_bonus_is_eos(self):
        mask, corr, stype, eos = verify_round([10, 11], [10, 11, EOS], 5, EOS)
        assert mask == [True, True]
        assert corr == EOS
        assert stype == STEP_FULL_EOS
        assert eos is True

    def test_c4_budget_blocks_bonus(self):
        # budget_left == len(proposal): no headroom -> no bonus
        mask, corr, stype, eos = verify_round([10, 11], [10, 11, 20], 2, EOS)
        assert mask == [True, True]
        assert corr is None
        assert stype == STEP_BUDGET
        assert eos is True

    def test_c4_budget_allows_bonus_with_headroom(self):
        mask, corr, stype, eos = verify_round([10, 11], [10, 11, 20], 3, EOS)
        assert corr == 20
        assert stype == STEP_FULL_BONUS

    def test_pred_length_validated(self):
        with pytest.raises(AssertionError):
            verify_round([10], [10], 5, EOS)


# ---------------------------------------------------------------------------
# generate_turn — the pinned reference scenarios
# ---------------------------------------------------------------------------


class TestGenerateTurn:
    def test_reference_three_step_case(self):
        # The hand-computed reference from tests/test_eval_acceptance.py's
        # docstring, produced end-to-end by the loop with scripted sides.
        # Verifier script entries are preds per round (proposal+1 long).
        # The reference's 3 steps emit 9 non-EOS tokens, so generation
        # continues; a 4th scripted round ends the turn with an EOS
        # correction, and a 5th proposal is never made.
        proposer = ScriptedProposer([
            [10, 11, 12],  # step 0: 2 accepted, rejected at 2 -> corr 99
            [13, 14, 15],  # step 1: all accepted -> bonus 98
            [16, 17],      # step 2: 1 accepted, rejected at 1 -> corr 97
            [12, 12],      # step 3: rejected at 0 -> corr EOS, turn ends
        ])
        verifier = ScriptedVerifier([
            [10, 11, 99, 99],   # preds for [10,11,12] + bonus
            [13, 14, 15, 98],   # preds for [13,14,15] + bonus
            [16, 97, 99],       # preds for [16,17] + bonus
            [EOS, 99, 99],      # preds for [12,12] + bonus
        ])
        res = generate_turn(proposer, verifier, [1, 2, 3, 4], EOS, k=3,
                           max_new_tokens=50, query_id="q0")
        # the reference: verified 2,3,1 (+ EOS-corr round) -> metrics
        # over the 3 reference steps match; the 4th adds a rejected-only
        m = flat_metrics(res.events[:3])
        assert m["alpha"] == pytest.approx(0.75)
        assert m["tau"] == pytest.approx(2.0)
        assert m["bonus_rate"] == pytest.approx(1.0)
        assert m["tokens_emitted"] == 9
        assert [e["step_type"] for e in res.events] == [
            STEP_REJECT, STEP_FULL_BONUS, STEP_REJECT, STEP_REJECT,
        ]
        assert res.output_ids == [10, 11, 99, 13, 14, 15, 98, 16, 97, EOS]
        assert res.events[3]["correction_token"] == EOS
        assert res.events[3]["eos"] is True

    def test_c2_short_proposal_ends_turn(self):
        # draft proposes EOS mid-proposal (len 2 < k); accepted -> turn
        # ends with NO bonus
        proposer = ScriptedProposer([[10, EOS]])
        verifier = ScriptedVerifier([[10, EOS, 99]])
        res = generate_turn(proposer, verifier, [1], EOS, k=3,
                            max_new_tokens=50, query_id="q")
        assert len(res.events) == 1
        e = res.events[0]
        assert e["accept_mask"] == [True, True]
        assert e["correction_token"] is None
        assert e["eos"] is True
        assert e["step_type"] == STEP_FULL_EOS
        assert res.output_ids == [10, EOS]

    def test_c4_budget_caps_proposal(self):
        # budget 3, k=5: proposal must be capped at 3 BEFORE proposing
        calls = []

        class CapProposer:
            def reset(self, tokens):
                self.canonical = list(tokens)

            def propose(self, k):
                calls.append(k)
                return [10] * k

            def sync(self, pre_len, emitted):
                self.canonical.extend(emitted)

        class AccVerifier(TableVerifier):
            pass

        p = CapProposer()
        res = generate_turn(p, AccVerifier(), [1], EOS, k=5,
                            max_new_tokens=3, query_id="q")
        assert calls == [3], f"cap not applied: propose called with {calls}"
        # all accepted, budget_left == 3 == len(proposal) -> no bonus, stop
        assert res.output_ids == [10, 10, 10]
        assert res.events[-1]["step_type"] == STEP_BUDGET
        assert res.events[-1]["eos"] is True

    def test_c4_budget_exact_no_overshoot(self):
        # acceptance + correction must never exceed max_new_tokens.
        # budget 4: round 1 accepts 3 w/ bonus (4 emitted) -> budget 0
        # -> the loop exits WITHOUT a second proposal.
        proposer = ScriptedProposer([[10, 11, 12], [13]])
        verifier = ScriptedVerifier([[10, 11, 12, 20], [13, 99]])
        res = generate_turn(proposer, verifier, [1], EOS, k=3,
                            max_new_tokens=4, query_id="q")
        assert len(res.output_ids) == 4
        assert proposer.n_propose_calls == 1
        assert res.events[-1]["step_type"] == STEP_FULL_BONUS

    def test_no_true_after_false_ever(self):
        # torture: long random-ish scripted sequence; the analyzer
        # raises on True-after-False, so round-tripping through
        # load_events-style validation via flat_metrics is the check.
        import random

        rng = random.Random(0)
        for trial in range(50):
            k = rng.randint(1, 5)
            n_rounds = rng.randint(2, 6)
            proposals = []
            preds = []
            for _ in range(n_rounds):
                n_p = rng.randint(1, k)
                proposals.append([rng.randint(0, 30) for _ in range(n_p)])
                preds.append(
                    [rng.randint(0, 30) for _ in range(n_p)] + [rng.randint(0, 30)]
                )
            proposer = ScriptedProposer(proposals)
            verifier = ScriptedVerifier(preds)
            try:
                res = generate_turn(proposer, verifier, [1], EOS, k=k,
                                    max_new_tokens=100, query_id="q")
            except IndexError:
                continue  # script exhausted before budget/EOS: fine
            # every mask is leading-Trues + optional single False
            for e in res.events:
                am = e["accept_mask"]
                seen_false = False
                for b in am:
                    assert not (seen_false and b), "True after False in mask"
                    seen_false = seen_false or not b

    def test_progress_guarantee(self):
        # a proposer that always gets rejected still emits one token
        # (the correction) per round -> terminates by budget alone
        class AlwaysRejectProposer(ChainProposer):
            def propose(self, k):
                return [666]  # never in T_ACCEPT -> always rejected

        res = generate_turn(AlwaysRejectProposer([]), TableVerifier(), [1],
                            EOS, k=4, max_new_tokens=7, query_id="q")
        assert res.output_ids == [TableVerifier.T_CORR] * 7
        # 7 corrections from 7 rejection rounds; the last round's
        # correction hits the budget exactly (budget_left == 1 == cap,
        # but the proposal is REJECTED so C4's bonus rule never applies —
        # the emitted correction fills the budget and stops the loop)
        assert all(e["step_type"] == STEP_REJECT for e in res.events)
        assert len(res.events) == 7


# ---------------------------------------------------------------------------
# C6: multi-turn records
# ---------------------------------------------------------------------------


class TestGenerateRecord:
    @staticmethod
    def _record(n_prompt=4, turns=((1, 3, 2), (0, 2, 4)), qid="q"):
        """Build a synthetic record: labels/-100 context, per-turn loss
        spans. `turns` = (pad_before, loss_len, pad_after) per turn."""
        ids, labels = [], []
        pad = 0
        for pad_before, loss_len, pad_after in turns:
            ids += [pad] * pad_before
            labels += [-100] * pad_before
            ids += list(range(100, 100 + loss_len))
            labels += list(range(100, 100 + loss_len))
            ids += [pad] * pad_after
            labels += [-100] * pad_after
        # relabel: context tokens are unique-ish filler (id 0 repeated ok)
        return {
            "input_ids": ids[: sum(t[0] + t[1] + t[2] for t in turns)],
            "labels": labels,
            "query_id": qid,
            "n_prompt_tokens": n_prompt,
        }

    def test_turn_starts_from_labels(self):
        from src.analysis.eval_acceptance import _turn_starts_from_labels
        rec = self._record()
        starts = _turn_starts_from_labels(rec["labels"])
        # turns = ((1,3,2),(0,2,4)): turn-0 loss at 1..3, turn-1 loss at 6..7
        assert starts == [1, 6]

    def test_multi_turn_events_carry_turn_and_reset_step(self):
        rec = self._record()
        # proposer proposes one thing forever; verifier accepts first
        # proposal token, rejects the rest; turn ends when corr == EOS
        class OneShot:
            def __init__(self):
                self.calls = 0

            def reset(self, tokens):
                pass

            def propose(self, k):
                self.calls += 1
                return [100, 101, 102][:k]

            def sync(self, pre_len, emitted):
                pass

        class FirstOk:
            """accepts position 0 (if proposal[0] in ACCEPT), rejects 1+;
            bonus never EOS; correction = EOS when we want the turn to
            end: track turn budget via emit count"""

            def __init__(self):
                self.emitted_this_turn = 0

            def reset(self, tokens):
                self.emitted_this_turn = 0

            def predict(self, tokens):
                # accept everything (matching) except make the FINAL
                # round of each turn end via proposal capped at budget
                preds = [t for t in tokens]
                preds.append(999)
                return preds

            def sync(self, pre_len, emitted):
                self.emitted_this_turn += len(emitted)

        p, v = OneShot(), FirstOk()
        events, outputs = generate_record(rec, p, v, EOS, k=3,
                                          max_new_tokens=2)
        # turn budget 2: round 1 = 3 proposals all accepted + bonus=999 ->
        # 4 emitted?? No: budget caps proposal at 2, all accepted, no
        # headroom -> budget stop, 2 tokens. So 1 event per turn.
        assert len(events) == 2
        assert [e["turn"] for e in events] == [0, 1]
        assert all(e["step"] == 0 for e in events)
        assert [o["turn"] for o in outputs] == [0, 1]
        assert all(o["n_tokens"] == 2 for o in outputs)

    def test_single_turn_omits_turn_field(self):
        rec = self._record(turns=((4, 3, 0),))
        p = ChainProposer([100, 101, 102, 103])
        # verifier accepts everything, bonus 999, budget 3 -> 1 event
        v = FirstOk() if False else _AcceptAllVerifier()
        events, outputs = generate_record(rec, p, v, EOS, k=5,
                                          max_new_tokens=3)
        assert len(events) == 1
        assert "turn" not in events[0]
        assert outputs[0]["turn"] is None

    def test_teacher_forcing_between_turns(self):
        """Between turns, adapters must be synced onto the ground-truth
        span: after every turn-transition sync the canonical view IS a
        prefix of the record's input_ids (the generated tokens are
        replaced by the truth); after a round sync it is a prefix plus
        generated tokens."""
        from src.analysis.eval_acceptance import _turn_starts_from_labels

        rec = self._record()
        starts = _turn_starts_from_labels(rec["labels"])
        audit_log = []

        class Auditing(TableVerifier):
            T_ACCEPT = set(range(200))  # accepts anything < 200

            def sync(self, pre_len, emitted):
                super().sync(pre_len, emitted)
                audit_log.append((pre_len, list(emitted)))

        p = ChainProposer([100, 101, 102, 103, 100, 101])
        v = Auditing()
        events, outputs = generate_record(rec, p, v, EOS, k=3,
                                          max_new_tokens=3)
        # two kinds of sync: turn transitions (emitted == the truth span
        # from this turn start to the next) and round syncs (emitted are
        # generated tokens). Distinguish by exact truth-span match.
        turn_syncs = [
            (pre, em) for pre, em in audit_log
            if any(
                pre == s and em == rec["input_ids"][s : s2]
                for s, s2 in zip(starts, starts[1:] + [len(rec["input_ids"])])
            )
        ]
        # exactly one turn-transition sync (before turn 1)
        assert len(turn_syncs) == 1
        pre, em = turn_syncs[0]
        assert pre == starts[0]
        assert em == rec["input_ids"][starts[0] : starts[1]]
        # at the moment of the turn sync, the canonical view equals the
        # record prefix exactly (audited inside Auditing.sync via the
        # truth-span match); afterwards turn 1 generates past it again:
        # the chain (pos rewound to the truth prefix by sync) proposes
        # [101, 102, 103], all accepted, budget stop at 3
        assert v.canonical == rec["input_ids"][: starts[1]] + [101, 102, 103]
        # and both turns generated their full budget
        assert all(len(o["output_ids"]) == 3 for o in outputs)

    @staticmethod
    def _is_truth_span(rec, pre, emitted):
        return rec["input_ids"][pre : pre + len(emitted)] == emitted


class _AcceptAllVerifier(TableVerifier):
    T_ACCEPT = set(range(1000))
    T_NEXT = 999


# ---------------------------------------------------------------------------
# C7: n-gram proposer
# ---------------------------------------------------------------------------


class TestNgramProposer:
    def test_basic_lookup(self):
        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        p.reset([5, 6, 7, 5, 6])
        # suffix [5,6] occurs at 0 -> followers [7,5,6] up to k=3
        # (it proposes up to k tokens continuing the matched n-gram run)
        assert p.propose(3) == [7, 5, 6]
        assert p.propose(1) == [7]

    def test_longest_suffix_wins(self):
        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        p.reset([1, 2, 3, 1, 2, 3])
        # suffix [1,2,3] matches at 0 -> follower 1; k=2 -> [1, 2]
        assert p.propose(2) == [1, 2]

    def test_no_match_pad_fallback(self):
        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        p.reset([1, 2, 3, 4, 5])
        # no suffix of len>=2 recurs -> pad fallback
        assert p.propose(4) == [PAD]
        assert p.stats()["n_pad_fallback"] == 1

    def test_proposal_never_exceeds_k(self):
        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        p.reset([1, 2, 3, 1, 2, 3, 4, 5])
        assert len(p.propose(2)) <= 2

    def test_sync_keeps_matched_prefix(self):
        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        p.reset([1, 2, 3, 4])
        p.seq.append(9)  # speculative tail
        p.sync(4, [9, 10])  # canonical now [1,2,3,4,9,10]
        assert p.seq == [1, 2, 3, 4, 9, 10]

    def test_sync_crops_rejected_tail(self):
        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        p.reset([1, 2, 3, 4])
        p.seq.extend([7, 8, 9])  # speculative tail, only 7 survives
        p.sync(4, [7, 55])  # canonical [1,2,3,4,7,55]
        assert p.seq == [1, 2, 3, 4, 7, 55]

    def test_end_to_end_with_loop(self):
        """n-gram loop run: copyable tokens accepted, pad fallback
        rounds rejected, output never diverges from the verifier."""
        # verifier: accepts 10..15, rejects everything else with corr 99,
        # bonus 20 (never EOS). Prompt [10, 11, 12] — the n-gram proposer
        # will find suffix matches inside the emitted stream.
        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        v = TableVerifier()
        res = generate_turn(p, v, [10, 11, 12], EOS, k=3, max_new_tokens=12,
                            query_id="q")
        assert len(res.output_ids) == 12
        # every emitted token is either an accepted proposal token in
        # T_ACCEPT, the correction 99, or the bonus 20 — never a raw
        # rejected proposal token
        for t in res.output_ids:
            assert t in (TableVerifier.T_CORR, TableVerifier.T_NEXT) or t in TableVerifier.T_ACCEPT

    def test_round_trip_through_analyzer(self):
        """n-gram events must satisfy the analyzer's schema: no
        True-after-False, load_events-validated JSONL, and a report."""
        import tempfile
        from pathlib import Path

        from src.analysis.eval_acceptance import load_events

        p = NgramProposer(min_n=2, max_n=5, pad_id=PAD)
        v = TableVerifier()
        res = generate_turn(p, v, [10, 11, 12], EOS, k=3, max_new_tokens=12,
                            query_id="q")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for e in res.events:
                f.write(json.dumps(e) + "\n")
            path = f.name
        events = load_events(path)
        Path(path).unlink()
        m = flat_metrics(events)
        assert m["n_events"] == len(res.events)
        assert 0.0 < m["alpha"] <= 1.0
