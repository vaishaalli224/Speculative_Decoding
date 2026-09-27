"""Proposers for the instrumented speculative-decoding loop (§6.5).

A *proposer* produces up to `k` candidate continuation tokens for the
current sequence. Anything that can do that can play the draft side of the
loop — an HF causal LM (greedy) or an n-gram / prompt-lookup matcher
(plan §4.6: the region-split analysis needs n-gram acceptance per region,
which vLLM does not expose).

Proposers are stateful (they see `append`/`truncate` as the loop runs) so
they can maintain their own cache; the loop drives them through the same
token bookkeeping as the target model.
"""

from __future__ import annotations

from typing import Protocol

from src.data_prep.render import get_tokenizer


class Proposer(Protocol):
    """The draft side of the loop: propose up to `k` next tokens."""

    def propose(self, k: int) -> list[int]:
        """Return 1..k proposed token ids for the current sequence end.

        MUST be non-empty (an empty proposal is a no-op round the loop
        cannot express; a proposer with nothing to offer must instead
        yield a single fill/pad token so the target corrects it — this
        keeps the loop's step semantics uniform and gives the n-gram
        baseline an honest per-round α). Returning more than k is a
        contract violation the loop asserts on.
        """
        ...  # pragma: no cover

    def reset(self, tokens: list[int]) -> None:
        """Start a fresh sequence from `tokens` (the prompt)."""
        ...  # pragma: no cover

    def sync(self, pre_len: int, emitted: list[int]) -> None:
        """Canonical sequence is now view[:pre_len] + emitted. Keep any
        speculative prefix of `emitted` the view already matches, crop
        the rest, append the remainder. (The loop never truncates without
        telling the proposer what survived — a single method keeps the
        n-gram window and the KV cache honest with one code path.)"""
        ...  # pragma: no cover


class NgramProposer:
    """Prompt-lookup / n-gram proposer (plan §4.6).

    Finds the longest suffix of the current sequence (length in
    [min_n, max_n]) that also occurs earlier in the sequence, and proposes
    the tokens that followed that earlier occurrence. Mirrors vLLM's
    `ngram` speculative method (prompt_lookup_min/max defaults 2/5) and
    PSG/PLD prompt-lookup decoding.

    On no match this proposer emits a single PAD token as its proposal —
    the target rejects it and emits the real next token. Those rounds are
    honest data points for the region analysis: n-gram's failure mode
    (nothing copyable in the prompt) is exactly where a trained draft must
    win. The zero-length-proposal alternative would make step counting
    conventions diverge between proposers, so it is banned by the Proposer
    contract (C7). Exactness is unaffected: the loop only ever emits the
    target's own argmax, so a rejected PAD costs a round, never a token
    of divergence.
    """

    def __init__(self, min_n: int = 2, max_n: int = 5, pad_id: int | None = None):
        self.min_n = min_n
        self.max_n = max_n
        if pad_id is None:
            pad_id = get_tokenizer().pad_token_id
            if pad_id is None:  # Qwen has no pad token; use im_end as sacrificial
                pad_id = get_tokenizer().convert_tokens_to_ids("<|im_end|>")
        self.pad_id = pad_id
        self.seq: list[int] = []
        self.n_proposed = 0
        self.n_pad_fallback = 0

    # -- Proposer API ------------------------------------------------------

    def reset(self, tokens: list[int]) -> None:
        self.seq = list(tokens)
        self.n_proposed = 0
        self.n_pad_fallback = 0

    def sync(self, pre_len: int, emitted: list[int]) -> None:
        keep = pre_len
        for i, tok in enumerate(emitted):
            if pre_len + i < len(self.seq) and self.seq[pre_len + i] == tok:
                keep += 1
            else:
                break
        del self.seq[keep:]
        self.seq.extend(emitted[keep - pre_len:])

    def propose(self, k: int) -> list[int]:
        # longest suffix match wins: try max_n down to min_n
        for n in range(min(self.max_n, len(self.seq)), self.min_n - 1, -1):
            suffix = self.seq[-n:]
            # search over all earlier occurrences (excluding the suffix itself)
            for start in range(len(self.seq) - n):
                if self.seq[start : start + n] == suffix:
                    proposal = self.seq[start + n : start + n + k]
                    if proposal:
                        self.n_proposed += 1
                        return list(proposal)
        # no match: single sacrificial pad token (see class docstring)
        self.n_pad_fallback += 1
        return [self.pad_id]

    # -- introspection for runsheets / analysis -----------------------------

    def stats(self) -> dict:
        return {
            "n_proposed": self.n_proposed,
            "n_pad_fallback": self.n_pad_fallback,
            "pad_id": self.pad_id,
        }
