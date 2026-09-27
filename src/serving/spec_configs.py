"""Speculative-decoding config construction for vLLM (plan.md §1.5, §4.6).

Single source of truth for every `speculative_config` dict the GPU-day
runbook passes to vLLM. The dicts are pinned verbatim against the plan:

  draft model: {"method": "draft_model", "model": <draft>,
                "num_speculative_tokens": k, "max_model_len": 16384}
  n-gram:      {"method": "ngram", "num_speculative_tokens": k,
                "prompt_lookup_min": 2, "prompt_lookup_max": 5}

Sampling params (temperature etc.) NEVER go in here — they belong to the
request (SamplingParams), not the engine config (plan §1.5). AR-only runs
carry speculative_config=None.

vllm-free and torch-free: buildable and golden-tested on the mac dev
host; the H100 host feeds the dicts straight into
`LLM(speculative_config=...)` (day-0 smoke test, plan §1.5/§5 — the one
thing that cannot be pre-verified locally).
"""

from __future__ import annotations

DEFAULT_DRAFT_MAX_MODEL_LEN = 16384  # plan §1.5/§9: prompt-cap before gpu_mem bump
NGRAM_LOOKUP_MIN = 2  # plan §4.6
NGRAM_LOOKUP_MAX = 5

METHODS = ("ar", "draft_model", "ngram")


def draft_model_config(
    draft_model: str, num_speculative_tokens: int, max_model_len: int = DEFAULT_DRAFT_MAX_MODEL_LEN
) -> dict:
    """Draft-model speculative config, verbatim per plan §1.5."""
    cfg = {
        "method": "draft_model",
        "model": draft_model,
        "num_speculative_tokens": num_speculative_tokens,
        "max_model_len": max_model_len,
    }
    validate_spec_config(cfg)
    return cfg


def ngram_config(
    num_speculative_tokens: int,
    prompt_lookup_min: int = NGRAM_LOOKUP_MIN,
    prompt_lookup_max: int = NGRAM_LOOKUP_MAX,
) -> dict:
    """n-gram / prompt-lookup config, verbatim per plan §1.5/§4.6."""
    cfg = {
        "method": "ngram",
        "num_speculative_tokens": num_speculative_tokens,
        "prompt_lookup_min": prompt_lookup_min,
        "prompt_lookup_max": prompt_lookup_max,
    }
    validate_spec_config(cfg)
    return cfg


def build_spec_config(
    method: str,
    draft_model: str | None = None,
    num_speculative_tokens: int = 5,
    prompt_lookup_min: int = NGRAM_LOOKUP_MIN,
    prompt_lookup_max: int = NGRAM_LOOKUP_MAX,
    max_model_len: int = DEFAULT_DRAFT_MAX_MODEL_LEN,
) -> dict | None:
    """Dispatch on `method`; "ar" returns None (no speculative_config key).

    `num_speculative_tokens` defaults to 5 only for CLI convenience (the §8
    draft-choice rule measures at k=5); the n-gram baseline's own §1.5
    example uses 4 — callers pass k explicitly (run_baselines.sh does).
    """
    if method == "ar":
        return None
    if method == "draft_model":
        return draft_model_config(draft_model, num_speculative_tokens, max_model_len)
    if method == "ngram":
        return ngram_config(num_speculative_tokens, prompt_lookup_min, prompt_lookup_max)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def validate_spec_config(cfg: dict) -> None:
    """Fail loudly on anything the plan's two shapes don't allow."""
    method = cfg.get("method")
    if method not in ("draft_model", "ngram"):
        raise ValueError(f"speculative_config method must be draft_model or ngram, got {method!r}")
    k = cfg.get("num_speculative_tokens")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"num_speculative_tokens must be an int >= 1, got {k!r}")
    if method == "draft_model":
        model = cfg.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("draft_model speculative_config needs a non-empty 'model'")
        mml = cfg.get("max_model_len")
        if not isinstance(mml, int) or mml < 1:
            raise ValueError(f"max_model_len must be an int >= 1, got {mml!r}")
    else:  # ngram: no model key, lookup window well-formed
        if "model" in cfg:
            raise ValueError("ngram speculative_config must not carry a 'model' key")
        lo, hi = cfg.get("prompt_lookup_min"), cfg.get("prompt_lookup_max")
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi):
            raise ValueError(
                f"prompt_lookup_min/max must be ints with 1 <= min <= max, got {lo!r}/{hi!r}"
            )
