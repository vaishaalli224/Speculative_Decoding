"""Stage-1 target generation harness (plan.md §2 Stage 1, §5 hours 3.5-4.5).

Serves the frozen target (Qwen2.5-Coder-14B-Instruct) and generates greedy
assistant continuations on the 5,000 frozen Stage-1 xLAM contexts, saving
top-k logprobs (k=20 per the plan; vLLM's max_logprobs default is exactly
20) for every generated token — the teacher distributions for the Stage-1
KD warm-start. Output JSONL schema: see src/data_prep/build_distill_data.py
(the assembling side; the two docstrings pin one schema).

Architecture mirrors bench_vllm.py: a vLLM-free, torch-free core (engine
protocol, logprob conversion, JSONL IO — golden-tested on the mac dev
host) plus two thin adapters that import their frameworks lazily.
VLLMGenEngine runs on the H100 (the only GPU-day path); HFGenEngine exists
for the local toy-scale dry run (plan §6.8) and the SPEC_REALMODELS tests
with Coder-0.5B as a stand-in target — never used on the GPU day.

Pinned conventions (golden-tested in tests/test_gen_stage1.py):

  E1 greedy: temperature 0.0, one seed. The emitted token is the target's
     argmax (plan: Stage-1 data is generated greedy).
  E2 EOS normalization: the terminal EOS (<|im_end|>) is ALWAYS part of
     output_ids when the model stopped (the KD labels must teach stopping)
     and carries its top-k logprobs like any other position; a "length"
     stop has no EOS. Both adapters include the stop token in their token
     ids (verified for vLLM semantics and empirically for transformers
     5.17's generate() — its scores cover the EOS row too); normalize_
     output still guards the withheld case, engine-independent.
  E3 logprob fidelity: entries are the engine's own floats, sorted by
     logprob DESC then id ASC; ragged per-position lengths preserved.
     A position with a shorter list than k is legal (engine cap).
  E4 single-call generation: all prompts go to ONE engine.generate call —
     vLLM's offline API continuous-batches internally; fixed-concurrency
     chunking is a bench (§4.2) concept, not a production one.
  E5 frozen order: prompts and output lines follow frozen/stage1_idx.json
     order; --limit N = first N (same subset rule everywhere, G6).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Protocol

DEFAULT_LOGPROBS = 20  # plan §2 Stage 1; vLLM max_logprobs default
DEFAULT_MAX_NEW_TOKENS = 512  # matches bench_vllm / the instrumented loop
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GPU_MEM_UTIL = 0.90
MAX_LOGPROBS_CAP = 20  # vLLM ModelConfig.max_logprobs default — refuse more


class GenEngine(Protocol):
    """The serving side; the only framework-importing code (adapters below)."""

    def generate(
        self, prompts: list[list[int]], params: dict
    ) -> list[dict]: ...
    # each output: {"token_ids": [...], "logprobs": [dict[int, float], ...],
    #               "finish_reason": "stop" | "length", "eos_included": bool}


# ---------------------------------------------------------------------------
# Pure core (vLLM-free, torch-free)
# ---------------------------------------------------------------------------


def sort_logprobs(entries: dict[int, float] | None) -> list[list[float]]:
    """One position's logprob dict -> [[id, value], ...] sorted by value
    DESC then id ASC (E3; JSON lists keep the order, so the file is stable)."""
    if entries is None:
        return []
    return [[int(t), float(v)] for t, v in
            sorted(entries.items(), key=lambda kv: (-kv[1], kv[0]))]


def normalize_output(out: dict, eos_id: int) -> dict:
    """Engine output -> the JSONL schema (E2): append the EOS when the
    engine withheld it (HF — its logprobs already cover the withheld
    position), never duplicate it when it did not (vLLM); sort each
    position's logprobs (E3)."""
    ids = list(out["token_ids"])
    lps = list(out["logprobs"])
    eos_included = bool(out.get("eos_included", False))
    if out["finish_reason"] == "stop" and not eos_included:
        ids.append(eos_id)  # E2: HF's scores cover the EOS position too
    if len(lps) != len(ids):
        raise AssertionError(
            f"engine returned {len(lps)} logprob positions for {len(ids)} tokens"
        )
    return {
        "token_ids": ids,
        "logprobs": [sort_logprobs(p) for p in lps],
        "finish_reason": out["finish_reason"],
    }


def write_generations(path: str, prompts: list[dict], outputs: list[dict]) -> None:
    """JSONL per prompt in frozen order; query_id joined from the prompt.
    Field names become the assembling side's schema (output_ids, not the
    internal token_ids)."""
    if len(prompts) != len(outputs):
        raise AssertionError("prompt/output count mismatch")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for pr, o in zip(prompts, outputs):
            rec = {"query_id": pr["query_id"], "output_ids": o["token_ids"],
                   "logprobs": o["logprobs"],
                   "finish_reason": o["finish_reason"]}
            f.write(json.dumps(rec) + "\n")


def _git_commit() -> str | None:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or None


# ---------------------------------------------------------------------------
# Adapters — the only framework-importing code, built lazily by the CLI
# ---------------------------------------------------------------------------


class VLLMGenEngine:
    """vLLM adapter (H100 only). Token-id prompts in, detokenize=False.

    SamplingParams(logprobs=k) returns, per generated position, a dict
    {token_id: Logprob(logprob, rank, decoded_token)} — ranks 1-based, the
    greedy token rank 1. The EOS vLLM emits on stop IS part of token_ids
    and carries its own logprobs, so eos_included=True.
    """

    def __init__(
        self,
        model: str,
        logprobs: int = DEFAULT_LOGPROBS,
        dtype: str = "bfloat16",
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL,
    ):
        if logprobs > MAX_LOGPROBS_CAP:
            raise ValueError(
                f"logprobs={logprobs} exceeds vLLM's max_logprobs default "
                f"({MAX_LOGPROBS_CAP}) — pass max_logprobs to the engine or "
                "lower --logprobs (the plan pins 20 anyway)"
            )
        import vllm  # fail loudly if missing (H100 host only)

        self.llm = vllm.LLM(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.logprobs = logprobs
        self._vllm_version = vllm.__version__

    def generate(self, prompts: list[list[int]], params: dict) -> list[dict]:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=params.get("temperature", 0.0),
            max_tokens=params["max_tokens"],
            seed=params.get("seed"),
            logprobs=self.logprobs,
            detokenize=params.get("detokenize", False),
        )
        outs = self.llm.generate(
            [{"prompt_token_ids": list(p)} for p in prompts], sp
        )
        results = []
        for o in outs:
            co = o.outputs[0]
            lps = []
            for pos in (co.logprobs or []):
                lps.append(None if pos is None else
                           {t: lp.logprob for t, lp in pos.items()})
            results.append({
                "token_ids": list(co.token_ids),
                "logprobs": lps,
                "finish_reason": "length" if co.finish_reason == "length"
                                  else "stop",
                "eos_included": True,  # vLLM includes the stop token
            })
        return results

    def info(self) -> dict:
        out = {"vllm_version": self._vllm_version}
        try:
            import torch

            if torch.cuda.is_available():
                out["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover — info only
            pass
        return out

    def teardown(self) -> None:
        import gc

        del self.llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


class HFGenEngine:
    """HF adapter for local toy-scale dry runs and the real-model tests.

    Greedy generate with output_scores=True, log_softmax in fp32, top-k
    per position. Verified against transformers 5.17 on real prompts
    (2026-09-27): generate() DOES include the stop token in the returned
    sequence AND g.scores has exactly one row per emitted token including
    the EOS row (its argmax == eos_id) — so eos_included=True and E2 needs
    no repair on this path either.
    """

    def __init__(self, model_id: str, logprobs: int = DEFAULT_LOGPROBS,
                 dtype=None, device: str = "auto"):
        import torch
        import transformers

        if dtype is None:
            dtype = torch.float32
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        )
        self.model.eval()
        if device != "auto":
            self.model = self.model.to(device)
        self.tok = transformers.AutoTokenizer.from_pretrained(model_id)
        self.eos_id = int(self.tok.eos_token_id)
        self.k = logprobs

    def generate(self, prompts: list[list[int]], params: dict) -> list[dict]:
        import torch

        results = []
        for ids in prompts:
            inp = torch.tensor([list(ids)], dtype=torch.long,
                               device=self.model.device)
            with torch.no_grad():
                g = self.model.generate(
                    inp,
                    max_new_tokens=params["max_tokens"],
                    do_sample=False,
                    temperature=None,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            scores = torch.stack(g.scores, dim=1)[0].float()  # [steps, vocab]
            lps = torch.log_softmax(scores, dim=-1)
            topk = torch.topk(lps, min(self.k, lps.shape[-1]), dim=-1)
            positions = []
            for step in range(topk.indices.shape[0]):
                positions.append({
                    int(t): float(v)
                    for t, v in zip(topk.indices[step].tolist(),
                                   topk.values[step].tolist())
                })
            out_ids = g.sequences[0][inp.shape[1]:].tolist()
            stopped = len(out_ids) < params["max_tokens"]
            results.append({
                "token_ids": out_ids,
                "logprobs": positions,
                "finish_reason": "stop" if stopped else "length",
                "eos_included": True,  # verified: sequence includes the EOS
            })
        return results

    def info(self) -> dict:
        return {"engine": "hf"}

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from src.data_prep.build_distill_data import load_stage1_prompts
    from src.data_prep.render import get_tokenizer

    ap = argparse.ArgumentParser(
        description="Stage-1 target generation (plan §2): greedy continuations "
                    "+ top-k logprobs on the frozen Stage-1 xLAM contexts"
    )
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--logprobs", type=int, default=DEFAULT_LOGPROBS)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N Stage-1 contexts in frozen order (E5)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto",
                    help="HF engine placement (cuda:0 on the GPU host)")
    ap.add_argument("--detokenize", action="store_true",
                    help="escape hatch: allow detokenize=True in the vLLM "
                         "request (default False — token-id fidelity)")
    ap.add_argument("--out", required=True, help="generation JSONL")
    ap.add_argument("--meta-out", default=None, help="run metadata JSON")
    ap.add_argument("--metrics-out", default=None,
                    help="append one metrics line here (results/metrics.jsonl)")
    args = ap.parse_args()

    eos_id = int(get_tokenizer().eos_token_id)  # C5-style: from the tokenizer
    prompts = load_stage1_prompts(args.limit)
    params = {
        "max_tokens": args.max_new_tokens,
        "temperature": 0.0,  # E1
        "seed": args.seed,
        "detokenize": args.detokenize,
    }

    t0 = time.perf_counter()
    if args.engine == "vllm":
        try:
            engine = VLLMGenEngine(
                args.model, args.logprobs, args.dtype, DEFAULT_MAX_MODEL_LEN,
                DEFAULT_GPU_MEM_UTIL,
            )
        except ModuleNotFoundError as e:
            raise SystemExit(
                f"vLLM is not importable on this host: {e}\n"
                "(gen_stage1's core is vLLM-free — the vllm engine runs "
                "only on the H100 host; use --engine hf for the local "
                "toy-scale dry run)"
            ) from e
    else:
        engine = HFGenEngine(args.model, args.logprobs, device=args.device)

    outputs = [normalize_output(o, eos_id)
               for o in engine.generate(
                   [p["prompt_ids"] for p in prompts], params)]
    if len(outputs) != len(prompts):
        raise AssertionError("engine output count mismatch")
    wall = time.perf_counter() - t0
    write_generations(args.out, prompts, outputs)
    engine.teardown()

    n_tok = sum(len(o["token_ids"]) for o in outputs)
    meta = {
        "engine": args.engine,
        "model": args.model,
        "logprobs": args.logprobs,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "n_prompts": len(prompts),
        "n_tokens": n_tok,
        "wall_s": round(wall, 2),
        "git_commit": _git_commit(),
        "engine_info": engine.info(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.meta_out:
        p = Path(args.meta_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=2))
    if args.metrics_out:
        from src.serving.bench_vllm import append_metrics

        append_metrics(args.metrics_out, {"kind": "stage1_datagen", **meta})
    print(
        f"generated {n_tok} tokens over {len(prompts)} prompts "
        f"({wall:.1f}s) -> {args.out}"
    )


if __name__ == "__main__":
    main()
