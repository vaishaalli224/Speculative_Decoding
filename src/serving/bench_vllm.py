"""vLLM bench harness (plan.md §6.6, §4.2, §4.4).

Given (model, spec config, prompts, params) -> tokens/sec + outputs, plus
an exactness mode that compares full token-id sequences of speculative vs.
non-speculative greedy decoding (§4.4). All wall-clock numbers for the
project come from this script (plan §1.5 two-track: α/τ internals come from
the instrumented loop, never from here).

Architecture mirrors instrumented_spec.py: a vLLM-free, torch-free core
(run_bench / exactness_compare / IO helpers — golden-tested on the mac dev
host, where vLLM has no wheels) and a thin VLLMEngine adapter that imports
vLLM lazily and only runs on the H100. The one thing that cannot be
pre-verified locally is that the pinned vLLM build accepts the §1.5
speculative_config dicts — that smoke test is GPU-day step 0 (plan §1.5).

Pinned conventions (golden-tested in tests/test_bench_vllm.py):

  B1 batch semantics: "batch B" (§4.2's fixed concurrency, offline API)
     = the prompt list is chunked deterministically into B-prompt chunks
     and each chunk is one engine.generate call. The engine therefore
     never sees more than B concurrent requests. Chunks stay in frozen
     order; a remainder chunk is last.
  B2 timing: only engine.generate calls are timed (engine construction,
     model load and teardown are outside the wall-clock — they are
     one-time costs, not decoding costs). wall_s for a run = sum over its
     chunks. A warmup pass (--warmup, untimed, outputs discarded) absorbs
     CUDA-graph/compile effects before run 1.
  B3 tokens/sec: gen_tok_s = generated (completion) tokens / wall_s — the
     spec-decode headline number; prompt tokens are reported separately
     (total_tok_s = (prompt+gen)/wall) so prefill-heavy TB-500 numbers can
     be read either way. Generated tokens are what speculation buys.
  B4 runs: --runs N repeats the whole chunk sweep; the reported wall and
     gen_tok_s are the MEDIAN across runs (plan §4: wall-clock = median of
     3 runs). Outputs saved are run 1's (greedy is deterministic; at
     temperature > 0 run 1's outputs are what the file documents — the
     metrics record says so).
  B5 prompts: token-id prompts (input_ids[:n_prompt_tokens] of the frozen
     records — the turn-0 context; n_prompt_tokens IS the first assistant
     turn's start by construction), first N in frozen order via the same
     loader as the instrumented loop. Token ids in, token ids out — no
     re-tokenization anywhere. --per-turn (TB-500) expands each record to
     one teacher-forced prompt per assistant turn — the loop's C6
     protocol, turn starts imported from the analyzer so the two
     instruments cannot desync; the transfer wall-clock then measures the
     same agentic states the acceptance analysis scores. "turn" is present
     only on multi-turn records, exactly like the loop's events.
  B6 exactness (§4.4): greedy, batch 1, one run per side; full output-id
     sequences must match 100% (equal ids => equal lengths). The AR side
     runs inside the same command (its engine is torn down before the
     speculative engine loads — one GPU, ~31 GB of weights at a time) or
     is read from a prior run's outputs file via --compare-vs (same
     records/--limit required, checked before any GPU time is spent).
     --per-turn is refused (turn-expanded outputs repeat query_ids —
     §4.4 runs at record granularity). Exit code 1 on any mismatch; the
     report JSON lists every divergent prompt with the first divergence
     position. If vLLM mismatches here, the plan's fallback is the
     instrumented HF loop for the exactness claim (§4.4).
  B7 outputs JSONL: one {"query_id", "output_ids"} per prompt, record
     order (+ "turn" for per-turn runs). metrics.jsonl: one JSON line per
     bench run appended (plan §5) — the report stage reads only this

CLI examples (GPU day; see scripts/run_*.sh):

  # AR baseline, batch sweep:
  python -m src.serving.bench_vllm frozen/xlam_eval.parquet --method ar \
      --model Qwen/Qwen2.5-Coder-14B-Instruct --temperature greedy \
      --batch 1 --runs 3 --limit 200 \
      --outputs-out results/vllm/ar_greedy_b1.jsonl --metrics-out results/metrics.jsonl

  # untuned 0.5B draft, k=5 greedy batch 1:
  ... --method draft_model --draft-model Qwen/Qwen2.5-Coder-0.5B-Instruct --k 5

  # n-gram baseline:
  ... --method ngram --k 5

  # exactness gate (§4.4), spec vs AR, greedy batch 1:
  python -m src.serving.bench_vllm frozen/xlam_eval.parquet --method draft_model \
      --draft-model <draft> --k 5 --exactness --limit 50 \
      --outputs-out results/vllm/draft05_k5_exact.jsonl \
      --exactness-out results/exactness/draft05_k5.json
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Protocol

from src.analysis.eval_acceptance import load_records
from src.serving.spec_configs import build_spec_config

DEFAULT_MAX_NEW_TOKENS = 512  # matches the instrumented loop's CLI default
DEFAULT_MAX_MODEL_LEN = 16384  # plan §1.5/§9: cap max_model_len before gpu-mem bump
DEFAULT_GPU_MEM_UTIL = 0.90  # plan §1.4: raise to 0.92 only if OOM
DEFAULT_RUNS = 3  # plan §4: wall-clock = median of 3 runs
DEFAULT_WARMUP = 4


class BenchEngine(Protocol):
    """The serving side; the only vLLM-touching code (VLLMEngine below)."""

    def generate(
        self, prompts: list[list[int]], params: dict
    ) -> list[list[int]]: ...


# ---------------------------------------------------------------------------
# Pure core (vLLM-free, torch-free)
# ---------------------------------------------------------------------------


def chunked(items: list, size: int):
    """Deterministic chunks of `size`, in order; remainder chunk last (B1)."""
    if size < 1:
        raise ValueError(f"batch must be >= 1, got {size}")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_prompts(
    records_path: str, limit: int | None = None, per_turn: bool = False
) -> list[dict]:
    """Frozen records -> prompt list, first N records in frozen order (B5).

    Default: one prompt per record — the turn-0 context, input_ids up to
    the record's first assistant token (n_prompt_tokens IS that turn's
    start by construction). xLAM (single-turn) is fully covered by this.

    per_turn=True (TB-500 multi-turn, §4.2 transfer wall-clock): one prompt
    per assistant turn, teacher-forced on the ground-truth context — the
    same protocol as the instrumented loop's C6 (turn starts derived by
    the analyzer's own helper, so the two sides can never desync). The
    "turn" key (0-based) is present only when the record has >1 assistant
    turn, exactly like the loop's events (C6) — single-turn datasets then
    produce identical prompts either way; EOS ends a turn naturally in
    vLLM, so one max_tokens cap serves every turn."""
    from src.analysis.eval_acceptance import _turn_starts_from_labels

    out = []
    for r in load_records(records_path, limit):
        ids = r["input_ids"]
        if per_turn:
            starts = _turn_starts_from_labels(r["labels"])
            multi = len(starts) > 1
            for t, s in enumerate(starts):
                p = {"query_id": r["query_id"], "prompt_ids": ids[:s]}
                if multi:
                    p["turn"] = t
                out.append(p)
            continue
        n = r["n_prompt_tokens"]
        if not 0 < n < len(ids):
            raise AssertionError(
                f"record {r['query_id']}: bad n_prompt_tokens {n} vs "
                f"{len(ids)} ids"
            )
        out.append({
            "query_id": r["query_id"],
            "prompt_ids": ids[:n],
        })
    return out


def run_bench(
    engine: BenchEngine,
    prompts: list[dict],
    params: dict,
    batch: int = 1,
    runs: int = DEFAULT_RUNS,
    warmup: int = 0,
    time_fn=time.perf_counter,
) -> dict:
    """One config's bench: warmup, then `runs` timed chunk sweeps (B1-B4).

    `params` is engine-agnostic ({"max_tokens", "temperature", "seed", ...});
    the adapter maps it to the engine's sampling params. Returns metrics
    plus run 1's outputs as [{"query_id", "output_ids"}].
    """
    if not prompts:
        raise ValueError("no prompts")
    if runs < 1:
        raise ValueError("runs must be >= 1")

    if warmup:
        engine.generate([p["prompt_ids"] for p in prompts[:warmup]], params)

    per_run: list[dict] = []
    outputs: list[dict] | None = None
    for _ in range(runs):
        t0 = time_fn()
        gen_tokens = prompt_tokens = 0
        run_outputs: list[dict] = []
        for chunk in chunked(prompts, batch):
            outs = engine.generate([p["prompt_ids"] for p in chunk], params)
            if len(outs) != len(chunk):
                raise AssertionError(
                    f"engine returned {len(outs)} outputs for {len(chunk)} prompts"
                )
            for p, o in zip(chunk, outs):
                if not isinstance(o, list) or not all(isinstance(t, int) for t in o):
                    raise AssertionError("engine must return token-id lists")
                rec = {"query_id": p["query_id"], "output_ids": o}
                if "turn" in p:  # per-turn prompts: keep outputs disambiguated
                    rec["turn"] = p["turn"]
                run_outputs.append(rec)
                gen_tokens += len(o)
                prompt_tokens += len(p["prompt_ids"])
        wall = time_fn() - t0
        per_run.append({
            "wall_s": wall,
            "gen_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens,
            "gen_tok_s": gen_tokens / wall if wall > 0 else None,
            "total_tok_s": (gen_tokens + prompt_tokens) / wall if wall > 0 else None,
        })
        if outputs is None:  # B4: run 1's outputs are the saved ones
            outputs = run_outputs

    walls = sorted(r["wall_s"] for r in per_run)
    mid = len(walls) // 2
    median_wall = walls[mid] if len(walls) % 2 else (walls[mid - 1] + walls[mid]) / 2
    gtok = sorted(r["gen_tok_s"] for r in per_run if r["gen_tok_s"] is not None)
    median_gtok = gtok[mid] if len(gtok) % 2 else (gtok[mid - 1] + gtok[mid]) / 2
    return {
        "n_prompts": len(prompts),
        "batch": batch,
        "runs": runs,
        "warmup": warmup,
        "max_tokens": params.get("max_tokens"),
        "temperature": params.get("temperature"),
        "seed": params.get("seed"),
        "per_run": per_run,
        "median_wall_s": median_wall,
        "median_gen_tok_s": median_gtok,
        "outputs": outputs,
    }


def exactness_compare(
    baseline: list[dict], spec: list[dict]
) -> dict:
    """§4.4 gate: full token-id sequences must match 100% (B6). Both lists
    are [{"query_id", "output_ids"}] in record order (same prompts, batch 1).
    Equal ids imply equal lengths; list equality enforces both at once."""
    if len(baseline) != len(spec):
        raise AssertionError(
            f"output lists differ in length: {len(baseline)} vs {len(spec)}"
        )
    mismatches = []
    for b, s in zip(baseline, spec):
        if b["query_id"] != s["query_id"]:
            raise AssertionError(
                f"output order desync: {b['query_id']} vs {s['query_id']}"
            )
        if b["output_ids"] == s["output_ids"]:
            continue
        first_div = next(
            (i for i, (x, y) in enumerate(zip(b["output_ids"], s["output_ids"])) if x != y),
            min(len(b["output_ids"]), len(s["output_ids"])),
        )
        mismatches.append({
            "query_id": b["query_id"],
            "first_divergence": first_div,
            "len_baseline": len(b["output_ids"]),
            "len_spec": len(s["output_ids"]),
        })
    return {
        "n_prompts": len(baseline),
        "n_exact": len(baseline) - len(mismatches),
        "n_mismatch": len(mismatches),
        "all_exact": not mismatches,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_outputs(path: str, outputs: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")


def read_outputs(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_metrics(path: str, record: dict) -> None:
    """One JSON line per run (B7); the report stage reads only this file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")


def _git_commit() -> str | None:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or None


# ---------------------------------------------------------------------------
# vLLM adapter — the only vllm-importing code, built lazily by the CLI
# ---------------------------------------------------------------------------


class VLLMEngine:
    """Wraps vllm.LLM; satisfies the BenchEngine protocol.

    Token-id prompts in ({"prompt_token_ids": [...]}), token-id completions
    out — LLM.generate returns RequestOutputs in input order (offline API),
    and only detokenize=False requests are used, so nothing re-tokenizes.
    Engine kwargs follow the plan defaults: bf16, max_model_len capped,
    TP=1, no pipeline parallelism (incompatible with spec decode anyway).
    """

    def __init__(
        self,
        model: str,
        speculative_config: dict | None = None,
        dtype: str = "bfloat16",
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL,
        tensor_parallel_size: int = 1,
    ):
        import vllm  # fail loudly if missing (H100 host only)

        self.llm = vllm.LLM(
            model=model,
            speculative_config=speculative_config,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._vllm_version = vllm.__version__

    def generate(self, prompts: list[list[int]], params: dict) -> list[list[int]]:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=params.get("temperature", 0.0),
            max_tokens=params["max_tokens"],
            seed=params.get("seed"),
            detokenize=params.get("detokenize", False),
        )
        outs = self.llm.generate(
            [{"prompt_token_ids": p} for p in prompts], sp
        )
        return [list(o.outputs[0].token_ids) for o in outs]

    def info(self) -> dict:
        """Environment record for the metrics line (guarded — never fatal)."""
        out = {"vllm_version": self._vllm_version}
        try:
            import torch

            if torch.cuda.is_available():
                out["gpu"] = torch.cuda.get_device_name(0)
                out["gpu_count"] = torch.cuda.device_count()
        except Exception:  # pragma: no cover — info only
            pass
        return out

    def teardown(self) -> None:
        """Free the GPU before the next engine loads (B6: two engines,
        one at a time — ~31 GB of weights each)."""
        del self.llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_temperature(val: str) -> float:
    if val == "greedy":
        return 0.0
    t = float(val)
    if t < 0:
        raise ValueError(f"temperature must be >= 0, got {val}")
    return t


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="vLLM bench harness (plan §6.6): tokens/sec + outputs; "
        "exactness mode compares token ids across two runs (§4.4)"
    )
    ap.add_argument("records", help="frozen parquet or save_to_disk dir")
    ap.add_argument("--method", choices=["ar", "draft_model", "ngram"],
        default="ar")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct",
        help="target model (frozen)")
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--k", type=int, default=5, help="num_speculative_tokens")
    ap.add_argument("--ngram-min", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--temperature", default="greedy",
        help="'greedy' (=0.0) or a float like 1.0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=None,
        help="first N records in frozen order (deterministic subset)")
    ap.add_argument("--per-turn", action="store_true",
        help="TB-500: one prompt per assistant turn, teacher-forced on the "
        "ground-truth context (C6 protocol); default is one prompt per "
        "record (the turn-0 context)")
    ap.add_argument("--batch", type=int, default=1,
        help="prompts per engine call = fixed concurrency (B1)")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
        help="untimed warmup prompts before run 1 (B2)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--outputs-out", default=None, help="outputs JSONL (B7)")
    ap.add_argument("--metrics-out", default="results/metrics.jsonl",
        help="metrics append target (B7); '' disables")
    ap.add_argument("--exactness", action="store_true",
        help="§4.4 gate: run AR + this spec config, greedy batch 1, "
        "compare full token-id sequences")
    ap.add_argument("--compare-vs", default=None,
        help="exactness: reuse a prior AR outputs JSONL instead of "
        "re-running the AR engine")
    ap.add_argument("--exactness-out", default=None, help="exactness report JSON")
    args = ap.parse_args()

    if args.method == "draft_model" and not args.draft_model:
        raise SystemExit("--method draft_model needs --draft-model")
    temp = _parse_temperature(args.temperature)
    if args.exactness:
        if temp != 0.0:
            raise SystemExit("exactness requires --temperature greedy (§4.4)")
        if args.per_turn:
            raise SystemExit(
                "--exactness with --per-turn: turn-expanded outputs repeat "
                "query_ids (one entry per turn), which exactness_compare "
                "cannot pair up. §4.4 runs at record granularity."
            )
        args.batch, args.runs, args.warmup = 1, 1, 0  # B6: batch 1, one run

    def build_engine(spec_config: dict | None) -> "VLLMEngine":
        try:
            return VLLMEngine(
                args.model, spec_config, args.dtype, args.max_model_len,
                args.gpu_mem_util,
            )
        except ModuleNotFoundError as e:
            raise SystemExit(
                f"vLLM is not importable on this host: {e}\n"
                "(the harness is vLLM-free until engine construction — "
                "bench_vllm runs only on the H100 host; see plan §6.6)"
            ) from e

    spec_cfg = build_spec_config(
        args.method, args.draft_model, args.k,
        args.ngram_min, args.ngram_max, args.max_model_len,
    )
    prompts = load_prompts(args.records, args.limit, args.per_turn)
    params = {"max_tokens": args.max_new_tokens, "temperature": temp,
              "seed": args.seed}

    def bench_and_record(
        engine, method: str, spec_config: dict | None, outputs_out: str | None
    ) -> dict:
        """One engine's bench + metrics line + optional outputs write.
        Takes the run descriptor explicitly — the exactness path benches AR
        and spec in one command and must not mutate shared state."""
        m = run_bench(engine, prompts, params, args.batch, args.runs, args.warmup)
        rec = {
            "kind": "bench",
            "tag": method if not args.exactness else f"{method}_exactness",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit(),
            "dataset": args.records,
            "limit": args.limit,
            "per_turn": args.per_turn,
            "model": args.model,
            "spec_method": method,
            "spec_config": spec_config,
            "method_detail": (
                args.draft_model if method == "draft_model"
                else f"ngram[{args.ngram_min},{args.ngram_max}]"
                if method == "ngram" else "ar_only"
            ),
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_mem_util": args.gpu_mem_util,
            "outputs_file": outputs_out,
            "engine_info": getattr(engine, "info", lambda: {})(),
            **{k: v for k, v in m.items() if k != "outputs"},
        }
        if args.metrics_out:
            append_metrics(args.metrics_out, rec)
        if outputs_out:
            write_outputs(outputs_out, m["outputs"])
        return m

    if not args.exactness:
        engine = build_engine(spec_cfg)
        m = bench_and_record(engine, args.method, spec_cfg, args.outputs_out)
        print(
            f"{args.method} k={args.k} T={temp} batch={args.batch}: "
            f"{m['median_gen_tok_s']:.1f} gen tok/s "
            f"(median of {args.runs} runs, wall {m['median_wall_s']:.2f}s, "
            f"{m['n_prompts']} prompts)"
        )
        return

    # -- exactness mode (§4.4): greedy, batch 1, one run per side (B6) ----
    if args.compare_vs:
        baseline_outputs = read_outputs(args.compare_vs)
        if len(baseline_outputs) != len(prompts):
            raise SystemExit(
                f"--compare-vs file has {len(baseline_outputs)} outputs but "
                f"this run benches {len(prompts)} prompts — same records/"
                f"--limit (and no --per-turn) required"
            )
        ar_meta = {"reused_from": args.compare_vs}
    else:
        ar_engine = build_engine(None)
        ar_out = (
            str(Path(args.outputs_out).with_name(
                Path(args.outputs_out).stem + "_ar.jsonl"))
            if args.outputs_out else None
        )
        m = bench_and_record(ar_engine, "ar", None, ar_out)
        baseline_outputs = m["outputs"]
        ar_meta = {"wall_s": m["median_wall_s"]}
        ar_engine.teardown()

    engine = build_engine(spec_cfg)
    m = bench_and_record(engine, args.method, spec_cfg, args.outputs_out)
    spec_outputs = m["outputs"]
    engine.teardown()

    rep = exactness_compare(baseline_outputs, spec_outputs)
    rep.update({
        "ar": ar_meta,
        "spec_method": args.method,
        "spec_config": spec_cfg,
        "model": args.model,
        "dataset": args.records,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "git_commit": _git_commit(),
    })
    if args.exactness_out:
        p = Path(args.exactness_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2))
    if args.metrics_out:
        append_metrics(args.metrics_out, {"kind": "exactness", **rep})
    print(
        f"exactness ({args.method} vs AR, {rep['n_prompts']} prompts): "
        f"{rep['n_exact']} exact, {rep['n_mismatch']} mismatch"
    )
    for mm in rep["mismatches"][:5]:
        print(f"  qid {mm['query_id']}: first divergence at {mm['first_divergence']} "
              f"(len {mm['len_baseline']} vs {mm['len_spec']})")
    if not rep["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
