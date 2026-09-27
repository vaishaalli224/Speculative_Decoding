# Speculative_Decoding

Speculative decoding for tool-calling workloads: distill a small Qwen2.5-Coder
draft on-policy toward the frozen Qwen2.5-Coder-14B-Instruct target, and
characterize where the speedup comes from via region-split acceptance
(tool-call JSON vs. free-form prose). Plan: [plan.md](plan.md). Hard
constraint: everything doable without the H100 is done locally first (§6).

## Repo layout

```
Speculative_Decoding/
├── plan.md, README.md, requirements.txt
├── frozen/               # COMMITTED: eval parquets + context indices + sha256 manifest
├── data/                 # gitignored: raw/ (HF downloads) + processed/ (rendered splits)
├── src/
│   ├── data_prep/        # download, xlam_prep, toolbench_clean, freeze_splits
│   ├── serving/          # instrumented_spec (done), proposers; bench_vllm (upcoming)
│   ├── training/         # (upcoming) kd_warmstart, onpolicy_gkd
│   └── analysis/         # (upcoming) metrics, region_split, plots
├── scripts/              # (upcoming) run_*.sh
├── results/               # metrics.jsonl + tables/plots (committed)
└── tests/                # golden render/clean/freeze tests
```

## Data pipeline (local pre-work, plan.md §6.1–6.3 — done)

Reproduce from scratch (Python 3.12 venv, pinned requirements.txt):

```bash
.venv/bin/python src/data_prep/download_datasets.py     # raw HF datasets -> data/raw/
.venv/bin/python -m src.data_prep.xlam_prep carve       # xLAM rarest-function carve
.venv/bin/python -m src.data_prep.xlam_prep build --split eval
.venv/bin/python -m src.data_prep.xlam_prep build --split train
.venv/bin/python -m src.data_prep.toolbench_clean clean   # 187,542 -> 47,870 clean convs
.venv/bin/python -m src.data_prep.toolbench_clean carve    # TB-500 / 45,023 prefix pool
.venv/bin/python -m src.data_prep.toolbench_clean build --split eval
.venv/bin/python -m src.data_prep.toolbench_clean build --split prefixes
.venv/bin/python -m src.data_prep.freeze_splits freeze     # rebuild frozen/ + manifest
.venv/bin/python -m pytest tests/                          # 63 tests
```

## Acceptance evaluation (plan.md §6.4 — done)

`src/analysis/eval_acceptance.py` turns the instrumented loop's event stream
(JSONL, one object per verification step) into every §4.1–4.3 metric: α, τ,
bonus_rate, per-position α_n, region-split α (prose / tool-call JSON / tag
/ final answer), the name-vs-arguments sub-cut, bootstrap 95% CIs, and
per-prompt records:

```bash
.venv/bin/python -m src.analysis.eval_acceptance events.jsonl \
    --records data/processed/toolbench/eval --subcut --k 5 \
    --out results/report.json --per-prompt results/per_prompt.jsonl
```

Built and golden-tested before the loop exists: the event schema and the
α/τ/bonus counting conventions are pinned in the module docstring and in
`tests/test_eval_acceptance.py` (hand-computed reference cases). Multi-turn
records carry a `turn` field per event; the cursor resets at each
assistant-turn boundary (derived from the record's labels).

## Instrumented speculative-decoding loop (plan.md §6.5 — done)

`src/serving/instrumented_spec.py` is the measurement half of that contract:
a torch-free core state machine (greedy rejection sampling, Leviathan-style)
driven by a **Proposer** protocol — an HF draft model *or* the n-gram /
prompt-lookup matcher (`src/serving/proposers.py`) — and a Verifier
(the frozen target). One JSON event per verification step, exactly the
schema above. The n-gram proposer keeps the §4.3 comparison honest:
region-split acceptance for the *cheap* baseline comes from the same
instrumented pipeline as the drafts, which vLLM cannot provide. A no-match
round emits a sacrificial PAD token so step semantics stay uniform.

Pinned conventions (golden-tested, C1–C7 in the module docstring): rejection
masks never carry a True after a False and events carry only the *scored
prefix* of a proposal; EOS ends a turn with no bonus past it; the proposal
is capped at the remaining token budget first and the bonus is dropped if
it would overshoot `max_new_tokens` (outputs stay comparable to plain
`generate()`); multi-turn records are generated per assistant turn with
teacher-forced turn transitions.

```bash
# n-gram baseline over the first 100 frozen TB-500 records:
.venv/bin/python -m src.serving.instrumented_spec frozen/tb_eval.parquet \
    --proposer ngram --target-model Qwen/Qwen2.5-Coder-14B-Instruct \
    --k 5 --limit 100 --dtype bfloat16 \
    --out results/events/ngram_tb_k5.jsonl \
    --outputs-out results/events/ngram_tb_k5_outputs.jsonl \
    --meta-out results/events/ngram_tb_k5_meta.json
.venv/bin/python -m src.serving.instrumented_spec frozen/xlam_eval.parquet \
    --proposer draft --draft-model <draft-path-or-id> \
    --target-model Qwen/Qwen2.5-Coder-14B-Instruct --k 5 --limit 100 \
    --out results/events/draft_k5.jsonl --outputs-out ... --meta-out ...
```

Local §4.4 sanity checks (CPU fp32, tiny models — run before the GPU day;
auto-skip unless the 0.5B weights are cached or `SPEC_REALMODELS=1`):

```bash
SPEC_REALMODELS=1 .venv/bin/python -m pytest tests/test_spec_realmodels.py
# self-consistency: draft = target ⇒ α = 1.0, τ = k every round
# exactness: loop output token-identical to plain greedy generate()
#           (non-Coder 0.5B draft forces real rejections → crop path exercised)
```

`frozen/` is the GPU-day entry point (plan.md §6.3): xLAM-500 + TB-500 eval
parquets, the Stage-1 (5k seeded xLAM sample) / Stage-2 (45k TB prefixes +
52.8k xLAM pool) context index files, and a sha256 manifest that also
checksums the raw arrow files. **GPU-day step 0**:

```bash
.venv/bin/python -m src.data_prep.freeze_splits verify
```

re-checks every checksum and the held-out disjointness properties from
`frozen/` alone — it re-derives nothing.

## Datasets (roles flipped 2026-09-26, plan.md §8.0)

- **xLAM** (in-domain train + eval): 57,794-example training pool (96.3% of
  60k), 500-example eval via rarest-function carve (206 held-out functions).
- **ToolBench** (agentic transfer eval + Stage 2 prefixes): 47,870 clean
  `give_answer` conversations from the mirror's 187,542 (25.5%); TB-500 eval
  via greedy rare-tool carve (416 held-out tools, 483/500 with real tool
  observations); 45,023-conversation Stage-2 prefix pool (94.1%).
- Rendering goes through the target's `apply_chat_template` only
  (never hand-rolled); tool-call tags are probe-derived, not hand-typed.
  Region codes per token: 0=context / 1=assistant prose / 2=tool-call JSON /
  3=tag / 4=final answer.

## Testing

93 tests: golden template/region facts, the instrumented loop's C1–C7 convention/multi-turn/n-gram golden cases, xLAM conversion, ToolBench cleaning
(columnar conversations, `Action Input:` variants, JSON `true/false`
payloads, retry-draft blocks, truncated envelopes, retry user turns, Finish
conversion), freeze tamper-detection, and acceptance-metrics golden cases
(α/τ/bonus conventions, per-position scoring, region/sub-cut assignment,
bootstrap CIs, the §4.4 self-consistency event shape). The instrumented
loop adds 27 torch-free golden tests (event conventions C1–C7, budget
capping, teacher-forced turn transitions, n-gram proposer + PAD fallback)
and 3 real-model §4.4 tests (self-consistency: draft=target ⇒ α=1.0, τ=k;
exactness: loop output == plain greedy generate(), token-identical).
