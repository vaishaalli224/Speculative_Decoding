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
│   ├── serving/          # (upcoming) bench_vllm, instrumented_spec
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

63 tests: golden template/region facts, xLAM conversion, ToolBench cleaning
(columnar conversations, `Action Input:` variants, JSON `true/false`
payloads, retry-draft blocks, truncated envelopes, retry user turns, Finish
conversion), freeze tamper-detection, and acceptance-metrics golden cases
(α/τ/bonus conventions, per-position scoring, region/sub-cut assignment,
bootstrap CIs, the §4.4 self-consistency event shape).
