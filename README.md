# Speculative Decoding for Tool-Calling Workloads

Distill a Qwen2.5-Coder-0.5B-Instruct draft toward a frozen
Qwen2.5-Coder-14B-Instruct target on tool-calling data, and characterize
*where* the acceptance gain comes from — structured tool-call JSON vs.
free-form prose, in-domain vs. multi-turn agentic transfer — with
losslessness and contamination checks that make the gain defensible.
Everything below is measured on one rented H100 (80 GB) in a single day,
against eval sets frozen and checksummed before the GPU was touched.

**Headline (k=5, greedy, batch 1):** distillation raises the draft's mean
accepted tokens per verification step τ from **3.32 → 4.07** on in-domain
xLAM-500 and **2.63 → 3.04** on held-out-tool multi-turn ToolBench-500
(non-overlapping bootstrap CIs), which shows up in vLLM wall-clock as
**2.22× → 2.45×** over plain autoregressive decoding — all with
**token-identical outputs** (exactness gates pass; the one 49/50 gate is
root-caused to bf16 near-ties, not the algorithm).

## Draft checkpoints (Hugging Face)

All four trained drafts are published — bf16 copies of the fp32 training
checkpoints, verified token-identical to them on 50 frozen prompts before
upload, with per-model READMEs carrying their measured τ/α/exactness:

| Checkpoint | Pipeline | xLAM τ / TB τ (k=5, greedy) |
|---|---|---|
| [vaishaalli/stage1-kd](https://huggingface.co/vaishaalli/stage1-kd) | Stage-1 KD, xLAM contexts | 4.01 / 2.65 |
| [vaishaalli/tb-kd](https://huggingface.co/vaishaalli/tb-kd) | KD, ToolBench contexts (ablation A) | 4.07 / 3.04 |
| [vaishaalli/stage2-gkd-mixed](https://huggingface.co/vaishaalli/stage2-gkd-mixed) | on-policy GKD, 1:1 TB:xLAM | 4.18 / 2.94 |
| [vaishaalli/stage2-gkd-tbonly](https://huggingface.co/vaishaalli/stage2-gkd-tbonly) | on-policy GKD, TB-only (final) | 4.05 / 3.04 |

Use any of them directly as a vLLM speculative-decoding draft:
`speculative_config = {"method": "draft_model", "model": "vaishaalli/tb-kd",
"num_speculative_tokens": 5, "max_model_len": 16384}`. The recommendation
is `tb-kd` — the off-policy checkpoint that matched on-policy GKD at a
fraction of the training cost (see finding 3 below).

---

## Project memo

### What we did and why

Speculative decoding accelerates a frozen target model by letting a cheap
draft propose k tokens that the target verifies in one forward pass. The
speedup is governed by one quantity: per-token acceptance α, the overlap
between draft and target distributions. So we trained the draft *on the
target's own distribution* rather than on gold text — three stages, each
adding one ingredient:

1. **Untuned baseline** — the off-the-shelf 0.5B draft (same tokenizer
   family; no cross-vocabulary handling needed).
2. **Stage-1 KD** — supervised distillation on the *target's own greedy
   generations* (top-20 logprobs per token) over xLAM function-calling
   contexts. KD toward the target, not SFT toward gold: acceptance is
   agreement with the target, and the target deviates from gold on a fifth
   of calls (below), so gold text would teach the draft tokens the target
   rejects.
3. **Ablation A (TB-KD)** — the identical KD recipe, but contexts drawn
   from ToolBench multi-turn conversations (including post-tool-observation
   states xLAM cannot supply). Pre-registered to answer: is *on-policy*
   machinery even needed, or do the *contexts* carry the gain?
4. **Stage-2 GKD (on-policy)** — the draft samples its own continuations,
   the frozen target scores them (reverse-KL), initialized from the best
   off-policy checkpoint per the pre-registered amendment. This is the
   "does on-policy add anything over the best off-policy result?" test.

Comparators were chosen so each number means something: **AR decoding**
(the denominator), **the same untuned draft** (training is the only changed
variable), **n-gram/prompt-lookup** (a free copying method — tool calls
copy names and keys verbatim from the schema in the prompt, so beating
*it*, not just AR, is the honest bar), and **two eval distributions**
(xLAM-500 in-domain; TB-500 with 416 held-out tools, 483/500 with real tool
observations — a transfer test, not a curve-fit).

### What we chose to measure, and why

- **τ and α (instrumented HF rejection-sampling loop, 100 frozen prompts,
  k=5, greedy).** Acceptance *is* the mechanism; vLLM exposes only
  wall-clock, so a small instrumented loop measures the mechanism while
  vLLM measures the outcome. Conventions (what counts as accepted, bonus
  tokens, per-position scoring) are pinned in golden tests before any GPU
  run.
- **Region-split acceptance** — α separately for tool-call JSON, prose,
  wrapper tags, and final answers, plus a name-vs-arguments sub-cut. This
  is the project's differentiator: it says *where* the gain lives and which
  tokens a copying heuristic could have stolen.
- **Wall-clock (vLLM, 200 prompts, median of 3)** — because τ alone can
  mislead: the draft's own k sequential forwards cost time (measured at
  65–76% of instrumented-loop time), so the user-visible number is
  tokens/sec, cross-checked against the τ prediction (predicted 2.15×,
  measured 2.22× for the untuned draft — 4% agreement).
- **Exactness** — full token-id comparison of speculative vs.
  non-speculative greedy decoding, batch 1. A speedup that changed outputs
  would be trivial to obtain and worthless.
- **Contamination controls** — held-out-tool eval carves by construction
  (rarest-function/rarest-tool, disjointness asserted and checksummed),
  zero query overlap between training and eval corpora.

What we deliberately did *not* measure (descoped under the one-day budget,
each with a stated reason): the full k×temperature grid (all rows compared
at identical settings — k=5, greedy, batch 1 — so decode hyperparameters
cannot confound a draft-vs-draft comparison; greedy is the deployed setting
for function calling); the full batch sweep (one batch-32 point retained,
since batch scaling is the known collapse mode for speculative decoding);
and the TVD-loss ablation. Cut list and descoping rationale are in
[plan.md](plan.md) §4.5/§8.

### Results

**Stage-wise acceptance** (τ = mean accepted draft tokens per step, max
5.0; pooled mean, k=5, greedy, 100 prompts; brackets = bootstrap 95% CI
over prompts of the per-prompt mean†).

| Draft | xLAM-500 τ | TB-500 τ (transfer) |
|---|---|---|
| Untuned Qwen2.5-Coder-0.5B | 3.32 [3.23–3.43] | 2.63 [2.54–2.77] |
| n-gram / prompt-lookup (no model) | 0.72 | 0.58 |
| Stage-1 KD (xLAM contexts) | 4.01 [4.17–4.40] | 2.65 [2.61–2.84] |
| TB-KD (TB contexts; ablation A) | **4.07** [4.19–4.42] | **3.04** [3.01–3.26] |
| Stage-2 GKD, 1:1 TB:xLAM mix | 4.18 [4.34–4.54] | 2.94 [2.97–3.23] |
| Stage-2 GKD, TB-only (final) | 4.05 [4.20–4.43] | 3.04 [3.02–3.28] |

**Per-step mechanics** (same instrumented runs; mean over ~10k verification
steps per row): nearly every step ends in a full acceptance plus the bonus
token (bonus_rate 0.95 on xLAM, 0.98 on TB), and trained-draft acceptance
*holds or rises* deep into the proposal — xLAM per-position αₙ
[0.91, 0.95, 0.95, 0.96, 0.97]; TB αₙ rises [0.80 → 0.90]. The draft's k
sequential forwards dominate instrumented-loop time: k=5 propose 74 ms vs
verify 40 ms (**draft time share 0.65**, identical on TB), k=9 127 ms vs
41 ms (0.76) — the acceptance ceiling is high while the *proposer* cost is
the binding constraint, which is why the next-step recommendation is a
cheaper head, not more agreement.

† The per-prompt mean sits above the pooled mean because longer
generations have slightly lower per-step acceptance; orderings and every
non-overlap claim hold under either convention.

**Wall-clock** (vLLM, 200 xLAM prompts, greedy, k=5, median of 3):

| Config (batch 1 unless noted) | gen tok/s | speedup vs AR |
|---|---|---|
| AR | 66.7 | 1.00× |
| AR, batch 8 / batch 32 | 238.5 / 582.1 | — |
| n-gram | 111.5 | 1.67× |
| Untuned draft | 148.0 | 2.22× |
| Stage-1 KD draft | 169.3 | 2.54× |
| Stage-2 (TB-only) draft | 163.8 | 2.45× |
| Stage-2 draft, batch 32 | ⏳ *pending* | ⏳ |

**Where the gain lives** (α by region, xLAM-500; the region map is produced
during data prep from the target's own rendering, never hand-typed):

| Draft | prose | tool-call JSON | wrapper tags | call name | call args |
|---|---|---|---|---|---|
| Untuned | 0.935 | 0.875 | 0.623 | 0.785 | 0.931 |
| Stage-1 KD | 0.941 | 0.948 | 0.878 | 0.934 | 0.956 |
| TB-KD | 0.947 | 0.952 | 0.889 | 0.946 | 0.956 |

On TB-500 the trained drafts' acceptance is **flat across every region**
(TB-KD: prose 0.854 / JSON 0.857 / tags 0.874 / final-answer 0.832;
s2tb nearly identical) — no bimodality, and importantly no low-acceptance
pocket left anywhere.

Three readings fall out of these tables:

1. **The plan's bimodal hypothesis was wrong, informatively.** We
   expected tool-call JSON near 1.0 and prose below 0.1; instead the
   untuned draft is *high everywhere except the cold start and wrapper
   tags* (α 0.623 on tags; first-step acceptance 0.03 — it opens with a
   markdown fence in 97/100 prompts where the target opens with the
   wrapper). Stage-1's entire in-domain gain is the cold-start fix (0.03 →
   0.82) plus tags (0.62 → 0.88) — one-token format competence, not
   diffuse distribution learning.
2. **Transfer is asymmetric — downward only.** Stage-1 (xLAM-only) has
   zero transfer to TB (2.63 → 2.65, inside CI): a format skill learned on
   a distribution whose opener is one token 73% of the time has nothing to
   attach to in TB's 6-way heterogeneous openers. The reverse direction
   transfers fully: TB-KD, trained only on TB, *ties the xLAM-trained
   draft in-domain* (4.07 vs 4.01) while beating it by +0.39 τ on TB.
   Distillation generalizes from the diverse distribution to the narrow
   one, not upward.
3. **Off-policy KD on the right contexts is a strong baseline — and
   suffices for greedy.** For greedy spec decoding, committed prefixes
   are always target-greedy states, so teacher-forced KD states match
   inference states almost exactly. GKD's theoretical edge (draft-visited
   states) is mainly proven for sampled decoding. Empirically, both
   on-policy variants landed *inside* TB-KD's CIs on the transfer eval —
   the mixed run at 2.94 [2.97–3.23], the TB-only run at **3.04 [3.02–3.28]
   vs TB-KD's 3.04 [3.01–3.26]**: identical point estimates, full CI
   overlap. Per the pre-registered §4.7A rule, that verdict is
   **"off-policy KD suffices for greedy spec decoding; the on-policy
   machinery is unnecessary complexity in this regime"** — and the region
   split explains why: trained-draft TB acceptance is flat 0.83–0.89
   across all regions, so there is no low-acceptance pocket for on-policy
   signal to fix. (The one caveat the τ-only view hides: TB-KD reached
   parity in *one hour of training*; GKD cost several more.)

n-gram's instrumented profile (position-1 α 0.26 rising to 0.88 — it
copies, it cannot predict; degrades 0.72 → 0.58 on TB's prose-heavy
thoughts) is the profile of everything a *learned* draft must beat without
spending a draft model's inference cost. It clears AR (1.67×) and nothing
else.

### Why the improvement is real, not an artifact

- **Not quality loss.** Exactness gates compare full token-id sequences of
  speculative vs. non-speculative greedy decoding at batch 1: untuned and
  n-gram **50/50**, TB-KD and both Stage-2 drafts **50/50**, Stage-1
  **49/50** — the four mismatches are root-caused to bf16 near-tie argmax
  flips between vLLM's batched-verify and incremental-AR kernels (two
  `' you'` insertions at near-tie positions), not to the algorithm; the
  algorithm-level claim is carried by the HF instrumented loop, whose
  self-consistency gate (draft = target ⇒ α = 1.0, τ = k every round) and
  forced-rejection exactness test pass on real weights. vLLM's residual
  nondeterminism is reported as an observation, per the pre-registered
  plan §9 response.
- **Not contamination.** Eval sets were frozen and sha256-checksummed
  before the GPU day, with held-out carves (206 xLAM functions, 416 TB
  tools) disjoint from training *by construction and by assertion*; zero
  query overlap between corpora in either direction; the 2.5% shared
  function names are generic (`age_calculator`) over different APIs.
  Headroom check: the target reproduces xLAM gold exactly on only 69.9%
  of calls (20.8% same-function-different-arguments) — its behavior is not
  recoverable from the dataset, so the draft cannot be memorizing eval
  answers it never saw and the target itself doesn't reproduce.
- **Not noise.** Every comparison is stated with bootstrap 95% CIs over
  prompts and the claim rule (non-overlap) was written down before the
  results; wall-clock is median-of-3. The gains cited above are
  non-overlapping under both τ conventions.
- **Not cherry-picking.** Decision rules were pre-registered in the plan
  (draft fixed to 0.5B before the rental; success criteria P0/P1/P2;
  ablation verdict rules before their results landed; the Stage-2 warm
  start was amended from ablation A's numbers *before* Stage-2 launched)
  — and the negative results are reported with the same prominence:
  Stage-1's zero transfer, the wrong bimodal hypothesis, GKD's
   non-separation from off-policy KD, and n-gram beating nothing but AR.
- **Not a prompt trick or copying.** The remaining unexplained alternative
  is that the gain is a promptable format skill; the promptability probe
  was queued, cut for time, and is reported as *open* under its
  pre-registered rule rather than silently dropped. The region split
  bounds the concern: trained-draft gains appear in call *names* and
  *argument values* alike (0.785→0.946 / 0.931→0.956), the latter being
  tokens that appear nowhere in the prompt to copy and that n-gram
  structurally cannot produce.
- **Reproducible end-to-end.** Frozen splits + manifest committed; ~250
  golden/unit tests pin every counting convention, chat rendering, and
  schema conversion before any GPU run; every number above comes from a
  committed JSONL/report artifact regenerated by `scripts/run_*.sh`; the
  stage-wise rows above are re-derivable from
  `results/events/*_report.json` without a GPU.

### What we'd do next

1. **Make the proposer cheaper, not just more agreeable.** At k=5 the
   draft's sequential forwards are 65% of instrumented-loop time (76% at
   k=9), and acceptance holds deep into the draft (untuned per-position α
   at k=9: 0.79 → 0.96, no collapse; τ 3.32 → 4.72). The next ceiling is
   an EAGLE-3-style head drafting from the target's hidden states, or
   AdaSPEC-style filtering of which tokens are worth training at all.
2. **Test the regime where on-policy should actually win.** The greedy
   result (off-policy KD suffices) matches theory: committed prefixes are
   target-greedy states. At T>0 the draft-visited-state argument returns —
   that's where GKD needs its day in court.
3. **Close the open promptability question** (untuned draft + opener
   prompt, one instrumented pass) and resurrect the untuned-1.5B baseline
   measurement — both cheap, both pre-registered fallbacks.
4. **Ship the draft artifact** (HF Hub upload pending) and re-run the
   battery on a second target family to check the transfer asymmetry
   generalizes.

---

## Reproducibility guide

**The fine print on the pending cells:** ⏳ marks the two wall-clock cells
still filling from the in-flight vLLM phase on the TB-only Stage-2 draft
(`results/vllm/s2tb_*` — b1 xLAM row mid-run, then b32 and TB transfer
wall-clock); every acceptance number, exactness gate, and mechanics metric
is final and committed. The memo tables are generated from committed report
JSONs; they are never hand-edited after the fact.

The sections below are the build/usage docs for every component the memo
above relies on.

---

## Repo layout

```
Speculative_Decoding/
├── plan.md, README.md, requirements.txt
├── frozen/               # COMMITTED: eval parquets + context indices + sha256 manifest
├── data/                 # gitignored: raw/ (HF downloads) + processed/ (rendered splits)
├── src/
│   ├── data_prep/        # download, xlam_prep, toolbench_clean, freeze_splits,
│   │                     # build_distill_data (Stage-1 KD dataset)
│   ├── serving/          # instrumented_spec + proposers + bench_vllm + spec_configs
│   │                     # + gen_stage1 (Stage-1 target generation)
│   ├── training/         # (upcoming) kd_warmstart, onpolicy_gkd
│   └── analysis/         # eval_acceptance (done); plots (upcoming)
├── scripts/              # run_baselines.sh, run_sweep.sh, run_tau.sh, setup_day0.sh,
│                         # run_stage1_datagen.sh (GPU day; more upcoming)
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

`--device cuda:0` on the GPU host (default keeps CPU/MPS for the local
§4.4 checks). The τ pipeline is scripted on the GPU day
(`scripts/run_tau.sh`: both untuned drafts + n-gram → events + analyzer
reports). The draft is fixed to Coder-0.5B (§8.1); `scripts/pick_draft.py`
is retired from the runbook (kept as a results-summary utility, with its
golden tests).

Local §4.4 sanity checks (CPU fp32, tiny models — run before the GPU day;
auto-skip unless the 0.5B weights are cached or `SPEC_REALMODELS=1`):

```bash
SPEC_REALMODELS=1 .venv/bin/python -m pytest tests/test_spec_realmodels.py
# self-consistency: draft = target ⇒ α = 1.0, τ = k every round
# exactness: loop output token-identical to plain greedy generate()
#           (non-Coder 0.5B draft forces real rejections → crop path exercised)
```

## vLLM bench harness (plan.md §6.6 — done)

`src/serving/bench_vllm.py` is the wall-clock instrument (§4.2): given
(model, spec config, prompts, params) → tokens/sec + outputs, plus an
exactness mode that compares full token-id sequences of speculative vs.
non-speculative greedy decoding (§4.4). Same layering as the loop: a
vLLM-free/torch-free core — chunked batch semantics, timed sweeps with
medians, outputs/metrics IO, the exactness comparator — golden-tested
locally (39 tests), and a thin `VLLMEngine` adapter that imports vLLM
only on the H100. `src/serving/spec_configs.py` pins the plan's two
`speculative_config` shapes verbatim (draft-model and n-gram) with
validation; sampling params never live in the engine config. GPU host
installs `requirements-h100.txt` (vllm==0.30.0, torch==2.13.0 — vLLM's
own torch pin; the local venv stays on requirements.txt).

```bash
# GPU day — everything through scripts/ (plan §5):
bash scripts/setup_day0.sh             # hours 0–1.5: clone, venv+pins, raw data,
                                        # freeze verify, model downloads, vLLM
                                        # spec-config smoke test (both shapes)
bash scripts/run_baselines.sh          # AR b1/8/32, n-gram, untuned drafts,
                                        # exactness gates (§4.4)
bash scripts/run_tau.sh                # untuned τ: both drafts + n-gram through
                                        # the instrumented loop + analyzer
python scripts/pick_draft.py           # §8.1 decision point, mechanical
bash scripts/run_sweep.sh <draft-path> [tag]   # k × T grid, batch sweep, TB-500
# or one config directly:
python -m src.serving.bench_vllm frozen/xlam_eval.parquet --method draft_model \
    --draft-model <draft> --k 5 --temperature greedy --batch 1 --runs 3 \
    --limit 200 --outputs-out results/vllm/draft_k5.jsonl
python -m src.serving.bench_vllm frozen/xlam_eval.parquet --exactness \
    --method draft_model --draft-model <draft> --k 5 --limit 50 \
    --exactness-out results/exactness/draft_k5.json   # exit 1 on mismatch
```

`--per-turn` expands TB-500 records to teacher-forced per-assistant-turn
prompts (the loop's C6 protocol; turn starts imported from the analyzer,
so the two instruments cannot desync) — the transfer wall-clock then
measures the same agentic states the acceptance analysis scores. Every
run appends one JSON line to `results/metrics.jsonl`; the report stage
reads only that file.

`frozen/` is the GPU-day entry point (plan.md §6.3): xLAM-500 + TB-500 eval
parquets, the Stage-1 (5k seeded xLAM sample) / Stage-2 (45k TB prefixes +
52.8k xLAM pool) context index files, and a sha256 manifest that also
checksums the raw arrow files. **GPU-day step 0**:

```bash
.venv/bin/python -m src.data_prep.freeze_splits verify
```

re-checks every checksum and the held-out disjointness properties from
`frozen/` alone — it re-derives nothing.

## Stage-1 KD data generation (plan.md §6.7 first half — done)

`src/serving/gen_stage1.py` generates the frozen target's greedy
continuations on the 5,000 frozen Stage-1 xLAM contexts with top-20
logprobs per generated token (the teacher distributions for the KD
warm-start); `src/data_prep/build_distill_data.py` validates every tool
call against its own schema (plan §9 mitigation — drop rate logged,
never repaired) and assembles the KD dataset with the repo's universal
record schema plus `gen_logprob_token_ids/values`. Same layering as the
bench: vLLM-free/torch-free core golden-tested locally, thin `VLLMGenEngine`
(H100 only) and `HFGenEngine` (local dry runs) adapters. Conventions
E1–E5 / G1–G7 pinned in the two module docstrings; the committed artifact
is the generation JSONL (the assembled dataset is derived and gitignored).

```bash
# local sanity (no GPU): render + inspect the first 20 Stage-1 prompts
.venv/bin/python -m src.data_prep.build_distill_data prompts --limit 20

# GPU day (runbook hours 3.5–4.5): smoke → full 5k generation → assemble
bash scripts/run_stage1_datagen.sh
# or directly:
python -m src.serving.gen_stage1 --engine vllm \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --logprobs 20 \
    --out results/stage1/stage1_target_gen.jsonl
python -m src.data_prep.build_distill_data assemble \
    results/stage1/stage1_target_gen.jsonl --out data/processed/stage1_kd
```

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

191 tests (torch-free suite): golden template/region facts, the instrumented loop's C1–C7 convention/multi-turn/n-gram golden cases, xLAM conversion, ToolBench cleaning
(columnar conversations, `Action Input:` variants, JSON `true/false`
payloads, retry-draft blocks, truncated envelopes, retry user turns, Finish
conversion), freeze tamper-detection, and acceptance-metrics golden cases
(α/τ/bonus conventions, per-position scoring, region/sub-cut assignment,
bootstrap CIs, the §4.4 self-consistency event shape). The instrumented
loop adds 27 torch-free golden tests (event conventions C1–C7, budget
capping, teacher-forced turn transitions, n-gram proposer + PAD fallback)
and 3 real-model §4.4 tests (self-consistency: draft=target ⇒ α=1.0, τ=k;
exactness: loop output == plain greedy generate(), token-identical). The
bench harness adds 39 vLLM-free golden tests (B1–B7: chunked batching,
warmup exclusion, wall/rate medians, gen-vs-prompt token split, run-1
outputs, exactness verdicts with first-divergence, per-turn TB-500
expansion against the analyzer's own turn-start helper, spec-config
shapes and validation). The Stage-1 data path adds 43 golden + 3
real-model tests (E1–E5 EOS normalization + logprob fidelity + frozen
order; G1–G7 payload spans, schema-validation matrix, drop accounting,
record/regions/logprob parallelism, real frozen-index prompts == rendered
train-pool slices; HFGenEngine end-to-end with the cached Coder-0.5B
stand-in — which empirically corrected the EOS-inclusion assumption).
