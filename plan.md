# Speculative Decoding for Tool-Calling Workloads — Project Plan

**One-line goal:** Make speculative decoding faster on tool-calling workloads by distilling a small draft model on-policy toward a frozen target — and characterize *where* the speedup comes from: structured tool-call JSON vs. free-form prose. The region-split acceptance analysis is this project's differentiator.

**Hard constraint:** 1× H100 (80 GB) for one day. Everything that can be done without the H100 (data prep, code, tests, dry runs) must be finished locally before the GPU window opens.

## 0. Executive summary

1. Serve frozen **Qwen2.5-Coder-14B-Instruct** as target; benchmark plain autoregressive (AR) decoding.
2. Run **untuned** speculative decoding with two draft candidates (Qwen2.5-0.5B-Instruct, 1.5B-Instruct) and a **no-model n-gram / prompt-lookup baseline**.
3. Pick the better draft candidate by a pre-registered rule, then **distill it on-policy** on ToolBench prompts with tools rendered in the target's exact chat format.
4. Re-benchmark: sweep draft length k and temperature, measure at batch size 1 and one larger batch, split acceptance by output region, and evaluate format transfer on xLAM.
5. Verify speculative outputs are token-identical to non-speculative greedy decoding.

---

## 1. Fixed choices and verified facts

### 1.1 Models

- **Target (frozen):** Qwen2.5-Coder-14B-Instruct, bf16 → ~28 GB weights. Never fine-tuned; the whole point is a frozen target.
- **Draft candidates:** Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct. Pick by untuned baseline results (rule in §8). Use *Instruct* SKUs, not base — the draft must imitate an instruct model, and an instruct draft starts much closer to the target distribution.
- **Optional 30-min add-on (P2):** also try Qwen2.5-Coder-0.5B/1.5B-Instruct as drafts. Same tokenizer family; may match Coder's JSON/code token distribution better than the general Qwen2.5 drafts.
- **Method:** on-policy distillation (DistillSpec-style). Not EAGLE-3 / DFlash etc. — no draft-head surgery, no extra target forward passes beyond data generation; fits the GPU and time budget. EAGLE needs training a static head against the frozen target and is far more engineering; DFlash is newer with less tooling.

### 1.2 Verified tokenizer facts (checked against HF configs — these remove a whole class of failure modes)

- The **entire Qwen2.5 family shares one tokenizer**: base / Instruct / Coder, 0.5B → 14B. Same special tokens (`<|im_start|>`, `<|im_end|>`, object/box/quad/vision tokens, etc.) in every member checked, including both draft candidates and the Coder target.
- Consequence: **no cross-vocabulary handling anywhere** — no `use_heterogeneous_vocab`, no token remapping between draft and target. Any Qwen2.5 model can draft for the Coder target directly.
- Target config: `vocab_size` 152,064 (embedding padded past tokenizer vocab 151,665), `max_position_embeddings` 32,768, bf16.
- Training data tokenized with the **target's** tokenizer/chat template is directly usable by any Qwen2.5 draft. The drafts' own templates are irrelevant — always render with the target's template.

### 1.3 Target's tool-call format (from the Coder-14B chat template; re-verify with a golden test, §6)

- Tools are announced in the **system** turn as `# Tools` + JSON schemas inside `<tools> ... </tools>`, followed by an instruction to emit `{"name": ..., "arguments": {...}}` inside `<tool_call> ... </tool_call>`.
- The assistant's tool call is therefore a short, highly regular JSON span between `<tool_call>` and `</tool_call>` tags (these render as ordinary text tokens, not special tokens).
- Tool results come back as **user** turns wrapped in `<tool_response> ... </tool_response>`.
- **Never hand-roll these strings.** All data formatting goes through `tokenizer.apply_chat_template(messages, tools=tools)` from the *target's* tokenizer, guarded by a golden snapshot test (§6).

### 1.4 Memory / serving budget (H100 80 GB)

- Weights: target ~28 GB + draft 1–3 GB → ~31 GB. Roughly half the GPU remains for KV cache and activations; batch 32 at 8–16k context fits comfortably. If OOM: cap `max_model_len`, or raise `gpu_memory_utilization` to 0.92, in that order.
- No tensor parallelism needed (single GPU, 14B).

### 1.5 Serving stack

- **vLLM** for all wall-clock numbers (offline `LLM` API; simplest and fastest). Pin the version in `requirements.txt` (draft-model speculative decoding is unsupported in very old vLLM; check the pinned version supports `speculative_config` on day 0).
- Speculative decoding via `speculative_config`:
  - draft model: `{"method": "draft_model", "model": <draft path>, "num_speculative_tokens": k, "max_model_len": 16384}`
  - n-gram: `{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5}`
  - Sampling params (`temperature` etc.) go in the request, not the speculative config.
  - No pipeline parallelism (incompatible anyway); TP is 1.
- **Two-track measurement (important practical detail):** vLLM gives trustworthy wall-clock but limited acceptance internals. α, τ, per-position acceptance, and region splits come from a small **instrumented HF rejection-sampling loop** (draft proposes k tokens, target verifies greedily) run on a ~100-prompt subset — HF is slower, so the subset is small; wall-clock numbers always come from vLLM. If the pinned vLLM logs per-step acceptance stats, use them as a cross-check of the instrumented loop.

---

## 2. Method: on-policy distillation, concretely

"On-policy" = the draft's training targets are **the target model's own generations**, not ChatGPT-written ToolBench answer trajectories and not human text. Concretely:

1. Take ToolBench conversation prefixes (user turn(s) + tool schemas, possibly earlier tool results) as contexts.
2. Let the **14B target** generate the assistant continuation on those contexts (greedy; see temperature note below).
3. Fine-tune the draft (full SFT, not LoRA — 0.5B/1.5B are small, full FT is fast on an H100, and the artifact stays a plain HF model vLLM can serve as a draft) on `(context, target_continuation)` pairs, **loss masked to assistant tokens only**.
4. Draft now imitates the target's token-level choices in exactly the format the target emits — which is what speculative acceptance measures.

**Two data-generation modes** (choose by GPU-time; A is the plan, B is stretch):
- **A. Target-regeneration (default):** use recorded ToolBench trajectories as scaffolding; for each assistant turn, keep the recorded *context* (user + tool schemas + prior turns incl. recorded tool results) but regenerate the assistant turn with the 14B target. Sequence-level on-policy for the target; replay of recorded API responses means no live API calls. Practical for 1 GPU-day.
- **B. Full self-play rollouts (stretch, only if time remains):** target generates a tool call → replay the recorded ToolBench API response for that call → target continues → etc. More on-policy (target conditions on its own earlier calls), costs more generation time.

**Temperature note (matches DistillSpec):** we evaluate at greedy and T=1.0. DistillSpec distills at the deployment temperature. Practical compromise for the time budget: generate distillation data at **greedy** (tool calling is usually deployed greedy), and if time remains generate a second pool at T=1.0 and distill a second checkpoint. Greedy first.

**SFT hyperparameters (starting point, tune only if loss diverges):** lr 1e-5 (0.5B) / 7e-6 (1.5B), cosine, 2–3 epochs, bf16, seq len 8192 with packing, global batch ~64 (grad accumulation). ToolBench prompts with many tool schemas are long — log the length distribution during local prep; left-truncate oldest turns if a sequence exceeds the cap.

---

## 3. Datasets

### 3.1 ToolBench (training + in-domain eval)

- **The original `ToolBench/ToolLLM-Instruct-196k` HF repo is gated/unavailable (401 as of 2026-09-26).** Use a mirror. Two verified candidates — inspect both during local prep and pick:
  - `Adorg/ToolBench` (Apache-2.0): raw per-query answer trajectories (`answer/G1_answer/<id>_ChatGPT_DFS_*.json`, etc.) — closest to the original release layout.
  - `Yhyu13/ToolBench_toolllama_G123_dfs` (Apache-2.0): already-processed SFT-format train/eval JSON (`toolllama_G123_dfs_train.json` / `_eval.json`) — pre-formatted conversations; less pipeline code, but verify its tool-rendering is complete (tool schemas present, DFS trajectories intact) before trusting it.
  - Fallback: the OpenBMB/ToolBench GitHub release (original data + retrieval corpus).
- **"10k subset" = 10k query–trajectory pairs** (not 10k tools). Sample across G1 (single-tool) / G2 (intra-cluster multi-tool, up to ~47 tools per query) / G3 (multi-cluster), dedupe near-identical queries, keep only trajectories that reached a final answer. Keep the G2/G3 mix — large tool-schema blocks in the prompt are exactly what makes tool-calling interesting for the n-gram baseline and the region analysis.
- **Split:** 10k train / 500 held-out eval (same distribution).
- **Formatting pipeline (the local work):**
  1. Parse trajectories → message lists (`system/user/assistant/tool` roles).
  2. Collect distinct tools per conversation; render via the target's `apply_chat_template(messages, tools=tools)`.
  3. Multi-turn → single training sequence with full chat history (loss on assistant turns only), as the plan says.
  4. Emit `datasets.Dataset` with `input_ids`, `labels`, and a `region` label map (assistant token spans, and within them: tool-call span vs. prose span) — this label map is what powers the §4.3 region analysis and should be produced during data prep, not patched in later.

### 3.2 xLAM (format-transfer eval only)

- `Salesforce/xlam-function-calling-60k` (CC-BY-4.0, single `xlam_function_calling_60k.json`, 60k examples, query + tools + answer). Confirmed available.
- **Eval-only** (never trained on) → sampling 500 random examples as prompts is fine; no leakage concern.
- Use xLAM's native tool-call format (its own system prompt + schema style), rendered through the target's chat template with *xLAM-style* tool descriptions — the point is a differently-formatted tool-calling distribution to test whether distillation gains transfer or were format-specific.

---

## 4. Metrics and evaluation protocol

All runs report: dataset (ToolBench-eval / xLAM), config (draft × tuned/untuned / n-gram / AR-only), k, temperature, batch, and raw tokens/sec. Every metric lands as JSON in `results/`; plots come later from the JSON, never hand-edited.

### 4.1 Acceptance metrics (instrumented HF loop, ~100–200 prompts)

- **α** per-token acceptance rate; **τ** mean accepted draft tokens per verification step. Define the convention up front in code: τ counts accepted draft tokens; the bonus token emitted after a rejection (the target's correction) and after a full acceptance counts separately as `bonus_rate`. Report both conventions in the README so numbers are comparable to other papers.
- **Per-position acceptance α_n (n = 1..k):** does the gain hold deeper into the draft, or is it all position-1 easy tokens? Compare curves pre/post distillation.

### 4.2 Wall-clock (vLLM, ~200–300 prompts)

- Tokens/sec and speedup vs. plain AR decoding, for: AR-only, untuned 0.5B, untuned 1.5B, distilled draft(s), n-gram.
- **Batch sensitivity is first-class:** measure at batch 1 and batch 8 and 32 (fixed concurrency, offline batched generate). Motivation is already in the plan: an EAGLE-3 reproduction found 2.3× at batch 4 degrading to roughly break-even at batch 32. Report the speedup-vs-batch curve for the best config and the baselines. If a bigger batch kills the speedup, that is a finding, not a failure — report it.
- Note vLLM's own caveat: logprobs/outputs can be slightly non-deterministic at larger batch; that's fine for wall-clock, and the exactness check (4.4) is run at batch 1.

### 4.3 Region-split acceptance — the differentiator

- Split generated tokens into **tool-call JSON region** (between `<tool_call>` and `</tool_call>`, incl. name vs. arguments sub-spans) vs. **free-form prose region** (everything else). Label tokens via offset mapping computed in the data/eval pipeline — decode speculatively, track which token indices fall in which span.
- Report α and τ **per region**. Hypothesis (from the agentic-serving result cited in the plan): acceptance is strongly bimodal — structured tool-call regions near 100%, prose sometimes below 10%.
- Extra cut worth 30 minutes: within the tool-call region, **argument keys / function names** (copyable from the schema in the prompt — n-gram's home turf) vs. **argument values** (must actually be predicted). This tells you exactly which tokens the distilled draft wins on that n-gram can't steal.

### 4.4 Correctness verification (gate for everything else)

- Greedy, batch 1, fixed seed: compare full **token id sequences** of speculative vs. non-speculative decoding on the eval set. Require 100% identical ids (and therefore identical lengths). Also verify the n-gram method passes.
- Sanity check of the harness before the GPU day, run locally: draft = target (same 0.5B model both roles) with greedy → α must equal exactly k... i.e. every draft token accepted, τ = k. If not, the instrumented loop is buggy.
- If vLLM mismatches at batch 1 (numerical nondeterminism), fall back to the instrumented HF loop for the exactness claim on a 50-prompt subset and report vLLM's behavior honestly as a separate observation.

### 4.5 Sweeps

- k ∈ {3, 5, 7} × temperature ∈ {greedy, T=1.0} at batch 1 (matching DistillSpec's setup).
- Batch sweep (1/8/32) only at greedy, only for the best draft config + AR + n-gram — keeps the grid affordable in one GPU day.
- Full grid at batch 1 for every config; the batch sweep is a targeted add-on.

### 4.6 Baseline: n-gram / prompt-lookup

- vLLM `ngram` method (no draft model; `num_speculative_tokens` 4, `prompt_lookup_min` 2, `prompt_lookup_max` 5; sweep the speculative-token count 3/5/7 same as drafts).
- Why it matters here: tool calls copy function names and argument keys verbatim from the schema in the prompt — precisely what prompt-lookup exploits. Reference points from the plan: suffix decoding 1.45× over baseline on code-heavy workloads, plain n-gram 1.10×. The distilled draft's job is to beat this *cheap* baseline; that is the real bar for the paper-grade claim.

---

## 5. GPU-day runbook (1× H100, ~24 h)

Priority-tagged so the day degrades gracefully: P0 must happen, P1 should, P2 only if ahead.

| Hours | Task | Priority |
|---|---|---|
| 0–1.5 | Env setup: clone repo, `pip install -r requirements.txt` (pinned vLLM), download target + 0.5B + 1.5B (+ Coder drafts if disk allows), smoke-test serve. Verify `speculative_config` works with this vLLM build. | P0 |
| 1.5–3 | Baselines: AR-only at batch 1/8/32; n-gram baseline; untuned 0.5B and 1.5B spec decode at k=5 greedy (batch 1). Exactness check (4.4). Compute untuned τ for both drafts. | P0 |
| 3–4 | **Decision point:** pick distillation candidate via the §8 rule. Full k/temp sweep for the untuned configs (batch 1). | P0 |
| 4–8 | Distillation data generation on GPU: 10k target regenerations (mode A). ~7M tokens aggregate; with continuous batching this is a few hours at most. Store as HF dataset + push to repo storage. | P0 |
| 8–11 | Draft SFT (0.5B ~1 h; 1.5B ~1.5–2 h incl. eval-after-each-epoch). | P0 |
| 11–14 | Re-benchmark distilled draft: batch 1 grid (k × temp), batch sweep, exactness re-check, region-split analysis (4.3), per-position curves. | P0 |
| 14–16 | xLAM transfer eval (4.1/4.2/4.3 on xLAM 500). | P1 |
| 16–18 | Second candidate distilled (the other draft size) if the first went smoothly → same battery. | P1 |
| 18–19 | Coder-draft add-on runs; T=1.0-distilled second checkpoint. | P2 |
| 19–22 | All plots from results JSON; README results tables; repo cleanup. | P0 |
| 22–24 | Buffer (model download failures, OOM debugging, reruns). | — |

Run everything through `scripts/run_*.sh` so the day is a sequence of typed commands, not decisions. Each run appends to `results/metrics.jsonl`; a `make report` regenerates all tables/plots from it.

---

## 6. Local pre-work (before renting the GPU — this is where the project is actually built)

1. **Repo skeleton** (§7), pinned `requirements.txt`, venv.
2. **ToolBench download + cleaning + formatting pipeline** → 10k train / 500 eval, target-template rendering, region label maps, length-distribution report. Unit tests on golden examples.
3. **xLAM prep** → 500 eval prompts in xLAM-native format.
4. **Instrumented speculative decoding loop** (draft proposes k, target verifies, greedy rejection sampling) with the §4.4 self-consistency test (draft=target ⇒ all accepted).
5. **vLLM bench harness** (`bench_vllm.py`): given (model, spec config, prompts, params) → tokens/sec + outputs; exactness mode comparing token ids across two runs.
6. **SFT training script** (`training/sft.py`) — tested end-to-end on a tiny model locally (e.g. random-init 0.5B or the real 0.5B for a few steps on CPU/MPS) so the GPU day never debugs training code.
7. **Dry-run the whole pipeline at toy scale** (tiny model as fake target, 20 prompts) — the full loop from raw ToolBench to metrics JSON must run green on CPU before the GPU is rented.

---

## 7. Repo layout

```
Speculative_Decoding/
├── plan.md, README.md, requirements.txt
├── configs/            # model paths, spec configs, sweep grids (JSON)
├── data/               # gitignored; raw/ + processed/
├── src/
│   ├── data_prep/      # toolbench_download, format_qwen, xlam_prep, build_distill_data
│   ├── serving/        # bench_vllm.py, spec_configs.py, instrumented_spec.py
│   ├── training/       # sft.py
│   └── analysis/       # metrics.py, region_split.py, per_position.py, plots.py
├── scripts/            # run_baselines.sh, run_sweep.sh, run_distill.sh, run_report.sh
├── results/            # metrics.jsonl, tables/, plots/   (committed)
└── tests/              # golden render tests, harness self-consistency tests
```

---

## 8. Pre-registered decisions and success criteria

Written down *before* the GPU day so choices aren't made after seeing results:

1. **Draft choice rule (decision point, hour ~4):** measure untuned wall-clock speedup vs. AR at k=5, greedy, batch 1 for both candidates. Distill the one with the higher untuned speedup. Tie-break toward 0.5B (cheaper to train and verify; more headroom). Distill the second candidate only as P1.
2. **Success criteria:**
   - **P0 (must):** exactness check passes; AR / n-gram / untuned-draft baselines all measured; τ, α, per-position and region-split analyses produced for at least one distilled draft.
   - **P1 (should):** distilled draft beats its untuned self on τ by ≥20% *and* beats the n-gram baseline on wall-clock at batch 1, greedy, on ToolBench eval.
   - **P2 (stretch):** gains hold at batch 8/32; xLAM transfer is positive; per-position curve shows improvement deep into the draft; second draft size distilled.
   - Negative results (e.g. n-gram wins, speedup collapses at batch 32) are reported as findings with the region-split explanation — that's still a publishable characterization.
3. **Eval sets frozen before the GPU day:** ToolBench 500, xLAM 500, fixed seeds, fixed prompt files committed to the repo.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Original ToolBench repo gated | Use verified mirrors (§3.1); inspect both during local prep; fallback to GitHub release. |
| vLLM version drift / spec-decode API changes | Pin exact version; `speculative_config` smoke test is the first GPU-hour task; keep the instrumented HF loop as the fallback measurement path. |
| Target hallucinates argument keys not in schema (distills bad habits into draft) | Spot-check 50 generations during data gen; drop trajectories with invalid tool calls (schema-validated); log the drop rate. |
| Draft never beats n-gram | That's a legitimate result — the region-split analysis (which tokens n-gram steals vs. which the draft must predict) is the deliverable either way. |
| Batch non-determinism confuses exactness check | Exactness at batch 1 only; wall-clock at all batches; report separately. |
| OOM at batch 32 with long ToolBench prompts (big schema blocks) | Cap `max_model_len` (16k), then `gpu_memory_utilization` 0.92; length distribution is measured locally in advance so this isn't a surprise. |
| GPU day slips on setup | Everything local (§6) is done first; day-0 checklist in the runbook is mechanical. |

---

## 10. Open questions

- **"E2E Networks"** in the original plan (under metrics): assumed here to mean end-to-end throughput (folded into §4.2). If it's the GPU *provider* (E2E Networks is an H100 rental option), it belongs in the runbook instead — clarify and move.
- Exact xLAM test split: the 60k file is a train set; since we use it eval-only, a random 500 slice suffices (no leakage), but if a canonical xLAM test set exists, prefer it.
- ToolBench mirror choice (raw trajectories vs. pre-processed SFT format) — decide during local prep after inspecting both.
- Whether to also distill a T=1.0 checkpoint (second data pool) — decide at hour 18 based on remaining time; greedy pool is the default.
