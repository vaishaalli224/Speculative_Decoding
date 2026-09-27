# Speculative Decoding for Tool-Calling Workloads — Project Plan
**Repository should be completely reproducible**
**One-line goal:** Make speculative decoding faster on tool-calling workloads by distilling a small draft model on-policy toward a frozen target — and characterize *where* the speedup comes from: structured tool-call JSON vs. free-form prose. The region-split acceptance analysis is this project's differentiator.

**Hard constraint:** 1× H100 (80 GB) for one day. Everything that can be done without the H100 (data prep, code, tests, dry runs) must be finished locally before the GPU window opens.

## 0. Executive summary

1. Serve frozen **Qwen2.5-Coder-14B-Instruct** as target; benchmark plain autoregressive (AR) decoding.
2. Run **untuned** speculative decoding with two draft candidates (Qwen2.5-Coder-0.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct) and a **no-model n-gram / prompt-lookup baseline**.
3. Pick the better draft candidate by a pre-registered rule, then distill it in two stages with tools rendered in the target's exact chat format: a short **supervised KD warm-start on xLAM** (clean, single-turn function calling), then **on-policy distillation (DistillSpec/GKD-style) over mixed xLAM + ToolBench agentic contexts** where the draft samples and the target scores.
4. Re-benchmark: sweep draft length k and temperature, measure at batch size 1 and one larger batch, split acceptance by output region, and evaluate transfer on ToolBench multi-turn trajectories (unseen tools). Dataset roles were flipped 2026-09-26 pre-GPU-day — xLAM is in-domain, ToolBench is the transfer eval (see §8.0).
5. Verify speculative outputs are token-identical to non-speculative greedy decoding.

---

## 1. Fixed choices and verified facts

### 1.1 Models

- **Target (frozen):** Qwen2.5-Coder-14B-Instruct, bf16 → ~28 GB weights. Never fine-tuned; the whole point is a frozen target.
- **Draft candidates:** Qwen2.5-Coder-0.5B-Instruct and Qwen2.5-Coder-1.5B-Instruct. Pick by untuned baseline results (rule in §8). Use *Instruct* SKUs, not base — the draft must imitate an instruct model, and an instruct draft starts much closer to the target distribution. Coder drafts match the Coder target's JSON/code token distribution.
- **Method:** two-stage distillation — supervised KD warm-start, then on-policy distillation (DistillSpec-style). Not EAGLE-3 / DFlash etc. — no draft-head surgery; fits the GPU and time budget. The target is used for Stage 1 data generation and for scoring draft samples during Stage 2 (forward passes only, no gradients). EAGLE needs training a head from scratch against the frozen target and is far more engineering; DFlash is newer with less tooling.

### 1.2 Verified tokenizer facts (checked against HF configs — these remove a whole class of failure modes)

- The **entire Qwen2.5 family shares one tokenizer**: base / Instruct / Coder, 0.5B → 14B. Same special tokens (`<|im_start|>`, `<|im_end|>`, object/box/quad/vision tokens, etc.) in every member checked, including both draft candidates and the Coder target.
- Consequence: **no cross-vocabulary handling anywhere** — no `use_heterogeneous_vocab`, no token remapping between draft and target. Any Qwen2.5 model can draft for the Coder target directly.
- Target config: `vocab_size` 152,064 (embedding padded past tokenizer vocab 151,665), `max_position_embeddings` 32,768, bf16.
- **Embedding padding differs by size:** the small drafts are expected to pad to 151,936 vs. the target's 152,064 (verify in configs during local prep). Irrelevant for token-level verification in vLLM, but the KD/GKD losses compare logit tensors, so **slice both draft and target logits to the real tokenizer vocab (151,665)** before computing any divergence.
- Training data tokenized with the **target's** tokenizer/chat template is directly usable by any Qwen2.5 draft. The drafts' own templates are irrelevant — always render with the target's template.

### 1.3 Target's tool-call format (from the Coder-14B chat template; re-verify with a golden test, §6)

- Tools are announced in the **system** turn as `# Tools` + JSON schemas inside `<tools> ... </tools>`, followed by an instruction to emit `{"name": ..., "arguments": {...}}` inside `<tool_call> ... </tool_call>`.
- The assistant's tool call is therefore a short, highly regular JSON span between `<tool_call>` and `</tool_call>` tags (these render as ordinary text tokens, not special tokens).
- Tool results come back as **user** turns wrapped in `<tool_response> ... </tool_response>`.
- **Never hand-roll these strings.** All data formatting goes through `tokenizer.apply_chat_template(messages, tools=tools)` from the *target's* tokenizer, guarded by a golden snapshot test (§6).

### 1.4 Memory / serving budget (H100 80 GB)

- Weights: target ~28 GB + draft 1–3 GB → ~31 GB. Roughly half the GPU remains for KV cache and activations; batch 32 at 8–16k context fits comfortably. If OOM: cap `max_model_len`, or raise `gpu_memory_utilization` to 0.92, in that order.
- **Training (Stage 2):** frozen target ~28 GB + draft full FT with AdamW (~8 GB for 0.5B, ~24 GB for 1.5B) + activations. 0.5B is comfortable; 1.5B is tight but fits with gradient checkpointing and small micro-batches. Compute losses on assistant tokens only to keep 152k-vocab logits small.
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

## 2. Method: two-stage distillation (DistillSpec-style), concretely

"On-policy" = the draft learns from **its own sampled continuations**, scored token-by-token by the frozen target. This matches inference, where the target verifies the draft's own guesses, and directly targets acceptance: per-token acceptance equals the overlap Σ min(p, q) between target and draft distributions.

**Stage 1 — supervised KD warm-start (short):**
1. Take ~5k xLAM training-pool examples (query + tool schemas) as contexts.
2. Let the **14B target** generate the assistant continuation greedily in vLLM, saving the **top-20 logprobs per generated token**.
3. Fine-tune the draft (full FT, not LoRA — 0.5B/1.5B are small, full FT is fast on an H100, and the artifact stays a plain HF model vLLM can serve as a draft) with a KL loss against the target's top-k distribution, **loss masked to assistant tokens only**.
4. Purpose: get the draft producing sensible tool-call states so Stage 2's feedback is useful from the first step.

**Stage 2 — on-policy distillation (main result):**
1. Use mixed contexts: remaining xLAM examples + ToolBench clean-trajectory prefixes (46k pool, §3.2). No target generations needed.
2. The **draft samples** an assistant continuation; the **frozen 14B scores it** in one forward pass.
3. Train the draft to minimize divergence between its distribution and the target's on those draft-generated tokens, loss on assistant tokens only.
4. Mix in a fraction of Stage 1's fixed data (GKD's mixing ratio) to stabilize early training.
5. Implementation: TRL `GKDTrainer` (student sampling, frozen teacher, mixing ratio, forward/reverse KL and JSD). Default divergence: JSD or forward KL. TVD (directly tied to acceptance) is a custom-loss ablation, P2.
6. Cap contexts at ~4–6k tokens (trim tool lists and oldest turns) — draft HF generation is the Stage 2 bottleneck and scales with context length. xLAM prompts are naturally short (median ~1.1k chars); the cap binds only for ToolBench prefixes.
7. Checkpoint every hour with a quick τ check on a small eval subset.

**Temperature note:** we evaluate at greedy and T=1.0. Stage 1 data is generated greedy (tool calling is usually deployed greedy). In Stage 2, the draft samples at T=1.0 so it visits a spread of its own states; gains are reported separately for greedy and T=1.0 evaluation, since they will differ.

**Ablation this produces for the memo:** untuned → Stage 1 → Stage 2, on every metric.

**Hyperparameters (starting point, tune only if loss diverges):** lr 1e-5 (0.5B) / 7e-6 (1.5B), cosine, bf16, gradient checkpointing. Stage 1: 1–2 epochs, seq len 8192 with packing, global batch ~64 (grad accumulation). Stage 2: small micro-batches, max new tokens ~512 per draft sample. xLAM prompts are short (median ~1.1k chars); ToolBench prefixes are long (sys+user median ~5.8k chars, p95 ~11k) — log the length distribution of the actual mix during local prep; left-truncate oldest turns if a sequence exceeds the cap.

---

## 3. Datasets

**Roles (flipped 2026-09-26, pre-GPU-day — see §8.0):** xLAM is the **in-domain training + eval** set; ToolBench is the **agentic multi-turn transfer eval** and supplies extra Stage 2 contexts. Rationale, measured on the downloaded data: xLAM is clean (100% usable, OpenAI-style schemas, answers already in the target's `{"name", "arguments"}` shape), 5× shorter prompts (median ~1.1k chars vs ~5.8k sys+user), zero query overlap with ToolBench, and its brevity buys several times more distillation per GPU-hour in the Stage 1/2 bottleneck steps. ToolBench's mirror is ChatGPT-3.5-era trajectory data with ~73% incomplete/failure conversations — good for testing transfer to hard agentic states, wasteful as the primary training source. The flip also sharpens the science: Stage 1 (xLAM-only) → ToolBench eval is a pure single-turn→multi-turn transfer measurement; Stage 2 (mixed contexts) shows how much on-policy agentic data closes the gap.

### 3.1 xLAM (training + in-domain eval)

- `Salesforce/xlam-function-calling-60k` (CC-BY-4.0, 60k examples: `query`, `tools` = OpenAI-style JSON schema strings, `answers` = JSON tool calls; 52.6% multi-call, median 3 tools/example, max 8; 3,605 distinct function names; 58,146 distinct queries after normalization). Downloaded to `data/raw/xlam/`.
- **Single-turn** — no post-observation states. This is a known limitation, addressed by the Stage 2 ToolBench mix (§3.2).
- **Split by tool, not randomly — and by *rare* tools specifically.** Functions repeat heavily (median 26 examples/function); a held-out-tool carve from a *random* 500 would exclude 71.8% of training examples. Measured design (validated 2026-09-26): rank examples by summed function frequency, take the 500 rarest → **206 held-out functions**, leaving **57,794 training-pool examples (96.3%)** with the held-out-tool property verified by assertion. Eval prompts are short (median 468 chars, p95 ~1k).
- **Usage:** ~5k training-pool examples for Stage 1 (target generations); the remainder (minus nothing else — the exclusion is already the whole carve) as Stage 2 on-policy contexts.
- **Formatting:** minimal — parse `tools`/`answers` JSON, render through the target's `apply_chat_template(messages, tools=tools)` with a system prompt stating the xLAM-style tool-call instruction. Answers become assistant tool-call turns; 52.6% multi-call answers become single assistant turns with parallel calls.

### 3.2 ToolBench (agentic transfer eval + Stage 2 context pool)

- **Mirror: `tuandunghcmut/toolbench-v1`** (Apache-2.0) — the original `ToolBench/ToolLLM-Instruct-196k` repo is gated (401). This mirror is **pre-processed ShareGPT-style** (`conversations` with `from`/`value`; `id` = query text), *not* raw answer trajectories. Downloaded to `data/raw/toolbench_default/` (187,542 train + 762 validation) and `data/raw/toolbench_benchmark/` (official G1/G2/G3 instruction subsets: 200/200/100).
- **Format facts (verified on a 3k-example scan):**
  - Tool schemas are a Python-literal list embedded at the end of the system prompt after `"…following APIs: "` — parse with `ast.literal_eval(sys_msg.split('APIs:', 1)[1].strip())` (works 100%). Entries have `name`/`description`/`parameters` with `required`+`optional` lists → must be converted to OpenAI-style schemas before `apply_chat_template(tools=…)`.
  - Assistant turns are ToolLLaMA-style `Thought:/Action:/Action Input:` text (uniform across all sampled turns) → must be converted to the target's `{"name": …, "arguments": {…}}` tool-call format, with `Thought:` kept as prose.
  - `function` turns hold results as `{"error": …, "response": …}` strings → become tool-response user turns.
  - The built-in `Finish` tool (with `return_type` `give_answer`/`give_up_and_restart`) is part of every API list.
- **Filters (measured):** 74,631/187,542 end with `Action: Finish`; **50,147 end with `give_answer`** (plan's "reached a final answer" filter → keeps ~27%); ~5% have action names not in the API list (hallucinated → drop; only the *executed* call is checked — retry-draft blocks in multi-block turns are never executed and never checked); ~2% near-duplicate queries (dedupe on normalized query prefix). After all filters + dedupe: **47,870 clean multi-turn conversations** (2026-09-26 full run; the extra ~1.2k dropped vs the 3k-sample projection are truncated `Finish` payloads, `give_up_and_restart` endings, and mid-conversation `Finish` calls).
- **Eval carve (measured, full run 2026-09-26):** greedy rare-tool carve → **TB-500**: 500 eval conversations, **416 tools held out**, 483/500 with a real tool observation, median 5 raw turns (min 3, max 14) — genuinely agentic. Leaves **45,023 Stage 2 prefix conversations (94.1%)** whose tools never appear in the eval set. Held-out property enforced by construction. Rendered TB-500: median 920 tokens (min/max 257–4,053), 5.1% of tokens tool-call region, 7.6% final-answer region.
- **Contamination checks (measured):** 0 xLAM↔ToolBench shared queries in either direction; 2.5% shared function names (generic API names like `age_calculator` — different underlying APIs; acceptable); the benchmark config's `api_list` uses human-readable names ("Get Word by Start") in a *different namespace* from train's subfunction slugs (`getwordbystart_…`) — never compare tool identity across those namespaces.
- **Formatting pipeline (the local work):**
  1. Parse conversations → message lists (`system/user/assistant/tool` roles), converting schema style and tool-call format per the format facts above.
  2. Render via the target's `apply_chat_template(messages, tools=tools)`; multi-turn → single sequence with full chat history (loss on assistant turns only).
  3. Emit `datasets.Dataset` with `input_ids`, `labels`, and a `region` label map (assistant token spans; within them tool-call vs. prose sub-spans) — this powers the §4.3 region analysis and must be produced during data prep, not patched in later.
- **Stage 2 usage:** the 46,242-conversation pool provides *prefix contexts* (any user-turn prefix ending before an assistant turn is a valid on-policy sampling point — including post-observation states, which xLAM cannot supply). Cap at 4–6k tokens (left-truncate oldest turns).

---

## 4. Metrics and evaluation protocol

All runs report: dataset (xLAM-eval / TB-500), config (draft × untuned / Stage 1 / Stage 2 / n-gram / AR-only), k, temperature, batch, and raw tokens/sec. Every metric lands as JSON in `results/`; plots come later from the JSON, never hand-edited. The stage-wise table doubles as the transfer measurement: Stage 1 (xLAM-only) on TB-500 is the single-turn→multi-turn transfer number; Stage 2 (mixed) shows the gap Stage 2 closes.

**Uncertainty:** bootstrap 95% CIs over prompts for α and τ; wall-clock = median of 3 runs. A gain is only claimed if the CIs don't overlap.

**Stage-wise table:** every metric below is reported for untuned, Stage 1, and Stage 2 side by side, including the region split.

### 4.1 Acceptance metrics (instrumented HF loop, ~100–200 prompts)

- **α** per-token acceptance rate; **τ** mean accepted draft tokens per verification step. Define the convention up front in code: τ counts accepted draft tokens; the bonus token emitted after a rejection (the target's correction) and after a full acceptance counts separately as `bonus_rate`. Report both conventions in the README so numbers are comparable to other papers.
- **Per-position acceptance α_n (n = 1..k):** does the gain hold deeper into the draft, or is it all position-1 easy tokens? Compare curves across untuned / Stage 1 / Stage 2.

### 4.2 Wall-clock (vLLM, ~200–300 prompts)

- Tokens/sec and speedup vs. plain AR decoding, for: AR-only, untuned 0.5B, untuned 1.5B, Stage 1 and Stage 2 draft(s), n-gram.
- **Batch sensitivity is first-class:** measure at batch 1 and batch 8 and 32 (fixed concurrency, offline batched generate). Motivation is already in the plan: an EAGLE-3 reproduction (E2E Networks blog) found 2.3× at batch 4 degrading to roughly break-even at batch 32. Report the speedup-vs-batch curve for the best config and the baselines. If a bigger batch kills the speedup, that is a finding, not a failure — report it.
- Note vLLM's own caveat: logprobs/outputs can be slightly non-deterministic at larger batch; that's fine for wall-clock, and the exactness check (4.4) is run at batch 1.

### 4.3 Region-split acceptance — the differentiator

- Split generated tokens into **tool-call JSON region** (between `<tool_call>` and `</tool_call>`, incl. name vs. arguments sub-spans) vs. **free-form prose region** (everything else). Label tokens via offset mapping computed in the data/eval pipeline — decode speculatively, track which token indices fall in which span.
- Report α and τ **per region**. Hypothesis (from the agentic-serving result cited in the plan): acceptance is strongly bimodal — structured tool-call regions near 100%, prose sometimes below 10%.
- Extra cut worth 30 minutes: within the tool-call region, **argument keys / function names** (copyable from the schema in the prompt — n-gram's home turf) vs. **argument values** (must actually be predicted). This tells you exactly which tokens the distilled draft wins on that n-gram can't steal, and whether Stage 2's gains over Stage 1 concentrate in argument values.

### 4.4 Correctness verification (gate for everything else)

- Greedy, batch 1, fixed seed: compare full **token id sequences** of speculative vs. non-speculative decoding on the eval set. Require 100% identical ids (and therefore identical lengths). Also verify the n-gram method passes.
- Sanity check of the harness before the GPU day, run locally: draft = target (same 0.5B model both roles) with greedy → every draft token accepted, τ = k. If not, the instrumented loop is buggy.
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
| 0–1.5 | Env setup: clone repo, `pip install -r requirements.txt` (pinned vLLM), download target + Coder 0.5B + Coder 1.5B, smoke-test serve. Verify `speculative_config` with a draft model works with this vLLM build. | P0 |
| 1.5–3 | Baselines: AR-only at batch 1/8/32; n-gram baseline; untuned 0.5B and 1.5B spec decode at k=5 greedy (batch 1). Exactness check (4.4). Compute untuned τ for both drafts. | P0 |
| 3–3.5 | **Decision point:** pick distillation candidate via the §8 rule. | P0 |
| 3.5–4.5 | Stage 1 data: 5k xLAM target generations (greedy) with top-20 logprobs. Store as HF dataset + push to repo storage. | P0 |
| 4.5–5 | Stage 1 supervised KD warm-start. | P0 |
| 5–8 | Stage 2 on-policy distillation (xLAM + TB prefix mix), hourly checkpoints with quick τ check. | P0 |
| 8–11 | Re-benchmark untuned / Stage 1 / Stage 2: batch 1 grid (k × temp), batch sweep, exactness re-check, region-split analysis (4.3), per-position curves, CIs. Full k/temp sweep for untuned configs here too. | P0 |
| 11–13 | ToolBench transfer eval (4.1/4.2/4.3 on TB-500 multi-turn). | P1 |
| 13–16 | Second candidate (the other draft size) through Stage 1 + Stage 2 if the first went smoothly → same battery. | P1 |
| 16–17 | TVD-loss ablation for Stage 2. | P2 |
| 17–20 | All plots from results JSON; README results tables; repo cleanup. | P0 |
| 20–24 | Buffer (model download failures, OOM debugging, reruns). | — |

Run everything through `scripts/run_*.sh` so the day is a sequence of typed commands, not decisions. Each run appends to `results/metrics.jsonl`; a `make report` regenerates all tables/plots from it.

---

## 6. Local pre-work (before renting the GPU — this is where the project is actually built)

1. **Repo skeleton** (§7), pinned `requirements.txt`, venv. ✅ done 2026-09-26 (Python 3.12 venv with `datasets`/`huggingface_hub`/`python-dotenv` pinned; ToolBench mirror + xLAM downloaded to `data/raw/`).
2. **Data pipeline** ✅ done 2026-09-26 (xLAM carve + render verified at commit c4f6f1d; ToolBench cleaning this session: 47,870 clean convs of 187,542 (25.5%; 595 duplicate queries, rest filtered by give_answer/parse/valid-name rules), TB-500 rare-tool carve = 416 held-out tools, 45,023 Stage 2 prefix convs (94.1%). TB-500 rendered: median 920 tok, p95 2.2k, max 4.1k; 5.1% of tokens in tool-call region, 7.6% in final-answer region. 22 golden tests in `tests/test_toolbench_clean.py`, all passing.)
3. **Split-freezing** ✅ done 2026-09-26 (`src/data_prep/freeze_splits.py freeze|verify`; `frozen/` committed: xLAM-500 + TB-500 parquets, stage1/train-pool/eval/prefix index files, held-out lists, sha256 manifest incl. raw-data arrow checksums. `verify` re-checks checksums + held-out disjointness from frozen/ alone and is GPU-day step 0; 5 tests in `tests/test_freeze_splits.py`, incl. tamper detection).
4. **Evaluation script** ✅ done 2026-09-27 (`src/analysis/eval_acceptance.py` + 22 golden tests in `tests/test_eval_acceptance.py`): consumes the acceptance-event stream the instrumented loop will emit and produces every §4.1–4.3 metric — α, τ, `bonus_rate`, per-position α_n, region-split α (prose / tool-call JSON / tag / final answer) + name-vs-arguments sub-cut, bootstrap 95% CIs over prompts, per-prompt JSONL. Event schema pinned in the module docstring: one JSON object per verification step (`query_id`, `step`, optional `turn` for multi-turn records, `draft_tokens`, `accept_mask`, `correction_token`, `eos`). Counting conventions (golden-tested on hand-computed cases): a rejection at position j scores positions 0..j (post-rejection positions get no verdict, so they don't deflate α_n); τ = accepted draft tokens per step, bonus/correction counted separately as `bonus_rate`; per-turn streams reset the position cursor at each assistant-turn boundary (turn starts derived from the record's labels). Validated end-to-end on a 20-record real TB-500 event stream: flat α = exact weighted mean of region αs, per-prompt sums == flat numerators, zero context-region proposals (the protocol contract — the loop must generate per assistant turn, not across the whole record). Torch-free.
5. **Instrumented speculative decoding loop** ✅ done 2026-09-27 (`src/serving/instrumented_spec.py` + `src/serving/proposers.py`; 27 torch-free golden tests in `tests/test_instrumented_spec.py` + 3 real-model §4.4 tests in `tests/test_spec_realmodels.py`, run with `SPEC_REALMODELS=1` when the 0.5B models are cached). The draft side is a generic **Proposer** protocol (`propose/reset/sync`) with two implementations: an HF greedy draft (`HFDraftProposer`) and an **n-gram/prompt-lookup proposer** (`NgramProposer`, min/max n 2/5, mirrors vLLM's `ngram` method) — so §4.3's key question (which tokens n-gram already gets vs. which only the trained draft gets) is answered from the same instrumented region-split pipeline, which vLLM cannot provide. Conventions pinned in the module docstring and golden-tested: C1 rejection mask shape (no True after a False; the event carries only the *scored prefix* of the proposal so `draft_tokens`/`accept_mask` stay parallel per the analyzer's schema); C2 short EOS proposals (no bonus past EOS); C3 bonus on full acceptance; **C4 token budget — the proposal is capped at the remaining budget first and the bonus is dropped if it would exceed the cap, so outputs are exactly comparable to `generate(max_new_tokens=N)`** (the mid-round budget edge case); C5 EOS read from the tokenizer (never hard-coded); C6 per-assistant-turn generation with teacher-forced turn transitions via `sync()` (no re-prefill), turn starts imported from the analyzer; C7 non-empty proposals — an n-gram no-match emits a sacrificial PAD token (Qwen pad 151643) so step semantics stay uniform across proposers (the analyzer never sees an empty event). KV-cache sync is guarded two ways: the §4.4 self-consistency test (draft=target ⇒ α=1.0, every non-terminal round τ=k) and the exactness test (non-Coder 0.5B draft vs Coder 0.5B target ⇒ loop output token-identical to plain greedy `generate()`, with rejections actually occurring so the crop path is exercised — this caught a real `_forward`-clobbers-held-argmax bug). Validated end-to-end on frozen TB-500 records: 5 records → 279 events → full analyzer report with zero context-region proposals (the protocol contract) and per-position α_n. CLI: `python -m src.serving.instrumented_spec frozen/tb_eval.parquet --proposer ngram|draft --target-model ... --k 5 --limit 100 --out events.jsonl --outputs-out outputs.jsonl --meta-out meta.json`; deterministic subset = first N in frozen order.
6. **vLLM bench harness** (`bench_vllm.py`): given (model, spec config, prompts, params) → tokens/sec + outputs; exactness mode comparing token ids across two runs.
7. **Training scripts** (`training/kd_warmstart.py`, `training/onpolicy_gkd.py`) — tested end-to-end on tiny models locally (e.g. Coder-0.5B as both student and stand-in teacher for a few steps on CPU/MPS), including the vocab-slicing in the loss, so the GPU day never debugs training code.
8. **Dry-run the whole pipeline at toy scale** (tiny model as fake target, 20 prompts) — the full loop from raw xLAM + raw ToolBench to metrics JSON must run green on CPU before the GPU is rented.
9. **Stage 2 mixing:** fix the xLAM:TB prefix ratio when building Stage 2 context files (default 1:1 by conversation; ablate only if time remains).

---

## 7. Repo layout

```
Speculative_Decoding/
├── plan.md, README.md, requirements.txt
├── configs/            # model paths, spec configs, sweep grids (JSON)
├── data/               # gitignored; raw/ + processed/
├── src/
│   ├── data_prep/      # download_datasets, xlam_prep (carve+render), toolbench_clean (parse+convert+carve), build_distill_data
│   ├── serving/        # bench_vllm.py, spec_configs.py, instrumented_spec.py
│   ├── training/       # kd_warmstart.py, onpolicy_gkd.py
│   └── analysis/       # eval_acceptance.py (§4.1–4.3 metrics from loop events), plots.py
├── scripts/            # run_baselines.sh, run_sweep.sh, run_distill.sh, run_report.sh
├── results/            # metrics.jsonl, tables/, plots/   (committed)
└── tests/              # golden render tests, harness self-consistency tests
```

---

## 8. Pre-registered decisions and success criteria

Written down *before* the GPU day so choices aren't made after seeing results:

1. **Draft choice rule (decision point, hour ~3):** measure untuned wall-clock speedup vs. AR at k=5, greedy, batch 1 for both candidates. Distill the one with the higher untuned speedup. Tie-break toward 0.5B (cheaper to train and verify; more headroom). Distill the second candidate only as P1.
2. **Success criteria:**
   - **P0 (must):** exactness check passes; AR / n-gram / untuned-draft baselines all measured; τ, α, per-position and region-split analyses produced for untuned, Stage 1, and Stage 2 of at least one draft.
   - **P1 (should):** Stage 2 draft beats its untuned self on τ by ≥20% (non-overlapping CIs) *and* beats the n-gram baseline on wall-clock at batch 1, greedy, on xLAM-eval (in-domain); Stage 2 measurably beats Stage 1.
   - **P2 (stretch):** gains hold at batch 8/32; ToolBench transfer (TB-500) is positive — i.e. the xLAM-trained draft also improves spec-decode on multi-turn agentic trajectories; per-position curve shows improvement deep into the draft; second draft size distilled; TVD ablation.
   - Negative results (e.g. n-gram wins, speedup collapses at batch 32, on-policy adds little over Stage 1) are reported as findings with the region-split explanation — that's still a publishable characterization.
3. **Eval sets frozen before the GPU day:** xLAM-500 (206 held-out functions, rarest-function carve), TB-500 (410 held-out tools, rare-tool carve, 483/500 with real tool observations), fixed seeds, fixed prompt files committed to the repo (§6.3).

**8.0. Dataset-role flip (decided 2026-09-26, pre-GPU-day, no results seen).** Originally: train on ToolBench, eval format-transfer on xLAM. **Flipped:** train on xLAM (+ ToolBench as Stage 2 context mix), eval in-domain on xLAM-500 and agentic transfer on TB-500. Grounds, all measured on the downloaded data (§3): (a) xLAM is 100% usable and already in the target's native tool-call shape; the ToolBench mirror needs a cleaning pipeline and only ~27% of its conversations survive filtering; (b) xLAM prompts are ~5× shorter (median ~1.1k chars vs ~5.8k), and Stage 1/2 both scale with context length — the GPU-day bottleneck; (c) zero query overlap between the datasets in either direction; (d) the flip makes the stage-wise table a *transfer experiment*: Stage 1 (xLAM-only) on TB-500 measures single-turn→multi-turn transfer, Stage 2 (mixed) shows how much on-policy agentic data closes it. Accepted cost: xLAM is single-turn, so a Stage-1-only draft has never seen post-observation states — mitigated by the Stage 2 ToolBench mix and reported honestly via the region split (expect transfer gains concentrated in tool-call JSON, less in post-observation prose).

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Original ToolBench repo gated | Using mirror `tuandunghcmut/toolbench-v1` (downloaded, inspected, format documented in §3.2). |
| vLLM version drift / spec-decode API changes | Pin exact version; `speculative_config` smoke test is the first GPU-hour task; keep the instrumented HF loop as the fallback measurement path. |
| Target hallucinates argument keys not in schema (distills bad habits into draft) | Spot-check 50 generations during Stage 1 data gen; drop trajectories with invalid tool calls (schema-validated); log the drop rate. |
| Draft never beats n-gram | That's a legitimate result — the region-split analysis (which tokens n-gram steals vs. which the draft must predict) is the deliverable either way. |
| Batch non-determinism confuses exactness check | Exactness at batch 1 only; wall-clock at all batches; report separately. |
| OOM at batch 32 with long ToolBench prompts (big schema blocks) | Cap `max_model_len` (16k), then `gpu_memory_utilization` 0.92; length distribution is measured locally in advance so this isn't a surprise. |
| Vocab-size mismatch breaks KD/GKD loss | Slice logits to 151,665 in both losses; covered by the local end-to-end test (§6). |
| Stage 2 slower than planned (draft generation on long contexts) | Cap contexts at 4–6k tokens; reduce prompt count; hourly checkpoints mean any checkpoint is usable. |
| Stage 2 unstable early | Stage 1 warm-start + mixing in fixed data; fall back to the Stage 1 checkpoint as the reported result if Stage 2 diverges. |
| xLAM-only training misses post-observation/agentic states | Stage 2 mixes ToolBench prefixes (46k pool) so the draft sees post-observation states on-policy; transfer to TB-500 is a first-class reported metric, not a footnote. |
| ToolBench mirror is pre-processed ShareGPT-style, not raw trajectories (schema embedded in system prompt, `Thought:/Action:` text format) | Parsing recipes verified on a 3k scan (§3.2): `ast.literal_eval` on the system-prompt API list works 100%; uniform `Thought:/Action:/Action Input:` structure. Golden-example unit tests (§6.2) guard the conversion. |
| Tool-identity namespaces differ between benchmark config and train (human-readable names vs subfunction slugs) | Never compare tool identity across namespaces (§3.2); TB-500 held-out property is enforced within the train pool's own namespace by construction. |
| GPU day slips on setup | Everything local (§6) is done first; day-0 checklist in the runbook is mechanical. |

---

## 10. Open questions

- Stage 2 mixing ratio xLAM : ToolBench prefixes (default 1:1 by conversation, §6.9) — ablate only if time remains.
- Stage 2 draft sampling temperature (T=1.0 default) — ablate greedy sampling only if time remains.
- **Next steps for the memo:** AdaSPEC-style token filtering (train only on tokens the draft can realistically learn), and an EAGLE-3 head as the higher-ceiling follow-up.