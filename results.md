# Results log — GPU day 2026-09-27 (rented vast.ai H100 80GB, instance #50993801)

A chronological record of decisions, their reasons, and the measured
results. Every number below comes from a committed, re-runnable artifact
(reports in `results/events/*_report.json`; metrics lines in
`results/metrics.jsonl`). Protocol for every τ/α row unless stated:
instrumented HF loop, k=5, greedy, first 100 prompts of the frozen eval
parquet in frozen order, bootstrap 95% CIs over prompts.

---

## 1. The stage-wise transfer matrix (the day's core result so far)

τ = mean accepted draft tokens per verification step (max 5.0 at k=5).

| Draft | xLAM-500 τ | TB-500 τ | Notes |
|---|---|---|---|
| Untuned Qwen2.5-Coder-0.5B-Instruct | 3.32 [3.23–3.43] | 2.63 [2.54–2.77] | α 0.890 / 0.811 |
| Stage-1 KD (xLAM contexts) | 4.01 | 2.65 [2.61–2.84] | α 0.942 / 0.812 |
| TB-KD (TB contexts; ablation A) | **4.07** [4.19–4.42]* | **3.04** [3.01–3.26] | α 0.948 / 0.852 |
| Stage-2 GKD (on-policy; from TB-KD) | *training* | *training* | §2 warm-start amendment |

*The bootstrap CI convention differs slightly from the flat mean (pinned
in the report stage); the ordering is unambiguous either way.

**Finding 1 — Stage-1's gain was real but did not transfer.** Stage-1 KD
(xLAM-only, target-greedy data) moved xLAM τ 3.32 → 4.01 (+21%, the plan's
P1 criterion, non-overlapping CIs) but TB τ 2.63 → 2.65 — zero transfer,
inside the CI. The mechanism was measured before the explanation: the
entire xLAM gain concentrated in the cold start (position-0 acceptance
α₀ 0.03 → 0.82) and the tool-call tags (region α 0.62 → 0.88), both
*format* skills of a distribution whose format is uniform (73/100 of the
target's xLAM openers are the same token). TB's opener distribution is
6-way heterogeneous (`<` 32%, `To` 21%, `Sure` 14%, `Certainly` 14%,
fence 9%, `{\n` 8%) and half its turn-0s carry no tool call at all — a
uniform format skill has nothing to attach to there.

**Finding 2 — transfer is asymmetric, downward only.** xLAM→TB: zero
(uniform skill, doesn't travel). TB→xLAM: ~full (the heterogeneous
distribution's skill includes the uniform case as one of its modes).
Distillation generalizes from diverse to narrow, not upward. This is the
"train on the harder distribution" result: TB-KD, trained only on TB,
*ties the xLAM-trained draft on xLAM* (4.07 vs 4.01) while beating it by
+0.39 τ on TB.

**Finding 3 — off-policy KD on the right contexts is a strong baseline.**
For greedy spec decoding there was a real theoretical case that
target-greedy data (off-policy) suffices — committed prefixes are always
target-greedy states, so teacher-forced KD states match inference states.
Ablation A confirmed it empirically: the Stage-1 recipe applied to TB
contexts closed 60% of the gap that xLAM Stage-1 closed none of. Stage-2
(on-policy GKD) must now beat 3.04/4.07 — not Stage-1's numbers — to
justify the sampling machinery.

## 2. Decisions taken mid-day (each pre-registered before its result landed)

### 2.1 Draft fixed to 0.5B (§8.1, decided before the rental)
The planned untuned 0.5B-vs-1.5B comparison was cancelled on prior
grounds (published reproductions favor 0.5B-class drafts for this target
class; the one-day budget shouldn't buy a decision the prior already
answers). Accepted cost, stated in the plan: no head-to-head numbers for
this setup. The untuned 0.5B run stayed as the baseline row.

### 2.2 Draft vocabulary padding (forced by a measured vLLM constraint)
vLLM 0.30.0's SpeculativeConfig hard-rejects draft/target pairs whose
`config.vocab_size` differ (target 152,064 vs draft 151,936) — caught by
the day-0 smoke test, contradicting plan §1.2's "irrelevant for vLLM".
Fix: `prepare_draft.py` pads the draft's embedding (+ untied lm_head) with
**zero rows** to 152,064, gated on greedy parity (padded draft output must
be token-identical on a frozen prompt). Residual, stated honestly: a zero
lm_head row gives padding ids logit 0; if every real logit were negative
the padded draft could argmax a padding id → one rejection, never a
correctness issue (only target argmax tokens are ever emitted; the §4.4
exactness gate passed 50/50 both methods).

### 2.3 KD data mirrors the target's actual behavior, not its instruction (G8)
The target at greedy wraps 92.6% of its xLAM tool calls in the
**schema-announcement tag pair** (`<tools></tools>`), not the
template-instructed assistant pair — with payloads always schema-valid.
A/B tested: neither the default system prompt nor an explicit
function-calling instruction flips this; the model is consistent and
correct in content, wrong only in wrapper. Decision: the KD data keeps
the wrapper **verbatim as emitted** (validating payloads, never repairing
wrappers) — repairing would teach the draft to propose tokens the target
rejects. Related measured facts folded into the validator: parallel calls
are comma-joined inside one wrapper (array elements without brackets);
prose-only generations are kept on TB (the target's majority class there,
49/51 wrapper/prose at turn-0) but would be wrong to keep on xLAM.

### 2.4 Stage-1 trained in fp32 (bf16 NaN, root-caused)
Full-bf16 training NaN-poisoned the weights within ~4 optimizer steps
(loss log NaN at every logged step; a 254 s "run" saved a garbage
checkpoint). Bisected to the optimizer path: forward/loss/backward all
verified finite individually (single pack, all-packs-sequence,
ckpt×backward matrix, lr=1e-12 — which *also* NaN'd, ruling out the
update magnitude). bf16 AdamW state precision is the diagnosis; fp32
master weights are the fix. The 0.5B trains in ~25 min fp32 — no
precision/speed trade-off at this scale. Recorded in kd_warmstart's
dtype note; Stage-2 inherited it (G7).

### 2.5 The ablation battery (§4.7, run while Stage-2 code finished)
Motivated by two user challenges, both promoted from §10 open questions
to running ablations with pre-registered decision rules:
- **A (TB-prefix KD)** — "why not SFT on ToolBench?" The weak form (SFT
  toward gold ToolLLaMA text) fails on a measured fact: acceptance is
  agreement with the *target*, and the target deviates from gold even on
  xLAM (70% exact, 21% args-differ — and gold there is target-format;
  TB gold is a different model's text entirely). The strong form (KD
  from the target's own greedy generations on TB contexts) is legitimate
  — and it **won** (Findings 2–3). Same recipe, different context
  distribution: 5k assistant-turn boundary contexts (median 1.4k tok,
  post-observation states included), T3 validation kept 4,911/5,000
  (1.8% dropped — genuine schema violations only), trained from the
  *untuned* draft to isolate the context distribution from any warm
  start.
- **B (promptability)** — "couldn't a stricter prompt get the format
  gain?" In flight; protocol: untuned draft + three prompt variants
  (base / demo turn teaching the instructed format / raw text naming the
  target's actual opening). The interesting twist the prototype surfaced:
  the template renders *supervised* demo turns with the canonical
  assistant tags while the target *generates* the schema pair — so the
  demo variant teaches the format the target was told to use, the raw
  variant the format it actually uses. If demo fails and raw works,
  prompts can inject knowledge but not the target's learned violation of
  its own instruction.

### 2.6 Stage-2 initializes from TB-KD, not Stage-1 (warm-start amendment)
Decided from A's results *before Stage-2's launch*: TB-KD dominates
Stage-1 in both columns (4.07/3.04 vs 4.01/2.65), and GKD samples on a
1:1 TB:xLAM mix — a Stage-1-initialized draft has never seen a TB state,
so its early TB samples are junk (wasted on-policy signal); the TB-KD
draft is competent on both from step one. Recorded in plan §2 with the
numbers; Stage-2's row becomes "does on-policy add anything over the
best off-policy result?"

## 3. Wall-clock baselines (vLLM; the speedup denominators)

| Config (greedy, 200 frozen xLAM prompts, median of 3) | gen tok/s | speedup |
|---|---|---|
| AR-only, batch 1 | 66.7 | 1.00× |
| AR-only, batch 8 | 238.5 | — |
| AR-only, batch 32 | 582.1 | — |
| n-gram k=5, batch 1 | 111.5 | 1.67× |
| Untuned 0.5B k=5, batch 1 | 148.0 | **2.22×** |

§4.4 exactness gates: **50/50 token-identical** for both draft-model and
n-gram speculative decoding vs plain greedy AR — the speedups are
lossless. The instrumented loop cross-checks the wall-clock: untuned
τ=3.32 predicts ≈2.15×, measured 2.22× (4% agreement).

**Acceptance holds deep into the draft (k-scope finding):** untuned
per-position α at k=9: 0.79 / 0.85 / 0.84 / 0.95 / 0.96 / 0.96 / 0.97 /
0.97 / 0.96 — no collapse; α at position 8 equals α at position 4. τ rose
3.32 → 4.72 from k=5 → k=9 with step count down 2,025 → 1,537. The
re-benchmark k-grid extends to {3, 5, 7, 9} on this evidence.

**The draft is the instrumented-loop bottleneck (C8 phase timing):**
propose (draft, k sequential forwards) vs verify (target, one forward):
k=5 → 72 ms vs 38 ms (**65% draft share**); k=9 → 127 ms vs 41 ms (76%);
TB runs identical (65%). In vLLM's compiled engine the split differs —
the wall-clock verdict comes from the k-sweep — but in the HF instrument,
larger k buys ~1.4 more accepted tokens per verify at roughly constant
verify cost.

**n-gram's profile explains where it loses:** α 0.44/τ 0.72 (xLAM),
0.38/0.58 (TB) — position-1 acceptance 0.26 (a fresh step often has no
prompt match), then it locks onto continuation runs (α *rises* with
position: 0.26 → 0.88). It copies; it cannot predict — and it degrades
worse on TB's prose-heavy thoughts (τ 0.72 → 0.58) while the draft's
advantage *widens* there.

## 4. Contamination / headroom checks (pre-registered questions, measured)

- **Gold-reproduction (target vs xLAM gold, 5,000 generations):** 69.9%
  exact name+arguments, 20.8% same-function-different-args, 0.4% wrong
  function, 9.0% no parseable call. Reading: the target is right about
  *which* function ~90%+ of the time but its argument *values* deviate
  from gold on a fifth of calls — that 21% band is exactly the token-level
  distribution KD can teach (which defaults the target fills, which
  spellings it prefers: `Off-road` vs `off-road`). xLAM is plausibly in
  both models' training diet, but the target's behavior is not
  deterministic from schema alone.
- **The cold start was the untuned draft's single biggest defect:**
  position-0 acceptance 0.03 — the draft opens with a markdown fence
  (```) in 97/100 prompts; the target opens with `<` (the wrapper) in
  72/97 corrections. One guaranteed rejection per prompt. Stage-1's
  entire xLAM gain is this fix plus tags (the region split says so), and
  TB-KD replicates it TB-style (α₀ 0.63 per-turn across 262 multi-turn
  boundaries, openers `<` 64/`To` 23/`Sure` 13 — the target's whole
  opening distribution).

## 5. What remains (at time of writing)

- Ablation B reports (promptability, three variants) — in flight.
- Stage-2 GKD from TB-KD — launching; hourly checkpoints with in-process
  τ probes; its number is judged against 3.04 (TB) / 4.07 (xLAM).
- Re-benchmark battery: k ∈ {3,5,7,9} × {greedy, T=1.0} at batch 1 for
  every draft row; batch sweep 1/8/32 for the best config + AR + n-gram;
  exactness re-check; full region-split/per-position/CI analysis on both
  eval sets; TB-500 transfer wall-clock per-turn.
- Plots + README tables + memo from `results/metrics.jsonl` only.

## 6. Infra decisions worth recording (they shaped the data)

- **Two environments, two requirement files** — vLLM's tree is pinned
  separately; over-pinning its transitive deps produced pip
  ResolutionImpossible on the host (fixed by installing vLLM first,
  alone).
- **xLAM is Hub-gated** — the host needed the HF token via scp + repo-
  root `.env` loading (the "datasets are public" assumption was half
  wrong: the ToolBench mirror is public, xLAM is not).
- **Auto-logging** — every `run_*.sh` writes `logs/<script>_<ts>_<pid>.log`
  + a `latest.log` symlink; the day is `tail -f`-able and every decision
  above is traceable to a log.
- **Cross-session lane discipline** — Stage-2 (apps-48) vs measurement/
  ablations (this session): one shared host, coordinated via drain
  watchers; the ablation lanes never modified the golden-tested Stage-1
  pipeline (the TB assembly is deliberately inline in the ablation
  script).
