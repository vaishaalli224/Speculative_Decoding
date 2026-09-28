#!/usr/bin/env bash
# Minimal eval battery (armed 2026-09-28): waits for the Stage-2 rerun to
# finish, then runs ONLY the runs the memo needs (~1.5-2h) instead of the
# full hours 9-14 battery. Cut list + descoping sentences live in the memo
# discussion; everything here maps to a plan-P0/P1 item or an artifact-
# killer. Order = priority, so a short clock kills rows from the bottom.
#
# Phase 0  wait: no src.training.onpolicy_gkd process AND
#          checkpoints/stage2/train_meta.json exists (final save done)
# Smoke    fail-fast: stage2 draft spec config + TB per-turn at limit 5
# Phase B  instrumented loop (plan protocol: k=5 greedy, 100 prompts,
#          bootstrap CIs, --subcut): Stage-2 on xLAM-500, then TB-500.
#          Region split + per-position come free from the same events.
# Phase C1 exactness gate, Stage-2 draft, 50 prompts (recorded, not
#          fatal: the plan's pre-registered response to bf16 near-ties is
#          that the HF loop is the exactness claim; vLLM is the observation)
# Phase C2 wall-clock, Stage-2 k=5 greedy b1 xLAM (P1 criterion; AR 66.7 /
#          n-gram 111.5 / untuned 148.0 are already measured - the P1 bar)
# Phase C3 TB-500 per-turn wall-clock: AR + n-gram + Stage-2, the same
#          teacher-forced states the instrumented loop scores (C6/B5) -
#          turns the transfer-tau row into transfer-speedup, with the same-
#          prompts baseline n-gram row (it degrades on TB's prose turns;
#          that contrast is the point)
# Phase C4 batch-32 point: Stage-2 + n-gram (AR b32 already measured) -
#          the one batching number the memo quotes
#
# Optional (env-gated, run only if the clock allows):
#   WITH_B32=1      phase C4 (default on; WITH_B32=0 skips)
#   WITH_K9=1       one k=9 tau point for the Stage-2 draft
#   WITH_S1_TBKD_WALL=1  stage1 + tbkd wall-clock rows
#
# Resumable: every phase checks for its committed artifact first and
# skips what is done, so a re-arm after a partial run never redoes work.
#
# Usage:  setsid nohup bash scripts/run_minimal_evals.sh \
#             > logs/minimal_evals.log 2>&1 < /dev/null &
#         (or plain `bash scripts/run_minimal_evals.sh` to watch it live)
# Env overrides: PY, TARGET, S2DIR, TAGP, TAU_LIMIT, LIMIT, TB_LIMIT,
#                MAX_NEW_TOKENS, METRICS, BEST_K, K9

set -euo pipefail
cd "$(dirname "$0")/.."

# --- auto-log: every invocation writes its own log ------------------------
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${0%.sh}")_$(date -u +%Y%m%d-%H%M%S)_$$.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest_evals.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PY="${PY:-.venv-h100/bin/python}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-14B-Instruct}"
S2DIR="${S2DIR:-checkpoints/stage2}"
# Artifact tag prefix for every output of this battery. The 2026-09-28
# TB-only rerun trains into stage2_tb/ and MUST run with TAGP=s2tb: the
# first (mixed) Stage-2 run's stage2_* artifacts are banked results for a
# DIFFERENT draft, and the cache checks would otherwise skip every phase
# and report the old draft's numbers as this one's.
TAGP="${TAGP:-stage2}"
TAU_LIMIT="${TAU_LIMIT:-100}"
LIMIT="${LIMIT:-200}"
TB_LIMIT="${TB_LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
METRICS="${METRICS:-results/metrics.jsonl}"
BEST_K="${BEST_K:-5}"
WITH_B32="${WITH_B32:-1}"
NPROC_S1_5B="${NPROC_S1_5B:-4}"

# -- Phase 0: wait for the training run to finish --------------------------
echo "== Phase 0: waiting for Stage-2 training to finish ($(date -u +%H:%M:%S)) =="
while true; do
  if ! pgrep -f "src.training.onpolicy_gkd" >/dev/null \
     && [ -f "$S2DIR/train_meta.json" ]; then
    break
  fi
  sleep 300
done
echo "training finished at $(date -u +%H:%M:%S) — GPU is ours"

# Resolve the Stage-2 draft: final if it exists, else the newest step_*.
S2="$S2DIR/final"
if [ ! -d "$S2" ]; then
  S2=$(ls -d "$S2DIR"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
  echo "[warn] $S2DIR/final missing — using newest checkpoint $S2"
fi
echo "Stage-2 draft under test: $S2"

# -- helpers (same shapes as run_overnight_evals.sh, so artifacts interlock) -
tau_run() {  # tau_run <draft-path> <tag> <records-parquet>  (resumable)
  local D=$1 TAG=$2 REC=$3
  if [ -f "results/events/${TAG}_k5_report.json" ]; then
    echo "  tau cached: $TAG"
    return 0
  fi
  $PY -m src.serving.instrumented_spec "$REC" --proposer draft \
      --draft-model "$D" --target-model "$TARGET" --k 5 \
      --limit "$TAU_LIMIT" --dtype bfloat16 --device cuda:0 \
      --out "results/events/${TAG}_k5.jsonl" \
      --outputs-out "results/events/${TAG}_k5_outputs.jsonl" \
      --meta-out "results/events/${TAG}_k5_meta.json"
  $PY -m src.analysis.eval_acceptance "results/events/${TAG}_k5.jsonl" \
      --records "$REC" --k 5 --subcut \
      --out "results/events/${TAG}_k5_report.json" \
      --per-prompt "results/events/${TAG}_k5_per_prompt.jsonl"
  echo "  tau done: $TAG"
}

bench() {  # bench <name> <extra-args...> — resumable wall-clock run
  local NAME=$1; shift
  if [ -f "results/vllm/${NAME}.jsonl" ]; then
    echo "  bench cached: $NAME"
    return 0
  fi
  $PY -m src.serving.bench_vllm "$@" --metrics-out "$METRICS" \
      --outputs-out "results/vllm/${NAME}.jsonl"
  echo "  bench done: $NAME"
}

exactness_gate() {  # exactness_gate <name> <draft> — RECORD, never abort
  # Trained drafts mismatch AR at bf16 near-ties (measured 46/50 on stage1;
  # plan §9 pre-registers the response: the HF loop is the exactness CLAIM,
  # vLLM's behavior is reported as an observation. So the gate records its
  # result and the queue continues.)
  local NAME=$1 D=$2
  if [ -f "results/exactness/${NAME}_k5.json" ]; then
    echo "  gate cached: $NAME"
    return 0
  fi
  if $PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --exactness \
      --method draft_model --draft-model "$D" --k 5 --model "$TARGET" \
      --limit 50 --max-new-tokens "$MAX_NEW_TOKENS" \
      --outputs-out "results/vllm/${NAME}_k5_exact.jsonl" \
      --exactness-out "results/exactness/${NAME}_k5.json" \
      --metrics-out "$METRICS"; then
    echo "  gate PASS: $NAME"
  else
    echo "  gate recorded mismatches: $NAME (continuing — plan §9: HF loop is the exactness claim; vLLM reported as observation)"
  fi
}

# -- Smoke: fail in minutes, not hours --------------------------------------
echo "== smoke: stage2 spec config + TB per-turn (limit 5) =="
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method draft_model \
    --draft-model "$S2" --k 5 --model "$TARGET" --temperature greedy \
    --batch 1 --runs 1 --limit 5 --max-new-tokens 64 --warmup 0 \
    --metrics-out "" --outputs-out ""
$PY -m src.serving.bench_vllm frozen/tb_eval.parquet --per-turn \
    --method draft_model --draft-model "$S2" --k 5 --model "$TARGET" \
    --temperature greedy --batch 1 --runs 1 --limit 5 --max-new-tokens 64 \
    --warmup 0 --metrics-out "" --outputs-out ""
echo "  smoke OK — spec config accepted, per-turn TB path runs"

# -- Phase B: the stage-wise Stage-2 rows (plan §4.1/§4.3 protocol) ----------
echo "== Phase B: instrumented tau/alpha rows (k=5 greedy, $TAU_LIMIT prompts) =="
tau_run "$S2" "${TAGP}_final_on_xlam" frozen/xlam_eval.parquet
tau_run "$S2" "${TAGP}_final_on_tb" frozen/tb_eval.parquet
echo "Phase B done — reports in results/events/stage2_final_on_*_k5_report.json"

# -- Phase C1: exactness gate on the trained draft ---------------------------
echo "== Phase C1: exactness gate (50 prompts, greedy b1; recorded, not fatal) =="
exactness_gate "$TAGP" "$S2"

# -- Phase C2: the P1 wall-clock row -----------------------------------------
echo "== Phase C2: stage2 wall-clock, k=5 greedy b1 xLAM (P1 criterion) =="
bench "${TAGP}_k5_greedy_b1" frozen/xlam_eval.parquet --method draft_model \
    --draft-model "$S2" --k 5 --model "$TARGET" --temperature greedy \
    --batch 1 --runs 3 --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"

# -- Phase C3: TB-500 per-turn transfer wall-clock ---------------------------
# All three configs on the SAME per-turn prompts, or the ratios mean
# nothing: AR is the denominator, n-gram is the free-copying comparator
# (instrumented profile predicts it degrades on TB's prose-heavy turns).
echo "== Phase C3: TB-500 per-turn wall-clock: AR + n-gram + stage2 =="
bench "tb_ar_k${BEST_K}_greedy_b1" frozen/tb_eval.parquet --per-turn \
    --method ar --k "$BEST_K" --model "$TARGET" --temperature greedy \
    --batch 1 --runs 3 --limit "$TB_LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"
bench "tb_ngram_k${BEST_K}_greedy_b1" frozen/tb_eval.parquet --per-turn \
    --method ngram --k "$BEST_K" --model "$TARGET" --temperature greedy \
    --batch 1 --runs 3 --limit "$TB_LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"
bench "tb_${TAGP}_k${BEST_K}_greedy_b1" frozen/tb_eval.parquet --per-turn \
    --method draft_model --draft-model "$S2" --k "$BEST_K" --model "$TARGET" \
    --temperature greedy --batch 1 --runs 3 --limit "$TB_LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS"

# -- Phase C4 (env-gated): the one batching number ---------------------------
if [ "$WITH_B32" = "1" ]; then
  echo "== Phase C4: batch-32 point, stage2 + n-gram (AR b32 measured) =="
  bench "${TAGP}_k${BEST_K}_greedy_b32" frozen/xlam_eval.parquet --method draft_model \
      --draft-model "$S2" --k "$BEST_K" --model "$TARGET" --temperature greedy \
      --batch 32 --runs 3 --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"
  bench "ngram_k${BEST_K}_greedy_b32" frozen/xlam_eval.parquet --method ngram \
      --k "$BEST_K" --model "$TARGET" --temperature greedy --batch 32 --runs 3 \
      --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"
else
  echo "== Phase C4 skipped (WITH_B32=0) =="
fi

# -- Optional extras (env-gated; run only if the clock allows) --------------
if [ "${WITH_K9:-0}" = "1" ]; then
  echo "== extra: k=9 tau point for the stage2 draft =="
  if [ ! -f "results/events/${TAGP}_final_on_xlam_k9_report.json" ]; then
    $PY -m src.serving.instrumented_spec frozen/xlam_eval.parquet --proposer draft \
        --draft-model "$S2" --target-model "$TARGET" --k 9 \
        --limit "$TAU_LIMIT" --dtype bfloat16 --device cuda:0 \
        --out "results/events/${TAGP}_final_on_xlam_k9.jsonl" \
        --outputs-out "results/events/${TAGP}_final_on_xlam_k9_outputs.jsonl" \
        --meta-out "results/events/${TAGP}_final_on_xlam_k9_meta.json"
    $PY -m src.analysis.eval_acceptance "results/events/${TAGP}_final_on_xlam_k9.jsonl" \
        --records frozen/xlam_eval.parquet --k 9 --subcut \
        --out "results/events/${TAGP}_final_on_xlam_k9_report.json" \
        --per-prompt "results/events/${TAGP}_final_on_xlam_k9_per_prompt.jsonl"
  fi
fi

if [ "${WITH_S1_TBKD_WALL:-0}" = "1" ]; then
  echo "== extra: stage1 + tbkd wall-clock rows =="
  bench "stage1_k5_greedy_b1" frozen/xlam_eval.parquet --method draft_model \
      --draft-model checkpoints/stage1/final --k 5 --model "$TARGET" \
      --temperature greedy --batch 1 --runs 3 --limit "$LIMIT" \
      --max-new-tokens "$MAX_NEW_TOKENS"
  bench "tbkd_k5_greedy_b1" frozen/xlam_eval.parquet --method draft_model \
      --draft-model checkpoints/tb_stage1/final --k 5 --model "$TARGET" \
      --temperature greedy --batch 1 --runs 3 --limit "$LIMIT" \
      --max-new-tokens "$MAX_NEW_TOKENS"
fi

echo "== MINIMAL EVALS DONE ($(date -u +%H:%M:%S)) =="
echo "Read-out order:"
echo "  1. results/events/stage2_final_on_{xlam,tb}_k5_report.json   (headline stage-wise rows)"
echo "  2. results/exactness/stage2_k5.json                          (lossless gate)"
echo "  3. results/vllm/${TAGP}_k5_greedy_b1.jsonl + metrics.jsonl   (P1 wall-clock)"
echo "  4. results/vllm/tb_{ar,ngram,stage2}_k${BEST_K}_greedy_b1    (transfer speedup)"
echo "  5. results/vllm/*_b32.jsonl                                  (batching point)"
echo "  comparators (amended §2): TB-KD 4.07 xLAM / 3.04 TB"
