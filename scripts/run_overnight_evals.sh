#!/usr/bin/env bash
# Overnight eval queue (plan §5 hours 9-12, armed 2026-09-27 evening): waits
# for the Stage-2 training process to exit, then runs the FULL re-bench
# battery unattended so every eval is done by morning. The user monitors
# the training run itself; this script only waits on process exit + the
# final-save marker, then owns the GPU.
#
# Phase 0  wait: no src.training.onpolicy_gkd process AND
#         checkpoints/stage2/train_meta.json exists (final save done)
# Smoke   fail-fast: stage2 spec config + TB per-turn + exactness at limit 5
# Phase B  instrumented loop (tau/alpha/region/per-position, k=5 greedy,
#         100 prompts, the stage-wise-table protocol) for the Stage-2
#         draft on xLAM-500 + TB-500: final FIRST (the headline rows),
#         then step_300/200/100 (checkpoint trajectory; step_400 is
#         skipped: final is a re-save of the same weights). Untuned /
#         Stage-1 / TB-KD rows already measured (results/events/*_report).
# Phase C  vLLM wall-clock: exactness gates (stage1, tbkd, stage2),
#         k5-greedy-b1 wall-clock rows for stage1 + tbkd (xLAM + TB
#         per-turn), untuned k x temp grid, then the full stage2 battery
#         via run_sweep.sh (k x temp grid + n-gram k sweep + batch sweep
#         1/8/32 + TB transfer). AR b1/8/32 + untuned k5 + n-gram k5 were
#         measured this morning (run_baselines.sh) and are not redone
#         except where run_sweep.sh's battery includes them.
#
# Skipped per user decision (2026-09-27): ablation B (promptability —
# inferable from TB-KD's transfer pattern), TVD-loss ablation (no time).
#
# Usage:  setsid nohup bash scripts/run_overnight_evals.sh \
#             > logs/overnight_evals.log 2>&1 < /dev/null &
# Env overrides: PY, TARGET, LIMIT, TB_LIMIT, TAU_LIMIT, MAX_NEW_TOKENS,
#                METRICS, K_VALUES, BEST_K

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
UNTUNED="${UNTUNED:-drafts/coder-0.5b-padded}"
STAGE1="${STAGE1:-checkpoints/stage1/final}"
TBKD="${TBKD:-checkpoints/tb_stage1/final}"
S2DIR="${S2DIR:-checkpoints/stage2}"
LIMIT="${LIMIT:-200}"
TB_LIMIT="${TB_LIMIT:-100}"
TAU_LIMIT="${TAU_LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
METRICS="${METRICS:-results/metrics.jsonl}"
K_VALUES="${K_VALUES:-3 5 7}"
BEST_K="${BEST_K:-5}"

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
  S2=$(ls -d "$S2DIR"/step_* 2>/dev/null | sort | tail -1)
  echo "[warn] $S2DIR/final missing — using newest checkpoint $S2"
fi
echo "Stage-2 draft under test: $S2"

# -- instrumented-loop helper (the run_tau.sh protocol) ---------------------
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

bench() {  # bench <name> <extra-args...> — b1 wall-clock shorthand (resumable)
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
  # Trained drafts mismatch AR at bf16 near-ties (vLLM's batched verify
  # forward vs incremental AR flip argmaxes differently; measured 46/50 on
  # stage1 2026-09-28, two ' you'(498) insertions). Plan §9 pre-registers
  # the response: the HF instrumented loop is the exactness CLAIM; vLLM's
  # behavior is reported as an observation. So the gate records its result
  # and the queue continues.
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

# -- Smoke: the overnight paths fail in minutes, not hours ------------------
echo "== smoke: stage2 spec config + TB per-turn + exactness (limit 5) =="
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method draft_model \
    --draft-model "$S2" --k 5 --model "$TARGET" --temperature greedy \
    --batch 1 --runs 1 --limit 5 --max-new-tokens 64 --warmup 0 \
    --metrics-out "" --outputs-out ""
$PY -m src.serving.bench_vllm frozen/tb_eval.parquet --per-turn \
    --method draft_model --draft-model "$S2" --k 5 --model "$TARGET" \
    --temperature greedy --batch 1 --runs 1 --limit 5 --max-new-tokens 64 \
    --warmup 0 --metrics-out "" --outputs-out ""
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --exactness \
    --method draft_model --draft-model "$S2" --k 5 --model "$TARGET" \
    --limit 5 --max-new-tokens 64 --metrics-out "" --outputs-out "" \
  || echo "  [note] smoke exactness mismatch (recorded; plan §9 — not fatal)"
echo "  smoke OK — spec config accepted, per-turn TB path + exactness run"

# -- Phase B: instrumented stage-wise rows + checkpoint trajectory -----------
echo "== Phase B: instrumented tau/alpha rows (k=5 greedy, $TAU_LIMIT prompts) =="
tau_run "$S2" stage2_final_on_xlam frozen/xlam_eval.parquet
tau_run "$S2" stage2_final_on_tb frozen/tb_eval.parquet
for CK in $(ls -d "$S2DIR"/step_* 2>/dev/null | sort -r); do
  NAME=$(basename "$CK")
  [ "$NAME" = "step_400" ] && continue  # final re-saves the same weights
  tau_run "$CK" "stage2_${NAME}_on_xlam" frozen/xlam_eval.parquet
  tau_run "$CK" "stage2_${NAME}_on_tb" frozen/tb_eval.parquet
done
echo "Phase B done — reports in results/events/stage2_*_report.json"

# -- Phase C: vLLM wall-clock battery ---------------------------------------
echo "== Phase C1: exactness gates (50 prompts, greedy b1; recorded, not fatal) =="
for PAIR in "stage1 $STAGE1" "tbkd $TBKD" "stage2 $S2"; do
  set -- $PAIR
  exactness_gate "$1" "$2"
done

echo "== Phase C2: k5-greedy-b1 wall-clock rows for stage1 + tbkd =="
for PAIR in "stage1 $STAGE1" "tbkd $TBKD"; do
  set -- $PAIR
  bench "${1}_k5_greedy_b1" frozen/xlam_eval.parquet --method draft_model \
      --draft-model "$2" --k 5 --model "$TARGET" --temperature greedy \
      --batch 1 --runs 3 --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS"
  bench "tb_${1}_draft_k5_greedy_b1" frozen/tb_eval.parquet --per-turn \
      --method draft_model --draft-model "$2" --k 5 --model "$TARGET" \
      --temperature greedy --batch 1 --runs 3 --limit "$TB_LIMIT" \
      --max-new-tokens "$MAX_NEW_TOKENS"
done

echo "== Phase C3: untuned full k x temp grid (batch 1, §5 hours 9-12) =="
for K in $K_VALUES; do
  for T in greedy 1.0; do
    bench "untuned_k${K}_${T}_b1" frozen/xlam_eval.parquet --method draft_model \
        --draft-model "$UNTUNED" --k "$K" --model "$TARGET" \
        --temperature "$T" --batch 1 --runs 3 --limit "$LIMIT" \
        --max-new-tokens "$MAX_NEW_TOKENS"
  done
done

echo "== Phase C4: full stage2 battery via run_sweep.sh (grid + n-gram k sweep + batch sweep + TB transfer) =="
BEST_K="$BEST_K" LIMIT="$LIMIT" TB_LIMIT="$TB_LIMIT" \
    bash scripts/run_sweep.sh "$S2" stage2

echo "== ALL EVALS DONE ($(date -u +%H:%M:%S)) =="
echo "Morning read-out order:"
echo "  1. results/events/stage2_final_on_{xlam,tb}_k5_report.json  (headline stage-wise rows)"
echo "  2. results/events/stage2_step*_on_*_k5_report.json          (checkpoint trajectory — pick the Stage-2 row: final vs an earlier step)"
echo "  3. results/exactness/{stage1,tbkd,stage2}_k5.json            (gates must be token-identical)"
echo "  4. results/vllm/*.jsonl + results/metrics.jsonl              (wall-clock: grids, batch sweep, TB transfer)"
echo "  comparators (amended §2): TB-KD 4.07 xLAM / 3.04 TB"
