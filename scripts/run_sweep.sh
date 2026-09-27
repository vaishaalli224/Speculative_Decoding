#!/usr/bin/env bash
# Sweeps (plan §4.5, hours 8–11, P0): k x temperature grid at batch 1 for
# the given draft; batch sweep for the best config + AR + n-gram; TB-500
# transfer bench. Run after distillation with DRAFT=<trained draft path>.
#
# Usage: bash scripts/run_sweep.sh <draft-path> [tag]
#   e.g.  bash scripts/run_sweep.sh checkpoints/stage2_0.5b stage2_0.5b
# Env overrides: MODEL, LIMIT, TB_LIMIT, MAX_NEW_TOKENS, METRICS, K_VALUES

set -euo pipefail
cd "$(dirname "$0")/.."

# --- auto-log: every invocation writes its own log ------------------------
# logs/<script>_<UTC-timestamp>_<pid>.log + logs/latest.log symlink, so
# progress is always `tail -f logs/latest.log` away. Composes safely with
# an outer `nohup ... > file 2>&1` (tee preserves that copy too).
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${0%.sh}")_$(date -u +%Y%m%d-%H%M%S)_$$.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

DRAFT="${1:?usage: run_sweep.sh <draft-path> [tag]}"
TAG="${2:-$(basename "$DRAFT")}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
LIMIT="${LIMIT:-200}"
TB_LIMIT="${TB_LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
METRICS="${METRICS:-results/metrics.jsonl}"
K_VALUES="${K_VALUES:-3 5 7}"
PY="${PY:-.venv-h100/bin/python}"

echo "== k x temperature grid at batch 1, xLAM-eval (§4.5) =="
for K in $K_VALUES; do
  for T in greedy 1.0; do
    $PY -m src.serving.bench_vllm frozen/xlam_eval.parquet \
        --method draft_model --draft-model "$DRAFT" --k "$K" \
        --model "$MODEL" --temperature "$T" --batch 1 --runs 3 \
        --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" \
        --outputs-out "results/vllm/${TAG}_k${K}_${T}_b1.jsonl" \
        --metrics-out "$METRICS"
  done
done

echo "== n-gram k sweep, greedy batch 1 (§4.6: same k grid) =="
for K in $K_VALUES; do
  $PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method ngram \
      --k "$K" --model "$MODEL" --temperature greedy --batch 1 --runs 3 \
      --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" \
      --outputs-out "results/vllm/ngram_k${K}_greedy_b1.jsonl" \
      --metrics-out "$METRICS"
done

echo "== batch sweep, greedy, best-config k (§4.5): $TAG + AR + n-gram =="
BEST_K="${BEST_K:-5}"
for CFG in "$TAG draft_model $DRAFT" "ar ar -" "ngram ngram -"; do
  set -- $CFG
  NAME=$1; METHOD=$2; DM=$3
  for B in 1 8 32; do
    EXTRA=""
    [ "$METHOD" = "draft_model" ] && EXTRA="--draft-model $DM"
    $PY -m src.serving.bench_vllm frozen/xlam_eval.parquet \
        --method "$METHOD" $EXTRA --k "$BEST_K" --model "$MODEL" \
        --temperature greedy --batch "$B" --runs 3 \
        --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" \
        --outputs-out "results/vllm/${NAME}_k${BEST_K}_greedy_b${B}.jsonl" \
        --metrics-out "$METRICS"
  done
done

echo "== TB-500 transfer bench, greedy, per-turn (§4.2/§3.2) =="
for METHOD in ar ngram draft_model; do
  EXTRA=""
  [ "$METHOD" = "draft_model" ] && EXTRA="--draft-model $DRAFT"
  $PY -m src.serving.bench_vllm frozen/tb_eval.parquet --per-turn \
      --method "$METHOD" $EXTRA --k "$BEST_K" --model "$MODEL" \
      --temperature greedy --batch 1 --runs 3 \
      --limit "$TB_LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" \
      --outputs-out "results/vllm/tb_${TAG}_${METHOD}_k${BEST_K}_greedy_b1.jsonl" \
      --metrics-out "$METRICS"
done

echo "sweep done — metrics in $METRICS"
