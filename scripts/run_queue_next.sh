#!/usr/bin/env bash
# Queued chain after the room-probe battery (user-directed 2026-09-27):
#   1. wait for room_probe to finish (TB-500 baselines own the GPU)
#   2. phase-timing pass: xLAM draft k=5 + k=9 with C8 events -> the
#      draft-vs-verify cost split ("is the draft the bottleneck at k=9")
#   3. Stage-1 KD warm-start on the assembled 4,548-record dataset
# All code is committed and locally tested; no tests run here.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/queue_next_$(date -u +%Y%m%d-%H%M%S)_$$.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PY=".venv-h100/bin/python"
TARGET="Qwen/Qwen2.5-Coder-14B-Instruct"
DRAFT="drafts/coder-0.5b-padded"
OUTDIR="results/events"

echo "== [1/3] waiting for room_probe battery =="
while pgrep -f "run_room_probe" > /dev/null; do sleep 30; done
echo "   room probe finished"

echo "== [2/3] phase-timing pass (C8): xLAM draft k=5 and k=9 =="
for K in 5 9; do
  $PY -m src.serving.instrumented_spec frozen/xlam_eval.parquet \
      --proposer draft --draft-model "$DRAFT" --target-model "$TARGET" \
      --k "$K" --limit 100 --dtype bfloat16 --device cuda:0 \
      --out "$OUTDIR/phasetiming_draft05_k${K}.jsonl" \
      --outputs-out "$OUTDIR/phasetiming_draft05_k${K}_outputs.jsonl" \
      --meta-out "$OUTDIR/phasetiming_draft05_k${K}_meta.json"
  $PY -m src.analysis.eval_acceptance "$OUTDIR/phasetiming_draft05_k${K}.jsonl" \
      --records frozen/xlam_eval.parquet --k "$K" --subcut \
      --out "$OUTDIR/phasetiming_draft05_k${K}_report.json" \
      --per-prompt "$OUTDIR/phasetiming_draft05_k${K}_per_prompt.jsonl"
done

echo "== [3/3] Stage-1 KD warm-start (plan §2 Stage 1, §5 hour 4.5-5.5) =="
$PY -m src.training.kd_warmstart \
    --data data/processed/stage1_kd \
    --draft "$DRAFT" \
    --out checkpoints/stage1 \
    --dtype bfloat16 --device cuda:0 \
    --save-vocab-padded \
    --metrics-out results/metrics.jsonl

echo "queue complete: phase-timing reports + stage1 checkpoint in checkpoints/stage1"
