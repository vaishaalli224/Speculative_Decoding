#!/usr/bin/env bash
# Room-to-improve battery (user-directed 2026-09-27):
#   1. xLAM draft k=9 probe -- deep-position acceptance (alpha_6..9):
#      is there room at larger k?
#   2. TB-500 untuned draft k=5 -- the held-out-tools / multi-turn baseline
#   3. TB-500 ngram k=5 -- the cheap baseline on the hard distribution
# Same 100-prompt frozen subsets and analyzer protocol as run_tau.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/room_probe_$(date -u +%Y%m%d-%H%M%S)_$$.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PY="${PY:-.venv-h100/bin/python}"
DRAFT="${DRAFT:-drafts/coder-0.5b-padded}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-14B-Instruct}"
OUTDIR="${OUTDIR:-results/events}"
LIMIT="${LIMIT:-100}"

run_one() {  # run_one <records> <tag> <proposer> <k> [draft]
  local REC=$1 TAG=$2 PROP=$3 K=$4 DM=${5:-}
  local EXTRA=""
  [ "$PROP" = "draft" ] && EXTRA="--draft-model $DM"
  echo "== $TAG: $PROP k=$K limit=$LIMIT =="
  $PY -m src.serving.instrumented_spec "$REC" \
      --proposer "$PROP" $EXTRA --target-model "$TARGET" \
      --k "$K" --limit "$LIMIT" --dtype bfloat16 --device cuda:0 \
      --out "$OUTDIR/${TAG}_k${K}.jsonl" \
      --outputs-out "$OUTDIR/${TAG}_k${K}_outputs.jsonl" \
      --meta-out "$OUTDIR/${TAG}_k${K}_meta.json"
  $PY -m src.analysis.eval_acceptance "$OUTDIR/${TAG}_k${K}.jsonl" \
      --records "$REC" --k "$K" --subcut \
      --out "$OUTDIR/${TAG}_k${K}_report.json" \
      --per-prompt "$OUTDIR/${TAG}_k${K}_per_prompt.jsonl"
}

run_one frozen/xlam_eval.parquet draft05_k9 draft 9 "$DRAFT"
run_one frozen/tb_eval.parquet tb_draft05 draft 5 "$DRAFT"
run_one frozen/tb_eval.parquet tb_ngram ngram 5

echo "room-probe battery done — reports in $OUTDIR"
