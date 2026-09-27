#!/usr/bin/env bash
# Untuned-τ runs (plan §5 hours 1.5–3, §4.1): the instrumented HF loop on a
# 100-prompt xLAM-500 subset for the fixed 0.5B draft (§8.1) + the n-gram
# proposer — acceptance internals (α, τ, per-position, region splits) for
# the untuned row of the stage-wise table. Wall-clock always comes from
# vLLM (run_baselines.sh); this script is the instrument.
#
# Usage: bash scripts/run_tau.sh
# Env overrides: PY, TARGET, DRAFT05, LIMIT, K, DEVICE, OUTDIR
#
# After this script: the analyzer (eval_acceptance.py) turns each events
# file into the §4.1–4.3 report (the untuned baseline row).

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

PY="${PY:-.venv-h100/bin/python}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-14B-Instruct}"
DRAFT05="${DRAFT05:-drafts/coder-0.5b-padded}"
LIMIT="${LIMIT:-100}"
K="${K:-5}"
DEVICE="${DEVICE:-cuda:0}"
OUTDIR="${OUTDIR:-results/events}"

mkdir -p "$OUTDIR"

run_one() {  # run_one <tag> <proposer> [draft-model]
  local TAG=$1 PROP=$2 DM=${3:-}
  local EXTRA=""
  [ "$PROP" = "draft" ] && EXTRA="--draft-model $DM"
  echo "== τ run: $TAG (k=$K, limit=$LIMIT) =="
  $PY -m src.serving.instrumented_spec frozen/xlam_eval.parquet \
      --proposer "$PROP" $EXTRA --target-model "$TARGET" \
      --k "$K" --limit "$LIMIT" --dtype bfloat16 --device "$DEVICE" \
      --out "$OUTDIR/${TAG}_k${K}.jsonl" \
      --outputs-out "$OUTDIR/${TAG}_k${K}_outputs.jsonl" \
      --meta-out "$OUTDIR/${TAG}_k${K}_meta.json"
  $PY -m src.analysis.eval_acceptance "$OUTDIR/${TAG}_k${K}.jsonl" \
      --records frozen/xlam_eval.parquet --k "$K" --subcut \
      --out "$OUTDIR/${TAG}_k${K}_report.json" \
      --per-prompt "$OUTDIR/${TAG}_k${K}_per_prompt.jsonl"
}

run_one draft05 draft "$DRAFT05"
run_one ngram ngram

echo "τ runs done — reports in $OUTDIR/*_report.json (untuned 0.5B baseline row)"
echo "next (§5): bash scripts/run_stage1_datagen.sh"
