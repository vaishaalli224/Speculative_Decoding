#!/usr/bin/env bash
# Stage-2 on-policy GKD distillation (plan §5 hours 5.5-9, P0): the draft
# (Stage 1's artifact) samples its own continuations on mixed xLAM + TB
# prefix contexts; the frozen 14B scores; hourly checkpoints with a quick
# tau probe. Runs AFTER run_stage1_datagen.sh (needs checkpoints/stage1/
# final + the assembled Stage-1 KD dataset for the mixing stream).
#
# The pools are rebuilt FIRST from this repo's builder (frozen indices ->
# raw -> render): an earlier host build wrote the pre-reconciliation
# schema, and the trainer's boundary cut (n_prompt_tokens / per-turn TB)
# requires the new records. Rebuilding is cheap (~minutes, CPU) and makes
# the script self-contained on a fresh host.
#
# Usage: bash scripts/run_stage2_gkd.sh
# Env overrides: PY, TARGET, DRAFT, XLAMCTX, TBCTX, S1DATA, LIMIT_CTX,
#                STEPS, CKPT_EVERY, TAU_LIMIT, OUTDIR, METRICS, DTYPE,
#                DEVICE

set -euo pipefail
cd "$(dirname "$0")/.."

# --- auto-log: every invocation writes its own log ------------------------
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${0%.sh}")_$(date -u +%Y%m%d-%H%M%S)_$$.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PY="${PY:-.venv-h100/bin/python}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-14B-Instruct}"
DRAFT="${DRAFT:-checkpoints/stage1/final}"
XLAMCTX="${XLAMCTX:-data/processed/xlam/stage2_contexts}"
TBCTX="${TBCTX:-data/processed/toolbench/prefixes}"
S1DATA="${S1DATA:-data/processed/stage1_kd}"
LIMIT_CTX="${LIMIT_CTX:-20000}"   # context-pool head; the stream cycles
STEPS="${STEPS:-400}"
CKPT_EVERY="${CKPT_EVERY:-100}"   # ~1 GPU hour at the default step pace
TAU_LIMIT="${TAU_LIMIT:-25}"
OUTDIR="${OUTDIR:-checkpoints/stage2}"
METRICS="${METRICS:-results/metrics.jsonl}"
DTYPE="${DTYPE:-float32}"          # G7: fp32 master (the H100-measured
DEVICE="${DEVICE:-cuda:0}"         # bf16-AdamW NaN applies here too)

echo "== rebuild the context pools from frozen indices (new schema) =="
$PY -m src.data_prep.build_stage2_contexts xlam
$PY -m src.data_prep.build_stage2_contexts tb

echo "== smoke: 2 steps through the full path incl. checkpoint + tau =="
$PY -m src.training.onpolicy_gkd \
    --draft "$DRAFT" --target "$TARGET" \
    --xlam-ctx "$XLAMCTX" --tb-ctx "$TBCTX" --s1-data "$S1DATA" \
    --steps 2 --micro-bs 1 --grad-accum 1 --max-new-tokens 32 \
    --limit-ctx 8 --ckpt-every 1 --tau-limit 2 --tau-max-new 16 \
    --dtype "$DTYPE" --device "$DEVICE" \
    --out "$OUTDIR/_smoke"
echo "  smoke OK — sampling, scoring, mixing, checkpoint, tau all ran"

echo "== full Stage-2 run (stages $STEPS, checkpoints every $CKPT_EVERY) =="
$PY -m src.training.onpolicy_gkd \
    --draft "$DRAFT" --target "$TARGET" \
    --xlam-ctx "$XLAMCTX" --tb-ctx "$TBCTX" --s1-data "$S1DATA" \
    --steps "$STEPS" --max-new-tokens 512 --max-ctx 4096 --tb-frac 0.5 \
    --mix-sft-frac 0.25 --div reverse_kl --sft-weight 0.1 \
    --limit-ctx "$LIMIT_CTX" \
    --ckpt-every "$CKPT_EVERY" --tau-limit "$TAU_LIMIT" \
    --dtype "$DTYPE" --device "$DEVICE" \
    --out "$OUTDIR"

echo "Stage-2 done — final draft at $OUTDIR/final (vocab-padded, servable"
echo "by vLLM as-is); tau trail in $OUTDIR/tau_*.json; the run's samples"
echo "record in $OUTDIR/samples.jsonl."
echo "Next: re-benchmark untuned / Stage 1 / Stage 2 (runbook hours 9-12)."
