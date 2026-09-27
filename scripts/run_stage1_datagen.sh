#!/usr/bin/env bash
# Stage-1 data generation (plan §5 hours 3.5-4.5, P0): the frozen target's
# greedy continuations + top-20 logprobs on the 5,000 frozen Stage-1 xLAM
# contexts, then assembly into the KD dataset consumed by
# training/kd_warmstart.py. Runs AFTER run_baselines.sh + pick_draft.py
# (hour 3.5+) — the GPU is otherwise idle during generation either way.
#
# The committed artifact is the generation JSONL (the assembled dataset is
# derived, gitignored, and rebuildable with the assemble command below).
#
# Usage: bash scripts/run_stage1_datagen.sh
# Env overrides: PY, TARGET, LIMIT, MAX_NEW_TOKENS, LOGPROBS, METRICS, OUTDIR

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-h100/bin/python}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-14B-Instruct}"
LIMIT="${LIMIT:-}"                       # empty = all 5,000
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOGPROBS="${LOGPROBS:-20}"
METRICS="${METRICS:-results/metrics.jsonl}"
OUTDIR="${OUTDIR:-results/stage1}"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
mkdir -p "$OUTDIR" data/processed/stage1_kd

echo "== smoke: 5 prompts through the full path (fail fast) =="
$PY -m src.serving.gen_stage1 --engine vllm --model "$TARGET" \
    --logprobs "$LOGPROBS" --max-new-tokens "$MAX_NEW_TOKENS" --limit 5 \
    --out "$OUTDIR/stage1_smoke.jsonl" --meta-out "$OUTDIR/stage1_smoke_meta.json" \
    --metrics-out ""
$PY -m src.data_prep.build_distill_data assemble "$OUTDIR/stage1_smoke.jsonl" \
    --limit 5 --out data/processed/stage1_kd_smoke
echo "  smoke OK — logprobs shape and EOS handling verified"

echo "== full Stage-1 generation ($([ -n "$LIMIT" ] && echo "$LIMIT" || echo 5000) prompts) =="
$PY -m src.serving.gen_stage1 --engine vllm --model "$TARGET" \
    --logprobs "$LOGPROBS" --max-new-tokens "$MAX_NEW_TOKENS" $LIMIT_ARG \
    --out "$OUTDIR/stage1_target_gen.jsonl" \
    --meta-out "$OUTDIR/stage1_target_gen_meta.json" \
    --metrics-out "$METRICS"

echo "== assemble the KD dataset (validate + drop rate, plan §9) =="
$PY -m src.data_prep.build_distill_data assemble "$OUTDIR/stage1_target_gen.jsonl" \
    $LIMIT_ARG --out data/processed/stage1_kd
$PY -m src.data_prep.build_distill_data stats data/processed/stage1_kd

echo "Stage-1 data done — commit $OUTDIR/stage1_target_gen.jsonl (the source"
echo "of truth); the dataset in data/processed/stage1_kd is rebuildable."
echo "Next: training/kd_warmstart.py"
