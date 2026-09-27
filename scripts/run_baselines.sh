#!/usr/bin/env bash
# Baselines (plan §5, hours 1.5–3, P0): AR-only batch sweep, n-gram,
# untuned drafts at k=5 greedy, exactness gate. Every run appends to
# results/metrics.jsonl (B7); outputs land in results/vllm/. Starts with
# a 5-prompt smoke through each spec config so a rejected speculative_
# config fails in minutes, not after the AR sweep.
#
# Usage: bash scripts/run_baselines.sh
# Env overrides: MODEL, DRAFT (0.5B path), LIMIT, MAX_NEW_TOKENS, METRICS, PY

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
DRAFT05="${DRAFT05:-drafts/coder-0.5b-padded}"
LIMIT="${LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
METRICS="${METRICS:-results/metrics.jsonl}"
PY="${PY:-.venv-h100/bin/python}"

echo "== smoke: 5 prompts through each spec config (fail fast, plan §1.5) =="
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method draft_model \
    --draft-model "$DRAFT05" --k 5 --model "$MODEL" --temperature greedy \
    --batch 1 --runs 1 --limit 5 --max-new-tokens 64 --warmup 0 \
    --metrics-out "" --outputs-out ""
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method ngram \
    --k 5 --model "$MODEL" --temperature greedy \
    --batch 1 --runs 1 --limit 5 --max-new-tokens 64 --warmup 0 \
    --metrics-out "" --outputs-out ""
echo "  both spec configs accepted — proceeding to full baselines"

echo "== AR-only baseline, batch 1/8/32 (§4.2) =="
for B in 1 8 32; do
  $PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method ar \
      --model "$MODEL" --temperature greedy --batch "$B" --runs 3 \
      --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" \
      --outputs-out "results/vllm/ar_greedy_b${B}.jsonl" \
      --metrics-out "$METRICS"
done

echo "== n-gram baseline, k=5 greedy, batch 1 (§4.6) =="
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method ngram \
    --k 5 --model "$MODEL" --temperature greedy --batch 1 --runs 3 \
    --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --outputs-out results/vllm/ngram_k5_greedy_b1.jsonl \
    --metrics-out "$METRICS"

echo "== untuned draft (0.5B, fixed per §8.1), k=5 greedy, batch 1 =="
for D in "$DRAFT05"; do
  $PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --method draft_model \
      --draft-model "$D" --k 5 --model "$MODEL" --temperature greedy \
      --batch 1 --runs 3 --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" \
      --outputs-out "results/vllm/draft_${D##*/}_k5_greedy_b1.jsonl" \
      --metrics-out "$METRICS"
done

echo "== exactness gate (§4.4): untuned 0.5B draft vs AR, greedy batch 1 =="
# 50-prompt subset per plan §4.4 fallback note; reuses nothing — the AR side
# runs in-command and tears down before the spec engine loads.
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --exactness \
    --method draft_model --draft-model "$DRAFT05" --k 5 --model "$MODEL" \
    --limit 50 --max-new-tokens "$MAX_NEW_TOKENS" \
    --outputs-out results/vllm/draft05_k5_exact.jsonl \
    --exactness-out results/exactness/draft05_k5.json \
    --metrics-out "$METRICS"

echo "== exactness gate: n-gram (§4.4: 'Also verify the n-gram method') =="
$PY -m src.serving.bench_vllm frozen/xlam_eval.parquet --exactness \
    --method ngram --k 5 --model "$MODEL" \
    --limit 50 --max-new-tokens "$MAX_NEW_TOKENS" \
    --outputs-out results/vllm/ngram_k5_exact.jsonl \
    --exactness-out results/exactness/ngram_k5.json \
    --metrics-out "$METRICS"

echo "baselines done — metrics in $METRICS"
