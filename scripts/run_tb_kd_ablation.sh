#!/usr/bin/env bash
# TB-prefix KD ablation (added 2026-09-27, the Stage-2 off-policy
# comparator): the Stage-1 recipe — target greedy datagen + top-20
# logprobs, then kd_warmstart training — applied to TB boundary contexts
# instead of xLAM. Same engine (gen_stage1), same trainer (kd_warmstart):
# only the prompt source + validation differ (build_tb_distill.py).
#
# Scale: 5,000 boundary contexts (matches Stage-1's xLAM scale for a clean
# same-recipe/different-context comparison).
#
# Usage: bash scripts/run_tb_kd_ablation.sh
# Env overrides: PY, TARGET, LIMIT, LOGPROBS, MAX_NEW_TOKENS, MAX_CTX
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/tb_kd_ablation_$(date -u +%Y%m%d-%H%M%S)_$$.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PY="${PY:-.venv-h100/bin/python}"
TARGET="${TARGET:-Qwen/Qwen2.5-Coder-14B-Instruct}"
LIMIT="${LIMIT:-5000}"
LOGPROBS="${LOGPROBS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_CTX="${MAX_CTX:-4096}"
OUTDIR="results/tb_ablation"
DRAFT_SRC="Qwen/Qwen2.5-Coder-0.5B-Instruct"

mkdir -p "$OUTDIR" data/processed/tb_ablation

echo "== [1/4] smoke: 5 boundary contexts through the full path =="
$PY - <<EOF
import sys
sys.path.insert(0, ".")
from src.data_prep.build_tb_distill import tb_boundary_contexts, validate_tb_generation
from src.data_prep.render import get_tool_call_tags, get_tokenizer
ctxs = tb_boundary_contexts(limit=5, max_ctx=$MAX_CTX)
print("smoke contexts:", [c["query_id"] for c in ctxs])
tok = get_tokenizer()
tags = get_tool_call_tags()
for c in ctxs:
    tail = tok.decode(c["prompt_ids"][-4:])
    assert tail.endswith("assistant\n"), c["query_id"]
print("smoke OK — every context ends at an assistant header")
EOF

echo "== [2/4] TB boundary-context datagen (target greedy + top-$LOGPROBS logprobs, $LIMIT contexts) =="
$PY - <<EOF
import sys, time
sys.path.insert(0, ".")
from src.data_prep.build_tb_distill import tb_boundary_contexts
from src.serving.gen_stage1 import VLLMGenEngine, normalize_output, write_generations
from src.serving.bench_vllm import append_metrics
from src.data_prep.render import get_tokenizer
import json, subprocess

LIMIT = $LIMIT
LOGPROBS = $LOGPROBS
MAX_NEW = $MAX_NEW_TOKENS
ctxs = tb_boundary_contexts(limit=LIMIT, max_ctx=$MAX_CTX)
print("contexts:", len(ctxs))
eos_id = int(get_tokenizer().eos_token_id)
engine = VLLMGenEngine("$TARGET", LOGPROBS, "bfloat16", 16384, 0.90)
t0 = time.perf_counter()
outputs = [normalize_output(o, eos_id) for o in engine.generate(
    [c["prompt_ids"] for c in ctxs],
    {"max_tokens": MAX_NEW, "temperature": 0.0, "seed": 1234,
     "detokenize": False},
)]
wall = time.perf_counter() - t0
write_generations("$OUTDIR/tb_target_gen.jsonl", ctxs, outputs)
engine.teardown()
n_tok = sum(len(o["token_ids"]) for o in outputs)
meta = {"kind": "tb_kd_ablation_datagen", "n_prompts": len(ctxs),
        "n_tokens": n_tok, "wall_s": round(wall, 2)}
append_metrics("results/metrics.jsonl", meta)
print(json.dumps(meta, indent=2))
EOF

echo "== [3/4] assemble the TB KD dataset (T3 validation) =="
$PY - <<EOF
import sys, json
sys.path.insert(0, ".")
from datasets import Dataset
from src.data_prep.build_tb_distill import tb_boundary_contexts, validate_tb_generation
from src.data_prep.build_distill_data import (
    _gen_token_offsets, find_gen_payloads, find_gen_tag_spans, _rle,
)
from src.data_prep.render import (
    REGION_CONTEXT, REGION_PROSE, REGION_TAG, REGION_TOOL_CALL,
    get_tokenizer, get_tool_call_tags,
)

# NOTE: assembly mirrors build_distill_data.build_gen_record but with T3
# validation (prose kept). Kept inline here deliberately: the ablation
# must not modify the golden-tested Stage-1 pipeline.
tok = get_tokenizer()
tags = get_tool_call_tags()
ctxs = tb_boundary_contexts(limit=$LIMIT, max_ctx=$MAX_CTX)
gens = [json.loads(l) for l in open("$OUTDIR/tb_target_gen.jsonl")]
by_qid = {g["query_id"]: g for g in gens}

records, drops = [], {"prose_or_valid": 0, "invalid": 0}
for c in ctxs:
    g = by_qid[c["query_id"]]
    p_ids = list(c["prompt_ids"])
    g_ids = list(g["output_ids"])
    n_prompt = len(p_ids)
    text = tok.decode(g_ids)
    offsets, exact = _gen_token_offsets(g_ids)
    payload_spans = find_gen_payloads(text, tags)
    tag_spans = find_gen_tag_spans(text, tags)
    v = validate_tb_generation(text, c["tools"], tags)
    if not v["valid"] or not exact:
        drops["invalid"] += 1
        continue
    drops["prose_or_valid"] += 1
    regions = [REGION_CONTEXT] * n_prompt
    for s, e in offsets:
        r = REGION_PROSE
        for ts, te in tag_spans:
            if ts <= s < te:
                r = REGION_TAG
                break
        if r == REGION_PROSE:
            for ps, pe in payload_spans:
                if ps <= s < pe:
                    r = REGION_TOOL_CALL
                    break
        regions.append(r)
    records.append({
        "input_ids": p_ids + g_ids,
        "labels": [-100] * n_prompt + list(g_ids),
        "regions": _rle(regions),
        "n_prompt_tokens": n_prompt,
        "n_tokens": n_prompt + len(g_ids),
        "n_tool_calls": len(payload_spans),
        "n_tools": c["n_tools"],
        "query_id": c["query_id"],
        "gen_ids": g_ids,
        "finish_reason": g.get("finish_reason"),
        "gen_logprob_token_ids": [
            [t for t, _ in lp] if lp is not None else None
            for lp in g.get("logprobs", [])
        ],
        "gen_logprob_values": [
            [v2 for _, v2 in lp] if lp is not None else None
            for lp in g.get("logprobs", [])
        ],
    })

out = "data/processed/tb_ablation/tb_kd"
Dataset.from_list(records).save_to_disk(out)
print(json.dumps({"n_kept": len(records), "drops": drops, "out": out},
                 indent=2))
EOF

echo "== [4/4] KD training on the TB dataset (kd_warmstart, fp32) =="
$PY -m src.training.kd_warmstart \
    --data data/processed/tb_ablation/tb_kd \
    --draft "$DRAFT_SRC" \
    --out checkpoints/tb_stage1 \
    --dtype float32 --device cuda:0 \
    --save-vocab-padded \
    --metrics-out results/metrics.jsonl

echo "TB KD ablation complete — checkpoint in checkpoints/tb_stage1/final"
echo "next: tau probe on TB-500 with this draft (the ablation's number)"
