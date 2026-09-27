#!/usr/bin/env bash
# Day-0 setup (plan §5, hours 0–1.5, P0): fresh-clone -> verified
# environment on the GPU host. Everything after this script is measurement,
# not setup. Idempotent: safe to re-run after a partial failure.
#
# Usage: bash scripts/setup_day0.sh
# Env overrides: REPO_URL, BRANCH, VENV_DIR, PY_VER
#
# Steps:
#   1. clone repo (or git pull if already cloned) + checkout the pinned branch
#   2. python venv + pip install -r requirements-h100.txt (pinned vLLM)
#   3. download raw datasets (xLAM + ToolBench mirror) — HF_TOKEN from .env
#   4. freeze_splits verify — every checksum + held-out property from
#      frozen/ alone (GPU-day step 0 per plan §6.3; run AFTER raw download
#      because the manifest checksums the raw arrow files too)
#   5. model downloads: target + both draft candidates (vLLM prefill,
#      not from_pretrained — same code path the bench uses)
#   6. vLLM smoke test: serve + speculative_config draft-model + ngram
#      (the one thing that cannot be pre-verified locally, plan §1.5)
#
# Exits non-zero on the first failure — fix and re-run from the top.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_URL="${REPO_URL:-git@github-personal:vaishaalli224/Speculative_Decoding.git}"
BRANCH="${BRANCH:-main}"
VENV_DIR="${VENV_DIR:-.venv-h100}"
PY_VER="${PY_VER:-3.12}"

echo "== [1/6] repo =="
if [ ! -d .git ]; then
  git clone "$REPO_URL" .
fi
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || echo "  (no upstream update)"
git log --oneline -1

echo "== [2/6] venv + pinned deps (requirements-h100.txt) =="
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PY_VER" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -q -r requirements-h100.txt
"$VENV_DIR/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
"$VENV_DIR/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "== [3/6] raw datasets (xLAM + ToolBench mirror) =="
if [ ! -f data/raw/download_report.json ]; then
  "$VENV_DIR/bin/python" src/data_prep/download_datasets.py
else
  echo "  already downloaded — re-running freeze verify (step 4) catches corruption"
fi

echo "== [4/6] frozen-split verify (GPU-day step 0, plan §6.3) =="
"$VENV_DIR/bin/python" -m src.data_prep.freeze_splits verify

echo "== [5/6] model downloads =="
"$VENV_DIR/bin/python" - <<'EOF'
import os
from dotenv import load_dotenv
load_dotenv(".env")
from huggingface_hub import snapshot_download
for repo in (
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
):
    p = snapshot_download(repo)
    n = sum(1 for f in os.scandir(p) if f.is_file())
    print(f"  {repo} -> {p} ({n} files)")
EOF

echo "== [6/6] vLLM speculative_config smoke test (plan §1.5) =="
# 5 frozen prompts through each spec method — the pinned build must accept
# both §1.5 config shapes before any number from this harness is trusted.
"$VENV_DIR/bin/python" - <<'EOF'
from src.serving.spec_configs import draft_model_config, ngram_config
import pyarrow.parquet as pq

prompts = pq.read_table("frozen/xlam_eval.parquet").column("input_ids")[:5].to_pylist()

import vllm
from vllm import SamplingParams

for name, cfg in (
    ("draft_model", draft_model_config("Qwen/Qwen2.5-Coder-0.5B-Instruct", 5)),
    ("ngram", ngram_config(5)),
):
    llm = vllm.LLM(
        model="Qwen/Qwen2.5-Coder-14B-Instruct",
        speculative_config=cfg,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.90,
    )
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        SamplingParams(temperature=0.0, max_tokens=32, detokenize=False),
    )
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"  {name}: {len(outs)} completions, {n} tokens — OK")
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
print("smoke test passed — both speculative_config shapes work on this build")
EOF

echo
echo "setup_day0 complete. Next: bash scripts/run_baselines.sh"
