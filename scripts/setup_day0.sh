#!/usr/bin/env bash
# Day-0 setup (plan §5, hours 0–1.5, P0): run INSIDE a checkout of this
# repo on the GPU host. Everything after this script is measurement,
# not setup. Idempotent: safe to re-run after a partial failure.
#
# Usage (on the rental host):
#   git clone <your-transfer-method> && cd Speculative_Decoding
#   bash scripts/setup_day0.sh
# Env overrides: VENV_DIR, PY_BIN
#
# Steps:
#   0. sanity: repo checkout present (plan.md + frozen/); git pull if a
#      remote exists (mid-day code drops land here — re-run this script's
#      step 0 alone via `git pull` when new commits are pushed)
#   1. python venv + pip install -r requirements-h100.txt (pinned vLLM)
#   2. download raw datasets (xLAM + ToolBench mirror — public; anonymous
#      download works, HF_TOKEN only helps rate limits)
#   3. freeze_splits verify — every checksum + held-out property from
#      frozen/ alone (GPU-day step 0 per §6.3; runs after raw download
#      because the manifest checksums the raw arrow files too)
#   4. model downloads: target + both draft candidates (public repos)
#   5. vLLM smoke: each §1.5 speculative_config shape in its own
#      subprocess — process exit guarantees full VRAM release between
#      engines, so the second 0.90-utilization engine can't OOM on the
#      first one's leftovers
#
# Exits non-zero on the first failure — fix and re-run from the top.

set -euo pipefail

echo "== [0/5] repo =="
if [ ! -f plan.md ] || [ ! -d frozen ] || [ ! -d scripts ]; then
  echo "not run from a repo checkout (plan.md/frozen/scripts missing)" >&2
  echo "clone the repo first, then: cd Speculative_Decoding && bash scripts/setup_day0.sh" >&2
  exit 1
fi
git pull --ff-only 2>/dev/null || echo "  (no upstream remote or offline — continuing at current commit)"
git log --oneline -1

VENV_DIR="${VENV_DIR:-.venv-h100}"

# python: vllm 0.30.0 needs >=3.10,<3.15; prefer 3.12 (the pin all local
# testing used), fall back through whatever the instance image ships
if [ -z "${PY_BIN:-}" ]; then
  for c in python3.12 python3.11 python3.13 python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 && PY_BIN="$c" && break
  done
fi
if [ -z "${PY_BIN:-}" ]; then
  echo "no python3 found in PATH — set PY_BIN=<python>" >&2
  exit 1
fi
"$PY_BIN" - <<'EOF'
import sys
v = sys.version_info
assert (3, 10) <= (v.major, v.minor) < (3, 15), \
    f"need python >=3.10,<3.15 (vllm 0.30.0), got {v.major}.{v.minor}"
EOF
echo "  python: $PY_BIN ($("$PY_BIN" --version))"

echo "== [1/5] venv + pinned deps (requirements-h100.txt) =="
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PY_BIN" -m venv "$VENV_DIR" || {
    echo "  venv creation failed — trying apt python3-venv (needs root)" >&2
    apt-get install -y "$(basename "$PY_BIN")-venv" python3-pip
    "$PY_BIN" -m venv "$VENV_DIR"
  }
fi
# vLLM owns its dependency tree (humming-kernels[cu13], flashinfer, ...):
# install it FIRST, alone, then the project tooling — pinning vLLM's
# transitive deps alongside it makes pip fail (ResolutionImpossible; hit
# 2026-09-27 on the rental). requirements-h100.txt documents this too.
if ! "$VENV_DIR/bin/python" -c "import vllm" 2>/dev/null; then
  "$VENV_DIR/bin/pip" install -q "vllm==0.30.0" "torch==2.13.0"
fi
"$VENV_DIR/bin/pip" install -q -r requirements-h100.txt
"$VENV_DIR/bin/python" -c "import vllm; print('  vllm', vllm.__version__)"
"$VENV_DIR/bin/python" -c "import torch; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "== [2/5] raw datasets (xLAM + ToolBench mirror) =="
if [ ! -f data/raw/download_report.json ]; then
  "$VENV_DIR/bin/python" src/data_prep/download_datasets.py
else
  echo "  already downloaded — step 3's checksum verify catches corruption"
fi

echo "== [3/5] frozen-split verify (GPU-day step 0, plan §6.3) =="
"$VENV_DIR/bin/python" -m src.data_prep.freeze_splits verify

echo "== [4/5] model downloads =="
"$VENV_DIR/bin/python" - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo in (
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
):
    p = snapshot_download(repo)
    n = sum(1 for f in os.scandir(p) if f.is_file())
    print(f"  {repo} -> {p} ({n} files)")
EOF

echo "== [4b/5] draft embedding padding (vLLM requires equal vocab_size) =="
# vLLM 0.30.0 SpeculativeConfig hard-rejects draft/target pairs whose
# config.vocab_size differ (152,064 target vs 151,936 draft — caught by
# this script's own smoke test on the rental, 2026-09-27). Pad the
# draft's embedding to the target's vocab with zero rows; prepare_draft
# gates on greedy parity before saving (src/serving/prepare_draft.py).
for D in 0.5B; do
  SRC="Qwen/Qwen2.5-Coder-${D}-Instruct"
  OUT="drafts/coder-${D,,}-padded"
  "$VENV_DIR/bin/python" -m src.serving.prepare_draft \
      --draft "$SRC" --out "$OUT" --dtype bfloat16
done

echo "== [5/5] vLLM speculative_config smoke test (plan §1.5) =="
# 5 frozen prompts through each spec method, one engine per subprocess —
# the pinned build must accept both §1.5 config shapes before any number
# from this harness is trusted.
for M in draft_model ngram; do
  "$VENV_DIR/bin/python" - "$M" <<'EOF'
import sys
from src.serving.spec_configs import build_spec_config
import pyarrow.parquet as pq

method = sys.argv[1]
prompts = pq.read_table("frozen/xlam_eval.parquet").column("input_ids")[:5].to_pylist()
cfg = (
    build_spec_config("draft_model", "drafts/coder-0.5b-padded", 5)
    if method == "draft_model"
    else build_spec_config("ngram", num_speculative_tokens=5)
)
import vllm
from vllm import SamplingParams

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
print(f"  {method}: {len(outs)} completions, "
      f"{sum(len(o.outputs[0].token_ids) for o in outs)} tokens — OK")
EOF
done
echo "smoke test passed — both speculative_config shapes work on this build"

echo
echo "setup_day0 complete. Next: bash scripts/run_baselines.sh"
