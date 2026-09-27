"""Split-freezing (plan.md §6.3): export the frozen eval parquets + Stage 1/2
context index files + sha256 manifest to frozen/ (committed — the GPU day
must never re-derive splits).

Frozen contents:
  frozen/xlam_eval.parquet        xLAM-500 rendered records (the eval set)
  frozen/tb_eval.parquet          TB-500 rendered records (the transfer eval)
  frozen/xlam_train_pool_idx.json raw xLAM indices of the training pool
                                  (57,794; Stage 2 xLAM contexts + the
                                  Stage 1 source pool)
  frozen/stage1_idx.json          raw xLAM indices for Stage 1 KD warm-start
                                  (5,000, seeded sample from the train pool;
                                  Stage 2 xLAM contexts = pool - stage1)
  frozen/tb_prefix_idx.json       raw ToolBench indices of the Stage 2 prefix
                                  pool (45,023; tools disjoint from TB-500)
  frozen/xlam_heldout_functions.json / tb_heldout_tools.json
  frozen/manifest.json            sha256 per file + row/token stats + code
                                  commit + the disjointness assertions

`verify` re-checks every checksum and the held-out properties from the
frozen files alone — run it as the first step of the GPU day (runbook §5,
hour 0). It re-derives nothing.

CLI:
  freeze_splits.py freeze   # build frozen/ from data/processed
  freeze_splits.py verify   # checksum + disjointness check (GPU-day step 0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from src.data_prep.render import REGION_TAG, REGION_TOOL_CALL
from src.data_prep.toolbench_clean import OUT_DIR as TB_OUT_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = REPO_ROOT / "frozen"
XLAM_OUT = REPO_ROOT / "data" / "processed" / "xlam"
STAGE1_SIZE = 5000
STAGE1_SEED = 42
MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def _dataset_stats(d) -> dict:
    lens = np.array([len(x) for x in d["input_ids"]])
    tool_region = np.array(
        [sum(n for c, n in x if c in (REGION_TOOL_CALL, REGION_TAG)) for x in d["regions"]]
    )
    return {
        "n": len(d),
        "tokens_total": int(lens.sum()),
        "n_tokens_median": float(np.median(lens)),
        "n_tokens_p95": float(np.percentile(lens, 95)),
        "n_tokens_max": int(lens.max()),
    }


def run_freeze() -> dict:
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    # -- eval parquets ------------------------------------------------
    files: dict[str, Path] = {}
    for name, src in (("xlam_eval", XLAM_OUT / "eval"), ("tb_eval", TB_OUT_DIR / "eval")):
        d = load_from_disk(src)
        out = FROZEN_DIR / f"{name}.parquet"
        d.to_parquet(out)
        files[name] = out

    # -- context index files -----------------------------------------
    train_pool = json.loads((XLAM_OUT / "train_idx.json").read_text())
    rng = np.random.default_rng(STAGE1_SEED)
    stage1 = sorted(rng.choice(len(train_pool), size=STAGE1_SIZE, replace=False))
    stage1_idx = [train_pool[i] for i in stage1]

    clean_index = json.loads((TB_OUT_DIR / "clean_index.json").read_text())
    prefix_idx = json.loads((TB_OUT_DIR / "prefix_idx.json").read_text())
    tb_prefix_raw = [clean_index[i]["raw_idx"] for i in prefix_idx]

    index_files = {
        "xlam_train_pool_idx.json": train_pool,
        "xlam_eval_idx.json": json.loads((XLAM_OUT / "eval_idx.json").read_text()),
        "stage1_idx.json": stage1_idx,
        "tb_prefix_idx.json": tb_prefix_raw,
        "tb_eval_idx.json": json.loads((TB_OUT_DIR / "eval_idx.json").read_text()),
        "xlam_heldout_functions.json": json.loads(
            (XLAM_OUT / "heldout_functions.json").read_text()
        ),
        "tb_heldout_tools.json": json.loads(
            (TB_OUT_DIR / "heldout_tools.json").read_text()
        ),
    }
    for fname, payload in index_files.items():
        p = FROZEN_DIR / fname
        p.write_text(json.dumps(payload))
        files[fname.removesuffix(".json")] = p

    # -- raw-data provenance: checksum the arrow files the indices point into,
    # so the GPU day can assert it re-renders from identical bytes
    raw_manifest: dict[str, dict[str, str]] = {}
    for name, raw_dir in (
        ("xlam", REPO_ROOT / "data" / "raw" / "xlam"),
        ("toolbench_default", REPO_ROOT / "data" / "raw" / "toolbench_default"),
    ):
        raw_manifest[name] = {
            str(f.relative_to(REPO_ROOT)): _sha256(f)
            for f in sorted(raw_dir.rglob("*.arrow"))
        }

    # -- manifest ------------------------------------------------------
    manifest = {
        "schema_version": 2,
        "frozen_date": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
        ).stdout.strip(),
        "code_commit": _git_commit(),
        "stage1": {
            "size": STAGE1_SIZE,
            "seed": STAGE1_SEED,
            "note": "seeded sample of xlam_train_pool_idx; Stage 2 xLAM "
            "contexts = train_pool - stage1",
        },
        "stage2_mix_default": "xLAM:TB 1:1 by conversation (plan.md §6.8)",
        "files": {
            key: {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for key, p in files.items()
        },
        "raw_data_sha256": raw_manifest,
        "dataset_stats": {
            "xlam_eval": _dataset_stats(load_from_disk(XLAM_OUT / "eval")),
            "tb_eval": _dataset_stats(load_from_disk(TB_OUT_DIR / "eval")),
        },
    }
    mpath = FROZEN_DIR / MANIFEST_NAME
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # -- self-check immediately after freezing --------------------------
    report = run_verify()
    if not report["ok"]:
        raise SystemExit(f"freeze failed self-verification:\n{json.dumps(report, indent=2)}")
    return {"manifest": manifest, "verify": report}


def run_verify() -> dict:
    """Checksum + disjointness check from frozen/ alone. Re-derives nothing."""
    problems: list[str] = []
    mpath = FROZEN_DIR / MANIFEST_NAME
    if not mpath.exists():
        return {"ok": False, "problems": [f"missing {MANIFEST_NAME}"]}
    manifest = json.loads(mpath.read_text())

    for key, meta in manifest["files"].items():
        fname = f"{key}.parquet" if key.endswith("_eval") else f"{key}.json"
        p = FROZEN_DIR / fname
        if not p.exists():
            problems.append(f"missing file: {fname}")
            continue
        if _sha256(p) != meta["sha256"]:
            problems.append(f"checksum mismatch: {fname}")
        if p.stat().st_size != meta["bytes"]:
            problems.append(f"size mismatch: {fname}")

    # held-out properties, from the frozen files alone. Files already
    # reported missing above are skipped here (reading them would crash).
    def _load_json(name: str) -> set | None:
        p = FROZEN_DIR / name
        if not p.exists():
            return None  # already flagged by the checksum loop
        return set(json.loads(p.read_text()))

    pool = _load_json("xlam_train_pool_idx.json")
    stage1 = _load_json("stage1_idx.json")
    xlam_eval_idx = _load_json("xlam_eval_idx.json")
    heldout_fns = _load_json("xlam_heldout_functions.json")
    if pool is not None and stage1 is not None:
        if stage1 & xlam_eval_idx:
            problems.append("stage1 overlaps xLAM eval")
        if not stage1 <= pool:
            problems.append("stage1 not a subset of the xLAM train pool")
        if len(stage1) != STAGE1_SIZE:
            problems.append(f"stage1 has {len(stage1)} entries, expected {STAGE1_SIZE}")
    if pool is not None and xlam_eval_idx is not None and pool & xlam_eval_idx:
        problems.append("xLAM train pool overlaps eval")
    if heldout_fns is not None and not heldout_fns:
        problems.append("empty xLAM held-out function list")

    heldout_tools = _load_json("tb_heldout_tools.json")
    if heldout_tools is not None and not heldout_tools:
        problems.append("empty TB held-out tool list")

    # eval parquet row counts match the manifest
    from datasets import load_dataset  # local import: parquet reading

    for key in ("xlam_eval", "tb_eval"):
        d = load_dataset("parquet", data_files=str(FROZEN_DIR / f"{key}.parquet"))["train"]
        if len(d) != manifest["dataset_stats"][key]["n"]:
            problems.append(f"{key}.parquet row count != manifest")

    # raw-data provenance (local-only check: raw data is gitignored, so on
    # the GPU day this only runs after data download, before any rendering)
    raw = manifest.get("raw_data_sha256", {})
    for name, per_file in raw.items():
        raw_dir = REPO_ROOT / "data" / "raw" / name
        if not raw_dir.exists():
            problems.append(f"raw dir missing: data/raw/{name} (re-download first)")
            continue
        for rel, digest in per_file.items():
            f = REPO_ROOT / rel
            if not f.exists():
                problems.append(f"raw arrow missing: {rel}")
            elif _sha256(f) != digest:
                problems.append(f"raw data checksum mismatch: {rel}")

    return {
        "ok": not problems,
        "problems": problems,
        "n_files": len(manifest["files"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "verify"])
    args = ap.parse_args()
    if args.cmd == "freeze":
        out = run_freeze()
        print(json.dumps(out["verify"], indent=2))
        print(f"\nmanifest: {FROZEN_DIR / MANIFEST_NAME}")
        print("commit frozen/ to the repo — the GPU day starts from these files")
    else:
        print(json.dumps(run_verify(), indent=2))


if __name__ == "__main__":
    main()
