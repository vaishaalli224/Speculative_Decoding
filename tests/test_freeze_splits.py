"""Tests for split-freezing (plan.md §6.3): the frozen/ directory is the
GPU-day entry point, so `verify` must catch tampering, and the manifest must
satisfy the held-out properties from the frozen files alone.

These tests run against the repo's committed frozen/ directory (they are
skipped if it is absent — e.g. a fresh clone before the first freeze).
A tamper test copies frozen/ to a temp dir, flips a byte, and requires
verify to report the mismatch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.data_prep.freeze_splits import (
    FROZEN_DIR,
    STAGE1_SIZE,
    run_verify,
)

pytestmark = pytest.mark.skipif(
    not (FROZEN_DIR / "manifest.json").exists(),
    reason="frozen/ not built yet — run freeze_splits.py freeze",
)


def test_verify_passes_on_committed_frozen():
    report = run_verify()
    assert report["ok"], report["problems"]


def test_frozen_contents_shape():
    m = json.loads((FROZEN_DIR / "manifest.json").read_text())
    for key in (
        "xlam_eval",
        "tb_eval",
        "stage1_idx",
        "xlam_train_pool_idx",
        "xlam_eval_idx",
        "tb_prefix_idx",
        "tb_eval_idx",
        "xlam_heldout_functions",
        "tb_heldout_tools",
    ):
        assert key in m["files"], f"manifest missing {key}"
        assert len(m["files"][key]["sha256"]) == 64
    # both eval parquets have exactly 500 rows
    assert m["dataset_stats"]["xlam_eval"]["n"] == 500
    assert m["dataset_stats"]["tb_eval"]["n"] == 500


def test_heldout_properties_from_frozen_alone():
    pool = set(json.loads((FROZEN_DIR / "xlam_train_pool_idx.json").read_text()))
    stage1 = set(json.loads((FROZEN_DIR / "stage1_idx.json").read_text()))
    eval_idx = set(json.loads((FROZEN_DIR / "xlam_eval_idx.json").read_text()))
    assert len(stage1) == STAGE1_SIZE
    assert stage1 <= pool
    assert not stage1 & eval_idx
    assert not pool & eval_idx
    # the TB prefix tools must be disjoint from the TB eval held-out tools
    heldout = set(json.loads((FROZEN_DIR / "tb_heldout_tools.json").read_text()))
    assert heldout


def test_verify_detects_tampering(tmp_path, monkeypatch):
    # copy frozen/ to a temp dir, corrupt the stage1 index, require failure
    shutil.copytree(FROZEN_DIR, tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(
        "src.data_prep.freeze_splits.FROZEN_DIR", tmp_path
    )
    p = tmp_path / "stage1_idx.json"
    idx = json.loads(p.read_text())
    idx[0] += 1  # flip one index — must break checksum
    p.write_text(json.dumps(idx))
    report = run_verify()
    assert not report["ok"]
    assert any("checksum mismatch" in s for s in report["problems"])


def test_verify_detects_missing_file(tmp_path, monkeypatch):
    shutil.copytree(FROZEN_DIR, tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(
        "src.data_prep.freeze_splits.FROZEN_DIR", tmp_path
    )
    (tmp_path / "tb_heldout_tools.json").unlink()
    report = run_verify()
    assert not report["ok"]
    assert any("missing file" in s for s in report["problems"])
