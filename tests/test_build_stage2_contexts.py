"""Golden tests for the Stage-2 context pool builder (plan.md §2 Stage 2,
§6.9). The pools render fresh from COMMITTED frozen indices + raw data —
the GPU host never needs gitignored data/processed index files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_prep.build_stage2_contexts import xlam_stage2_idx

REPO = Path(__file__).resolve().parents[1]
RAW_XLAM = REPO / "data" / "raw" / "xlam"
RAW_TB = REPO / "data" / "raw" / "toolbench_default"


class TestXlamStage2Idx:
    """P1's pool-minus-carve math from the committed frozen files."""

    @pytest.mark.skipif(not RAW_XLAM.exists(), reason="raw xLAM not downloaded")
    def test_pool_minus_carve_sorted(self):
        pool = json.loads((REPO / "frozen" / "xlam_train_pool_idx.json").read_text())
        s1 = set(json.loads((REPO / "frozen" / "stage1_idx.json").read_text()))
        idx = xlam_stage2_idx()
        assert len(idx) == 52_794  # 57,794 - 5,000, measured
        assert idx == sorted(set(pool) - s1)  # sorted: pure function

    def test_desync_refused(self, monkeypatch, tmp_path):
        # stage1 indices outside the pool => frozen files desynced: hard
        # error, never a silent gap
        fake = tmp_path
        (fake / "xlam_train_pool_idx.json").write_text(json.dumps([1, 2]))
        (fake / "stage1_idx.json").write_text(json.dumps([4]))
        import src.data_prep.build_stage2_contexts as m

        monkeypatch.setattr(m, "FROZEN", fake)
        with pytest.raises(AssertionError, match="desynced"):
            xlam_stage2_idx()

    @pytest.mark.skipif(not RAW_XLAM.exists(), reason="raw xLAM not downloaded")
    def test_limit_takes_sorted_head(self):
        assert xlam_stage2_idx(limit=5) == xlam_stage2_idx()[:5]


@pytest.mark.skipif(not RAW_XLAM.exists(), reason="raw xLAM not downloaded")
class TestBuildXlam:
    def test_records_carry_boundary(self, tmp_path, monkeypatch):
        """A built xLAM record carries the full conversation with
        n_prompt_tokens as the sampling boundary (G5's contract with the
        trainer's _prefixes)."""
        from src.data_prep import build_stage2_contexts as m
        from src.data_prep.xlam_prep import OUT_DIR

        monkeypatch.setattr(
            "src.data_prep.xlam_prep.OUT_DIR", tmp_path, raising=True
        )
        stats = m.build_xlam_stage2(limit=3)
        assert stats["n"] == 3
        from datasets import load_from_disk

        ds = load_from_disk(tmp_path / "stage2_contexts")
        for r in ds:
            assert 0 < r["n_prompt_tokens"] < len(r["input_ids"])
            # the boundary is a prompt prefix: the gold assistant turn
            # lives strictly after it
            assert any(l != -100 for l in r["labels"])


@pytest.mark.skipif(not RAW_TB.exists(), reason="raw ToolBench not downloaded")
class TestBuildTbPrefixes:
    def test_records_carry_labels_and_full_conv(self, tmp_path, monkeypatch):
        """TB prefix records carry labels so the trainer can enumerate
        EVERY assistant-turn boundary (post-observation states)."""
        from src.data_prep import build_stage2_contexts as m

        monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
        monkeypatch.chdir(REPO)  # relative imports stay repo-rooted
        stats = m.build_tb_prefixes(limit=3)
        assert stats["n"] == 3
        from datasets import load_from_disk

        ds = load_from_disk(
            tmp_path / "data" / "processed" / "toolbench" / "prefixes"
        )
        for r in ds:
            assert any(l != -100 for l in r["labels"])
            # multi-turn: >= 2 assistant spans in at least one record
            # (median 5 raw turns in the pool)
            starts = sum(
                1 for i in range(1, len(r["labels"]))
                if r["labels"][i] != -100 and r["labels"][i - 1] == -100
            )
            assert starts >= 1
