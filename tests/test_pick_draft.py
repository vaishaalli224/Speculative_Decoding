"""Golden tests for the §8.1 draft picker (plan §6: GPU-day decisions must
be mechanical). The picker reads results/metrics.jsonl records in the exact
shape bench_vllm.py's bench_and_record writes (B7), so these fixtures ARE
that shape — they double as a pin on the metrics-record schema the picker
depends on: spec_method, method_detail, tag, runs, _k (from spec_config),
median_gen_tok_s."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

DRAFT05 = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DRAFT15 = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def bench(spec_method, method_detail, tok_s, *, tag=None, batch=1, runs=3,
          k=None, limit=200):
    """One metrics.jsonl bench record, in bench_vllm.py's exact shape."""
    rec = {
        "kind": "bench",
        "tag": tag or spec_method,
        "spec_method": spec_method,
        "method_detail": method_detail,
        "median_gen_tok_s": tok_s,
        "batch": batch,
        "runs": runs,
        "temperature": 0.0,
        "per_turn": False,
        "limit": limit,
        "spec_config": (
            {"method": "draft_model", "model": method_detail,
             "num_speculative_tokens": k, "max_model_len": 16384}
            if spec_method == "draft_model" else None
        ),
    }
    return rec


@pytest.fixture
def run_picker(tmp_path):
    def _run(records, expect_fail=False):
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text("".join(json.dumps(r) + "\n" for r in records))
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "pick_draft.py"),
             "--metrics", str(metrics)],
            capture_output=True, text=True, cwd=tmp_path,
        )
        if expect_fail:
            assert proc.returncode != 0, proc.stdout + proc.stderr
            return proc
        assert proc.returncode == 0, proc.stderr
        # the decision JSON is followed by a human summary line — read the
        # saved decision file instead of parsing stdout
        return json.loads(
            (tmp_path / "draft_decision.json").read_text()
        )
    return _run


class TestPickDraft:
    def test_15b_wins_on_speed(self, run_picker):
        out = run_picker([
            bench("ar", "ar_only", 100.0),
            bench("draft_model", DRAFT05, 150.0, k=5),
            bench("draft_model", DRAFT15, 190.0, k=5),
        ])
        assert out["winner"] == "1.5B"
        assert out["candidates"]["1.5B"]["speedup"] == pytest.approx(1.9)
        assert out["candidates"]["0.5B"]["speedup"] == pytest.approx(1.5)

    def test_05b_wins_on_speed(self, run_picker):
        out = run_picker([
            bench("ar", "ar_only", 100.0),
            bench("draft_model", DRAFT05, 180.0, k=5),
            bench("draft_model", DRAFT15, 160.0, k=5),
        ])
        assert out["winner"] == "0.5B"

    def test_exact_tie_breaks_to_05b(self, run_picker):
        out = run_picker([
            bench("ar", "ar_only", 100.0),
            bench("draft_model", DRAFT05, 150.0, k=5),
            bench("draft_model", DRAFT15, 150.0, k=5),
        ])
        assert out["winner"] == "0.5B"
        assert "tie-break" in out["reason"]

    def test_exactness_records_are_ignored(self, run_picker):
        # the AR side of an --exactness run: 50 prompts, runs=1, tag
        # *_exactness — must NOT be picked as the rule's AR baseline
        out = run_picker([
            bench("ar", "ar_only", 100.0),
            bench("ar", "ar_only", 999.0, tag="ar_exactness", runs=1, limit=50),
            bench("draft_model", DRAFT05, 150.0, k=5),
            bench("draft_model", DRAFT15, 140.0, k=5),
        ])
        # with 999 as AR the 1.5B (140 < 150 relative) would still win —
        # assert the AR number used is the 3-run baseline's
        assert out["ar_gen_tok_s"] == 100.0
        assert out["winner"] == "0.5B"

    def test_most_recent_run_wins_on_rerun(self, run_picker):
        # a re-run config appends; the picker takes the latest record
        out = run_picker([
            bench("ar", "ar_only", 100.0),
            bench("draft_model", DRAFT05, 150.0, k=5),
            bench("draft_model", DRAFT15, 190.0, k=5),
            bench("draft_model", DRAFT05, 175.0, k=5),  # re-run
        ])
        assert out["candidates"]["0.5B"]["gen_tok_s"] == 175.0
        assert out["winner"] == "1.5B"

    def test_wrong_k_ignored(self, run_picker):
        # a k=7 sweep record must not satisfy the rule's k=5 requirement
        proc = run_picker([
            bench("ar", "ar_only", 100.0),
            bench("draft_model", DRAFT05, 150.0, k=5),
            bench("draft_model", DRAFT15, 190.0, k=7),
        ], expect_fail=True)
        assert "1.5B" in proc.stderr and "run_baselines.sh" in proc.stderr

    def test_missing_ar_fails_loudly(self, run_picker):
        proc = run_picker([
            bench("draft_model", DRAFT05, 150.0, k=5),
            bench("draft_model", DRAFT15, 190.0, k=5),
        ], expect_fail=True)
        assert "AR" in proc.stderr

    def test_empty_metrics_fails_loudly(self, run_picker, tmp_path):
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text("")
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "pick_draft.py"),
             "--metrics", str(metrics)],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert proc.returncode != 0
