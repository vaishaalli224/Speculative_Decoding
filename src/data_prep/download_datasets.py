"""Download raw datasets to data/raw/ and emit a stats report.

Stage 0 of the data pipeline (plan.md §3, §6.2, §6.3):
  - ToolBench mirror `tuandunghcmut/toolbench-v1` (train/validation conversations
    + benchmark G1/G2/G3 instruction splits) -> data/raw/toolbench_{default,benchmark}/
  - xLAM `Salesforce/xlam-function-calling-60k` (eval-only) -> data/raw/xlam/

HF credentials come from HF_TOKEN (the workspace .env is loaded via python-dotenv).

Run:  .venv/bin/python src/data_prep/download_datasets.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load the workspace .env (repo root's parent) so HF_TOKEN is available.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT.parent / ".env")

from datasets import DatasetDict, load_dataset  # noqa: E402  (needs env loaded first)

TOOLBENCH_REPO = "tuandunghcmut/toolbench-v1"
XLAM_REPO = "Salesforce/xlam-function-calling-60k"
DATA_DIR = _REPO_ROOT / "data" / "raw"


def download_toolbench() -> dict:
    """Download both ToolBench configs and save them under data/raw/."""
    stats = {}
    for config in ("default", "benchmark"):
        ds: DatasetDict = load_dataset(TOOLBENCH_REPO, config)
        out = DATA_DIR / f"toolbench_{config}"
        ds.save_to_disk(out)
        stats[config] = {
            "splits": {name: len(split) for name, split in ds.items()},
            "columns": {name: split.column_names for name, split in ds.items()},
            "path": str(out),
        }
    return stats


def download_xlam() -> dict:
    """Download xLAM (eval-only; never trained on)."""
    ds: DatasetDict = load_dataset(XLAM_REPO)
    out = DATA_DIR / "xlam"
    ds.save_to_disk(out)
    return {
        "splits": {name: len(split) for name, split in ds.items()},
        "columns": {name: split.column_names for name, split in ds.items()},
        "path": str(out),
    }


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set — check the workspace .env")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading ToolBench mirror ...")
    toolbench_stats = download_toolbench()
    print("Downloading xLAM ...")
    xlam_stats = download_xlam()

    report = {
        "toolbench": toolbench_stats,
        "xlam": xlam_stats,
    }
    report_path = DATA_DIR / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
