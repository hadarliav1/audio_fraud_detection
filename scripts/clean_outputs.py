#!/usr/bin/env python3
"""
Remove results and report files before re-running the pipeline.

Default: removes only mission result files (JSON in results/, report in reports/).
  python scripts/clean_outputs.py

Full clean (entire results folder + report): use --all.
  python scripts/clean_outputs.py --all
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_DIR, REPORTS_DIR

FILES_TO_REMOVE = [
    "fusion_ab_results.json",
    "transformer_results.json",
    "cnn_results.json",
    "baseline_results.json",
    "noise_robustness_results.json",
]
REPORT_FILES = ["final_report.md"]


def clean_mission_files() -> list[str]:
    """Remove only mission result files. Returns list of removed names."""
    removed = []
    for name in FILES_TO_REMOVE:
        p = OUTPUTS_DIR / name
        if p.exists():
            p.unlink()
            removed.append(name)
    for name in REPORT_FILES:
        p = REPORTS_DIR / name
        if p.exists():
            p.unlink()
            removed.append(str(REPORTS_DIR / name))
    return removed


def clean_all() -> bool:
    """Remove entire contents of results/ (all files and subdirs). Returns True if anything was removed."""
    if not OUTPUTS_DIR.exists():
        return False
    had_anything = False
    for item in OUTPUTS_DIR.iterdir():
        if item.is_file():
            item.unlink()
            had_anything = True
        else:
            shutil.rmtree(item)
            had_anything = True
    return had_anything


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean outputs before re-running the pipeline.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove entire results/ folder and report files. Default: only mission result files.",
    )
    args = parser.parse_args()

    if args.all:
        if not OUTPUTS_DIR.exists():
            print(f"{OUTPUTS_DIR} does not exist. Nothing to clean.")
        elif clean_all():
            print("Cleaned entire results/ folder.")
        else:
            print("results/ was already empty.")
        if REPORTS_DIR.exists():
            for f in REPORT_FILES:
                p = REPORTS_DIR / f
                if p.exists():
                    p.unlink()
                    print(f"Removed {p}")
        return 0

    if not OUTPUTS_DIR.exists():
        print(f"{OUTPUTS_DIR} does not exist. Nothing to clean.")
        return 0
    removed = clean_mission_files()
    if removed:
        print("Removed:", ", ".join(removed))
    else:
        print("No mission result files found to remove.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
