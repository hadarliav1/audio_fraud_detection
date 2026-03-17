#!/usr/bin/env python3
"""
Run the full mission pipeline in the correct order.

Prerequisites: data/processed/ and data/features/acoustic_features.csv exist
  (run scripts/run_preprocessing.py and scripts/extract_features.py if needed).

Usage:
  python scripts/run_pipeline.py              # run all steps
  python scripts/run_pipeline.py --clean       # clean results/ and reports/ first, then run all
  python scripts/run_pipeline.py --timing      # run all steps and log elapsed time for each
  python scripts/run_pipeline.py --from 3 --timing   # run from step 3 (fusion) onward, with timing
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "acoustic_features.csv"
TIMINGS_LOG = PROJECT_ROOT / "results" / "pipeline_timings.log"

# Pipeline order: Baseline (CNN only) → Transformers → Fusion → Noise Robustness → Report
# train_baseline.py (RF/LR on acoustic) is optional; main baseline is CNN.
SCRIPTS = [
    "train_cnn.py",
    "train_transformers.py",
    "experiment_fusion_ab.py",
    "run_noise_robustness.py",
    "generate_report.py",
]


def run_script(name: str) -> bool:
    """Run a script from the project root. Return True on success."""
    script_path = PROJECT_ROOT / "scripts" / name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    code = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    return code.returncode == 0


def log_timing(step_name: str, elapsed_sec: float, timings_log: Path) -> None:
    """Append one timing line to the log file."""
    timings_log.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {step_name}: {elapsed_sec:.1f}s ({elapsed_sec / 60:.1f} min)\n"
    with open(timings_log, "a") as f:
        f.write(line)
    print(f"  Time: {elapsed_sec:.1f}s ({elapsed_sec / 60:.1f} min)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full mission pipeline in order.")
    parser.add_argument(
        "--clean",
        "--clean-all",
        dest="clean",
        action="store_true",
        help="Run clean_outputs.py --all before the pipeline.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Log elapsed time for each step to results/pipeline_timings.log",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        type=int,
        default=1,
        metavar="N",
        help="Start from step N (1=train_cnn, 2=train_transformers, 3=fusion, 4=noise, 5=report). Default 1.",
    )
    args = parser.parse_args()

    if args.clean:
        print("Cleaning results/ and reports/ ...")
        code = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "clean_outputs.py"), "--all"],
            cwd=str(PROJECT_ROOT),
        )
        if code.returncode != 0:
            return code.returncode
        print()

    start_idx = max(0, args.from_step - 1)
    if start_idx > 0:
        print(f"Starting from step {args.from_step}: {SCRIPTS[start_idx]}")
    scripts_to_run = list(enumerate(SCRIPTS[start_idx:], start=start_idx + 1))

    # Prerequisite check when starting from the beginning
    if start_idx == 0:
        if not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.iterdir()):
            print("Error: data/processed/ is missing or empty. Run first:")
            print("  python scripts/run_preprocessing.py")
            return 1
        if not FEATURES_CSV.exists():
            print("Error: data/features/acoustic_features.csv not found. Run first:")
            print("  python scripts/extract_features.py")
            return 1

    for i, name in scripts_to_run:
        print(f"[{i}/{len(SCRIPTS)}] Running {name} ...")
        t0 = time.perf_counter()
        if not run_script(name):
            print(f"Failed at {name}. Stop.")
            return 1
        elapsed = time.perf_counter() - t0
        if args.timing:
            log_timing(name, elapsed, TIMINGS_LOG)
        print()
    print("Pipeline finished.")
    if args.timing:
        print(f"  Timings: {TIMINGS_LOG}")
    print("  Results: results/ (checkpoints + JSON)")
    print("  Report: reports/final_report.md (from generate_report.py)")
    print("  View fusion and noise: notebooks/07_fusion.ipynb, 08_noise_robustness.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
