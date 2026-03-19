#!/usr/bin/env python3
"""
Google Colab (CPU) runner for Stage 8 — Noise Robustness.

Colab notes:
- Full Stage 8 (12 noise conditions × 4 models trained) can take a very long time on CPU.
- This script supports running a smaller subset first (recommended).

Expected repo layout (after cloning):
  /content/audio_fraud_detection/...

Expected data layout inside the repo:
  data/processed/real|fake/*.wav
  data/features/acoustic_features.csv

Usage (in Colab):
  !python3 scripts/colab_run_stage8_cpu.py \
      --drive_root "/content/drive/MyDrive/audio_fraud_detection" \
      --sync_from_drive \
      --noise_types white,pink \
      --snrs 20,10 \
      --models hubert_base_ls960,wavlm
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def copytree_merge(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = item
        d = dst / item.name
        if s.is_dir():
            copytree_merge(s, d)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive_root",
        type=Path,
        default=None,
        help="Drive folder that contains data/ and optionally results/. Example: /content/drive/MyDrive/audio_fraud_detection",
    )
    parser.add_argument(
        "--sync_from_drive",
        action="store_true",
        help="Copy data/processed and data/features + optional results/*.json from Drive into this repo.",
    )
    parser.add_argument(
        "--noise_types",
        type=str,
        default="white,pink,compression",
        help="Comma-separated noise types to run (subset). Default: all",
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default="20,10,5,0",
        help="Comma-separated SNRs to run (subset). Default: 20,10,5,0",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="hubert_base_ls960,wavlm",
        help="Comma-separated base model short names to run: hubert_base_ls960,wavlm. Default: both.",
    )
    parser.add_argument(
        "--skip_install",
        action="store_true",
        help="Skip pip install -r requirements.txt",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplcache"))  # avoid unwritable home on Colab

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if args.sync_from_drive:
        if args.drive_root is None:
            print("--drive_root is required with --sync_from_drive")
            return 2
        drive_root = args.drive_root
        # Drive folder should include: data/processed, data/features, and optionally results/*.json
        print(f"Syncing from Drive: {drive_root}")
        copytree_merge(drive_root / "data" / "processed", PROJECT_ROOT / "data" / "processed")
        copytree_merge(drive_root / "data" / "features", PROJECT_ROOT / "data" / "features")
        # results jsons are small and helpful (split + baselines)
        if (drive_root / "results").exists():
            (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
            for p in (drive_root / "results").glob("*.json"):
                shutil.copy2(p, PROJECT_ROOT / "results" / p.name)

    # Sanity check
    if not (PROJECT_ROOT / "results" / "split.json").exists():
        print("Missing results/split.json. You need the exact Stage 5/6 split for Stage 8.")
        print("Copy it from your local machine into Drive results/ or generate it via train_cnn.py.")
        return 1
    if not (PROJECT_ROOT / "data" / "processed").exists():
        print("Missing data/processed. Put processed audio in Drive and use --sync_from_drive.")
        return 1

    noise_types = [x.strip() for x in args.noise_types.split(",") if x.strip()]
    snrs = [int(x.strip()) for x in args.snrs.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    # We control subsets by editing config at runtime (write a temporary JSON override)
    override = {
        "NOISE_TYPES": noise_types,
        "SNR_LEVELS_NOISY": snrs,
        "MODELS": models,
    }
    override_path = PROJECT_ROOT / "results" / "stage8_colab_override.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(json.dumps(override, indent=2))
    print(f"Wrote override: {override_path}")

    print("\nRunning Stage 8. This can be slow on CPU; start with a small subset.")
    run([sys.executable, "scripts/run_noise_robustness.py"])

    print("\nDone. Outputs:")
    print(f"  - {PROJECT_ROOT / 'results' / 'noise_robustness.json'}")
    print(f"  - {PROJECT_ROOT / 'reports' / 'noise_robustness_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

