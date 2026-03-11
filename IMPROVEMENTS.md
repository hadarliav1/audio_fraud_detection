# Improvements — What Changed & What to Rerun

## Summary of Changes

| Change | Why |
|--------|-----|
| **Speaker-disjoint for baseline** | RF/LR used random split; others used speaker-disjoint. Now all models share the same evaluation, so comparisons are fair. |
| **Centralized splits** | `get_speaker()` and `speaker_disjoint_split()` were duplicated in 4 scripts. Moved to `src/utils/splits.py`. |
| **Bootstrap confidence intervals** | Metrics were point estimates. Now AUC (and F1) include 95% CI for baseline and fusion A/B. |
| **Inference script** | No single-file prediction existed. Added `predict.py` for deployment/demos. |
| **Auto-generated report** | `final_report.md` was a manual template. Now populated from JSON outputs. |

---

## What You Must Rerun

**Baseline only** — it switched to speaker-disjoint split, so its metrics changed:

```bash
python scripts/train_baseline.py
```

**Optional** — regenerate the report after any training:

```bash
python scripts/generate_report.py
```

**No rerun needed:** CNN, transformers, fusion A/B, noise robustness — they already used speaker-disjoint.

---

## New Commands

```bash
# Inference on one audio file (default: wav2vec2)
python scripts/predict.py path/to/audio.wav
python scripts/predict.py --model rf path/to/audio.wav
python scripts/predict.py --model cnn path/to/audio.wav

# Regenerate final report from JSON outputs
python scripts/generate_report.py
```

---

## Files Touched

| File | Change |
|------|--------|
| `src/utils/splits.py` | **New** — shared split logic |
| `src/utils/eval.py` | Added `evaluate_binary_bootstrap()` |
| `scripts/train_baseline.py` | Speaker-disjoint split + bootstrap CI |
| `scripts/train_transformers.py` | Uses `src.utils.splits` |
| `scripts/train_cnn.py` | Uses `src.utils.splits` |
| `scripts/experiment_fusion_ab.py` | Uses `src.utils.splits` + bootstrap CI |
| `scripts/run_noise_robustness.py` | Uses `src.utils.splits` |
| `scripts/predict.py` | **New** — inference |
| `scripts/generate_report.py` | **New** — report generation |
