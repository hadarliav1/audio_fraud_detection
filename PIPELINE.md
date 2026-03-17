# Voice Fraud Detection — Pipeline

> **Canonical order:** Dataset → EDA (raw + acoustic) → Baseline (CNN only) → Transformers → Fusion → Noise Robustness → Summary

## Execution Order

```
1. download_dataset.py              → data/raw/
2. notebooks/01_research_question.ipynb
3. notebooks/02_dataset_understanding.ipynb   (raw data overview)
4. notebooks/03_eda_raw_data.ipynb           (EDA on raw audio)
5. scripts/run_preprocessing.py     → data/processed/
6. scripts/extract_features.py     → data/features/acoustic_features.csv
7. notebooks/04_eda_acoustic_features.ipynb (EDA on acoustic features)
8. scripts/train_cnn.py             → Baseline: CNN on mel-spectrogram only
9. scripts/train_transformers.py    → HuBERT, Wav2Vec2, WavLM, Whisper
10. scripts/experiment_fusion_ab.py → Fusion (fine-tuned embeddings + acoustic)
11. scripts/run_noise_robustness.py → Noise experiments
12. scripts/generate_report.py      → reports/final_report.md
13. notebooks/09_conclusions.ipynb  → Final summary
```

**Note:** `notebooks/optional_classical_baseline.ipynb` and `scripts/train_baseline.py` (RF/LR on acoustic features) are **optional** for ablation; the official baseline is the CNN (step 8).

## Commands

```bash
# 1. Download (requires Kaggle API)
python download_dataset.py

# 2. Run notebooks 01, 02, 03 (EDA raw) on raw data
# 3. Preprocess: resample, trim, normalize, clip/pad
python scripts/run_preprocessing.py

# 4. Extract acoustic features from processed audio
python scripts/extract_features.py

# 5. Run 03+ notebooks (EDA on features, then models)

# 6. Train models (speaker-disjoint split for all)
python scripts/train_cnn.py           # Baseline: CNN on mel-spectrogram
python scripts/train_transformers.py  # HuBERT, Wav2Vec2, WavLM, Whisper
python scripts/experiment_fusion_ab.py  # Fusion: fine-tuned embeddings + acoustic
python scripts/run_noise_robustness.py

# Optional: classical baseline (RF/LR on acoustic only)
python scripts/train_baseline.py

# Or run full pipeline (after preprocessing + features)
python scripts/run_pipeline.py

# 7. Generate final report
python scripts/generate_report.py

# 8. Inference on a single file
python scripts/predict.py path/to/audio.wav
python scripts/predict.py --model wav2vec2 path/to/audio.wav
```

## EDA Checklist

**Raw audio (03_eda_raw_data):**
- [x] Class distribution
- [x] Speaker / generator distribution
- [x] Duration distribution
- [x] Waveform or spectrogram examples

**Acoustic features (04_eda_acoustic_features):**
- [x] Feature distributions (real vs fake)
- [x] Correlation heatmap
- [x] Feature importance / separability (AUC per feature, PCA)

## Model Approaches

| Script | Input | Classifier | Output |
|--------|--------|------------|--------|
| `train_cnn.py` | Mel-spectrogram | CNN (**baseline**) | results/cnn_spectrogram.pt, cnn_results.json |
| `train_transformers.py` | Raw audio | Fine-tuned HF (encoder + head) | results/transformer_*/ |
| `experiment_fusion_ab.py` | Fine-tuned 768-d emb + acoustic | LR | results/fusion_ab_results.json |
| `run_noise_robustness.py` | Corrupted audio | All above | results/noise_robustness_results.json |
| `train_baseline.py` (optional) | Acoustic features | RF, LR | results/baseline_*.joblib, baseline_results.json |

## Pipeline Notes

- **Speaker-disjoint split** — All models use the same speaker-disjoint train/val/test split for fair comparison.
- **Report** — `generate_report.py` writes `reports/final_report.md` and `results/metrics_summary.csv` from JSON in `results/`.
- **Results** — Checkpoints and metrics live under `results/` (see PROJECT_STRUCTURE.md). If you had an older `outputs/` folder, copy its contents to `results/` or re-run the training scripts.
