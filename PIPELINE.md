# Voice Fraud Detection — Pipeline

> **Recent changes:** See `IMPROVEMENTS.md` for what changed and what to rerun.

## Execution Order

```
1. download_dataset.py     → data/raw/
2. 01_research_question.ipynb
3. 02_dataset_understanding.ipynb   (uses original/raw data)
4. run_preprocessing.py    → data/processed/
5. extract_features.py     → data/features/acoustic_features.csv
6. 03_eda_acoustic_features.ipynb   (uses acoustic_features.csv)
7. 04_classical_baseline.ipynb
8. ...
```

## Commands

```bash
# 1. Download (requires Kaggle API)
python download_dataset.py

# 2. Run 01, 02 on raw data
# 3. Preprocess: resample, trim, normalize, clip/pad
python scripts/run_preprocessing.py

# 4. Extract acoustic features from processed audio
python scripts/extract_features.py

# 5. Run 03+ notebooks (EDA on features, then models)

# 6. Train models (speaker-disjoint split for all)
python scripts/train_baseline.py
python scripts/train_cnn.py
python scripts/train_transformers.py
python scripts/experiment_fusion_ab.py
python scripts/run_noise_robustness.py

# 7. Generate final report
python scripts/generate_report.py

# 8. Inference on a single file
python scripts/predict.py path/to/audio.wav
python scripts/predict.py --model wav2vec2 path/to/audio.wav
```

## Model approaches

| Script | HF encoder | Classifier | AUC (approx) |
|--------|------------|------------|--------------|
| `train_transformers.py` | fine-tuned | built-in head | 0.95 |
| `experiment_fusion_ab.py` | frozen | LR on 768-d or 768+118-d | 0.87 |
| `train_baseline.py` | — | RF, LR on acoustic | 0.63–0.70 |
| `train_cnn.py` | — | CNN on mel-spec | 0.70 |

---

## Pipeline Notes

- **02_dataset_understanding** — runs on **original (raw)** data to characterise the dataset before preprocessing (counts, balance, duration, sample rate variability).
- **03_eda_acoustic_features** — runs on **extracted features** from processed audio (after preprocessing and feature extraction).
- **Speaker-disjoint split** — All models (baseline, CNN, transformers, fusion) use the same speaker-disjoint evaluation for fair comparison.
- **Report** — `generate_report.py` auto-populates `outputs/final_report.md` from JSON results.
