#!/usr/bin/env python3
"""
Fine-tune HuBERT, Wav2Vec2, WavLM, Whisper for binary real/fake classification.

Full model (encoder + head) trained end-to-end. Speaker-disjoint split.
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    LEARNING_RATE,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    TEST_SIZE,
    TRANSFORMER_MODELS,
    VAL_SIZE,
)
from src.utils.eval import evaluate_binary
from src.utils.paths import get_audio_paths_with_labels
from src.utils.splits import load_split, save_split, speaker_disjoint_split, SPLIT_FILENAME


def model_short_name(model_id: str) -> str:
    """e.g. facebook/wav2vec2-base -> wav2vec2, openai/whisper-tiny -> whisper"""
    name = model_id.split("/")[-1].split(".")[0]
    # Strip common suffixes: base, tiny, small, etc.
    for suffix in ["-base", "-tiny", "-small", "_base", "_tiny", "_small"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("-", "_")


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # Remove previous run so re-runs don't mix old and new results (mission reproducibility)
    out_file = OUTPUTS_DIR / "transformer_results.json"
    if out_file.exists():
        out_file.unlink()
    np.random.seed(RANDOM_SEED)

    split_path = OUTPUTS_DIR / SPLIT_FILENAME
    loaded = load_split(split_path) if split_path.exists() else None
    if loaded is not None:
        tr_pairs, val_pairs, test_pairs = loaded
        print("Using same split as CNN (results/split.json)")
    else:
        pairs = get_audio_paths_with_labels(PROCESSED_DIR)
        if not pairs:
            print("No processed audio. Run: python scripts/run_preprocessing.py")
            return 1
        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        tr_pairs, val_pairs, test_pairs = speaker_disjoint_split(
            pairs, TEST_SIZE, val_ratio, RANDOM_SEED
        )
        if not val_pairs:
            val_pairs = tr_pairs[: len(tr_pairs) // 10]
        save_split(tr_pairs, val_pairs, test_pairs, split_path)

    train_paths = [str(p) for p, _ in tr_pairs]
    train_labels = [l for _, l in tr_pairs]
    val_paths = [str(p) for p, _ in val_pairs]
    val_labels = [l for _, l in val_pairs]
    test_paths = [str(p) for p, _ in test_pairs]
    test_labels = [l for _, l in test_pairs]

    print(f"Speaker-disjoint: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # Build datasets with path + label; load audio with librosa to avoid
    # datasets Audio/torchcodec (requires specific FFmpeg versions)
    max_samples = int(5.0 * SAMPLE_RATE)

    def load_audio_batch(paths):
        arrays = []
        for p in paths:
            y, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
            if len(y) > max_samples:
                start = (len(y) - max_samples) // 2
                y = y[start : start + max_samples]
            elif len(y) < max_samples:
                y = np.pad(y, (0, max_samples - len(y)), mode="constant")
            arrays.append(y.astype(np.float32))
        return arrays

    train_ds = Dataset.from_dict({"path": train_paths, "label": train_labels})
    val_ds = Dataset.from_dict({"path": val_paths, "label": val_labels})
    test_ds = Dataset.from_dict({"path": test_paths, "label": test_labels})

    label2id = {"real": 0, "fake": 1}
    id2label = {0: "real", 1: "fake"}
    num_labels = 2

    all_results = {}

    for model_id in TRANSFORMER_MODELS:
        short = model_short_name(model_id)
        out_subdir = OUTPUTS_DIR / f"transformer_{short}"
        results_file = out_subdir / "results.json"

        print(f"\n--- {model_id} ---")
        out_subdir.mkdir(parents=True, exist_ok=True)

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        sr = getattr(feature_extractor, "sampling_rate", SAMPLE_RATE)
        max_duration = 5.0
        max_length = int(max_duration * sr)
        # Whisper expects mel of length 3000 (30s). Its feature extractor truncates
        # raw audio by max_length; 3000 samples -> ~18 mel frames. Use 30s in samples.
        if "whisper" in model_id.lower():
            max_length = 30 * sr  # 480_000 at 16kHz

        def preprocess(examples):
            arrays = load_audio_batch(examples["path"])
            return feature_extractor(
                arrays,
                sampling_rate=getattr(feature_extractor, "sampling_rate", SAMPLE_RATE),
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

        enc_train = train_ds.map(preprocess, remove_columns="path", batched=True, desc="Preprocess train")
        enc_val = val_ds.map(preprocess, remove_columns="path", batched=True, desc="Preprocess val")
        enc_test = test_ds.map(preprocess, remove_columns="path", batched=True, desc="Preprocess test")

        model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )

        training_args = TrainingArguments(
            output_dir=str(out_subdir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            warmup_ratio=0.1,
            logging_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            seed=RANDOM_SEED,
        )

        def compute_metrics(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=1)
            logits = eval_pred.predictions - eval_pred.predictions.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            prob_pos = probs[:, 1]
            m = evaluate_binary(eval_pred.label_ids, preds, prob_pos)
            return {"accuracy": m["accuracy"], "f1": m["f1"], "auc_roc": m["auc_roc"]}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=enc_train,
            eval_dataset=enc_val,
            processing_class=feature_extractor,
            compute_metrics=compute_metrics,
        )

        # If a fine-tuned checkpoint already exists, reuse it and only recompute metrics
        if not any(out_subdir.iterdir()):
            trainer.train()
            trainer.save_model(str(out_subdir))
        else:
            print("  Using existing checkpoint (eval-only).")

        def _probs_from_preds(preds):
            logits = preds.predictions - preds.predictions.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            y_prob = probs[:, 1]
            return y_prob

        # Predict once to get probabilities
        pred_train = trainer.predict(enc_train)
        pred_val = trainer.predict(enc_val)
        pred_test = trainer.predict(enc_test)

        y_val = pred_val.label_ids
        prob_val = _probs_from_preds(pred_val)

        # Choose threshold to maximise validation F1 (balanced precision/recall)
        best_threshold = 0.5
        best_score = -1.0
        for t in np.linspace(0.05, 0.95, 19):
            y_pred_val = (prob_val >= t).astype(np.int64)
            m_val = evaluate_binary(y_val, y_pred_val, prob_val)
            score = m_val["f1"]
            if score >= best_score:
                best_score = score
                best_threshold = float(t)

        def metrics_at_threshold(preds, threshold: float):
            y_true = preds.label_ids
            prob = _probs_from_preds(preds)
            y_pred = (prob >= threshold).astype(np.int64)
            return evaluate_binary(y_true, y_pred, prob)

        train_metrics = metrics_at_threshold(pred_train, best_threshold)
        val_metrics = metrics_at_threshold(pred_val, best_threshold)
        test_metrics = metrics_at_threshold(pred_test, best_threshold)

        all_results[short] = {
            "decision_threshold": best_threshold,
            **test_metrics,
            "train": train_metrics,
            "val": val_metrics,
        }

        out_results = {
            "decision_threshold": best_threshold,
            **test_metrics,
            "train": train_metrics,
            "val": val_metrics,
        }
        with open(results_file, "w") as f:
            json.dump(out_results, f, indent=2)
        print(
            f"  Threshold: {best_threshold:.3f}  "
            f"Train AUC: {train_metrics['auc_roc']:.3f}  "
            f"Val AUC: {val_metrics['auc_roc']:.3f}  "
            f"Test AUC: {test_metrics['auc_roc']:.3f}"
        )

    with open(OUTPUTS_DIR / "transformer_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved transformer_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
