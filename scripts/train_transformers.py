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
from src.utils.splits import speaker_disjoint_split


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
    np.random.seed(RANDOM_SEED)

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
        print(f"\n--- Training {model_id} ---")
        out_subdir = OUTPUTS_DIR / f"transformer_{short}"
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

        trainer.train()
        trainer.save_model(str(out_subdir))

        preds = trainer.predict(enc_test)
        y_pred = np.argmax(preds.predictions, axis=1)
        logits = preds.predictions - preds.predictions.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y_prob = probs[:, 1]

        metrics = evaluate_binary(preds.label_ids, y_pred, y_prob)
        all_results[short] = metrics

        with open(out_subdir / "results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Test AUC: {metrics['auc_roc']:.3f}")

    with open(OUTPUTS_DIR / "transformer_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved transformer_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
