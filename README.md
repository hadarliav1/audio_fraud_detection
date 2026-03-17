<div dir="rtl">

# זיהוי דיבור מזויף מבוסס AI

פרויקט לזיהוי דיבור אמיתי לעומת דיבור סינתטי (TTS).

## שאלת המחקר

האם ניתן לזהות דיבור שנוצר ב-AI בדיוק מספק, ולהיות עמידים לרעש בתנאי שטח? האם שילוב פיצ'רים אקוסטיים עם embeddings של טרנספורמר משפר את הזיהוי?

## ממצאים עיקריים

- **מודל מומלץ:** HuBERT (AUC 0.953 באודיו נקי)
- **עמידות לרעש:** HuBERT ו-WavLM עמידים; Wav2Vec2 קורס ברעש לבן/ורוד
- **Fusion:** שילוב פיצ'רים אקוסטיים לא שיפר — הוחלט לא להשתמש
- **Fine-tuned vs Frozen:** אימון מלא עדיף (~0.95 לעומת ~0.87)

## הרצה

```bash
# התקנת תלויות
pip install -r requirements.txt

# הורדת דאטה (דורש Kaggle API)
python download_dataset.py

# עיבוד מקדים
python scripts/run_preprocessing.py
python scripts/extract_features.py

# אימון מודלים (בסדר: CNN → טרנספורמרים → Fusion → רעש)
python scripts/run_pipeline.py

# יצירת דוח סופי
python scripts/generate_report.py

# אופציונלי: baseline קלאסי (RF/LR על פיצ'רים אקוסטיים)
# python scripts/train_baseline.py

# הסקה על קובץ בודד
python scripts/predict.py path/to/audio.wav
```

## מבנה הפרויקט

| תיקייה | תיאור |
|--------|-------|
| `notebooks/` | ניתוח וניסויים (01–09) |
| `scripts/` | סקריפטי אימון והרצה |
| `src/` | קוד משותף (features, models, utils) |
| `data/` | raw, processed, features |
| `results/` | תוצאות אימון, מטריקות, checkpoints |
| `reports/` | דוח סופי (final_report.md) |

לפרטים נוספים: `PIPELINE.md`, `PROJECT_STRUCTURE.md`, `reports/final_report.md`

</div>
