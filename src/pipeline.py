from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "tickets.csv"
MODEL_PATH = ROOT / "artifacts" / "ticket_priority.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"


# ============================================================
# TRAINING
# ============================================================

def train():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Check required columns
    required_columns = {"text", "urgent"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom wajib hilang: {sorted(missing)}"
        )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["urgent"],
        test_size=0.25,
        random_state=42,
        stratify=df["urgent"],
    )

    # Build model pipeline
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "test_rows": int(len(y_test)),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "classification_report": classification_report(
            y_test,
            pred,
            zero_division=0,
        ),
    }

    # Create directories
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save metrics
    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Display results
    print(json.dumps(metrics, indent=2))
    print(f"Model tersimpan di: {MODEL_PATH}")


# ============================================================
# PREDICTION / INFERENCE
# ============================================================

def predict(text):
    # Check whether model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Jalankan train lebih dahulu"
        )

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict label
    label = int(
        model.predict([text])[0]
    )

    # Get prediction probability
    probability = float(
        model.predict_proba([text])[0][label]
    )

    # Display result
    result = {
        "text": text,
        "urgent": label,
        "confidence": round(
            probability,
            4,
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    command = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "train"
    )

    if command == "train":
        train()

    elif command == "predict" and len(sys.argv) > 2:
        predict(
            " ".join(sys.argv[2:])
        )

    else:
        raise SystemExit(
            'Gunakan: python src/pipeline.py train '
            'atau predict "teks tiket"'
        )
