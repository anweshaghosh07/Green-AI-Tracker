#!/usr/bin/env python3
"""
Baseline training script.
- Dataset: sklearn digits (small image dataset)
- Model: LogisticRegression (multiclass)
- Outputs: prints metrics, saves model, writes metrics CSV
- MLflow logging (set USE_MLFLOW=True)
"""
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
import joblib

# MLflow logging
USE_MLFLOW = True
try:
    if USE_MLFLOW:
        import mlflow
except Exception:
    USE_MLFLOW = False
    print("MLflow not available; running without mlflow logging.")

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
METRICS_CSV = ROOT / "data" / "baseline_metrics.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    print("Dataset shape:", X.shape, "Labels:", np.unique(y).shape)

    # 2) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4) Model (baseline)
    model = LogisticRegression(max_iter=500, solver='saga', multi_class='multinomial', n_jobs=-1)

    # 5) Train & measure time
    start = time.time()
    model.fit(X_train_s, y_train)
    train_time_sec = time.time() - start

    # 6) Predictions & metrics
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"Training time (s): {train_time_sec:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {ll:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # 7) Save the model & scaler
    model_path = MODELS_DIR / "baseline_logreg.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to: {model_path}")

    # 8) Save metrics to CSV/JSON for later visualization
    metrics = {
        "timestamp": time.time(),
        "train_time_sec": train_time_sec,
        "accuracy": float(acc),
        "log_loss": float(ll),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "model": "LogisticRegression",
        "notes": "sklearn digits baseline"
    }

    # Append to CSV
    dfm = pd.DataFrame([metrics])
    if METRICS_CSV.exists():
        dfm.to_csv(METRICS_CSV, mode='a', header=False, index=False)
    else:
        dfm.to_csv(METRICS_CSV, index=False)

    # Save separate JSON for quick MLflow artifact upload if needed
    with open(DATA_DIR / "last_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 9) MLflow logging
    if USE_MLFLOW:
        mlflow.set_experiment("baseline_digits_experiment")
        with mlflow.start_run():
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("log_loss", float(ll))
            mlflow.log_metric("train_time_sec", float(train_time_sec))
            # log artifacts: model & metrics json
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(DATA_DIR / "last_metrics.json"))
        print("Logged run to MLflow.")

    print("Done.")
    
if __name__ == '__main__':
    main()