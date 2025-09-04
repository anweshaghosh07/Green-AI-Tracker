#!/usr/bin/env python3
"""
Baseline training + CodeCarbon integration.
- Dataset: sklearn digits
- Model: LogisticRegression
- Tracks: accuracy, log_loss, train_time_sec
- Adds: energy_kwh, emissions_kg from CodeCarbon
- Logs: CSV/JSON locally and MLflow
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

# CodeCarbon
from codecarbon import EmissionsTracker

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
EMISSIONS_DIR = DATA_DIR / "emissions"
METRICS_CSV = DATA_DIR / "baseline_metrics.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    print("Dataset shape:", X.shape, "Labels:", np.unique(y).shape)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4) Model (baseline)
    model = LogisticRegression(max_iter=300, solver='saga', multi_class='multinomial', n_jobs=-1)

    # ---- CodeCarbon tracker (wrap the training loop) ----
    tracker = EmissionsTracker(
        project_name="baseline_digits_cc",
        output_dir=str(EMISSIONS_DIR),
        measure_power_secs=1,        # sampling interval (seconds)
        save_to_file=True,
        log_level="error",
        tracking_mode="process"      # track this process only (safe default)
    )
    tracker.start()

    # 5) Train & measure wall-clock time
    t0 = time.time()
    model.fit(X_train_s, y_train)
    train_time_sec = time.time() - t0

    # stop tracker -> returns emissions in kg CO2eq
    emissions_kg = tracker.stop()

    # 6) Predictions & metrics
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"Training time (s): {train_time_sec:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {ll:.4f}")
    print(f"Emissions (kg CO2eq): {emissions_kg:.6f}")
    print("Classification report:\n", classification_report(y_test, y_pred))

    # 7) Read latest row from CodeCarbon's emissions.csv for energy & duration
    emissions_csv = EMISSIONS_DIR / "emissions.csv"
    energy_kwh = None
    cc_duration_s = None
    if emissions_csv.exists():
        df_cc = pd.read_csv(emissions_csv)
        last = df_cc.tail(1).squeeze()
        # column names typically include: emissions, energy_consumed, duration, etc.
        energy_kwh = float(last.get("energy_consumed", np.nan))
        cc_duration_s = float(last.get("duration", np.nan))

    # 8) Save model & scaler
    model_path = MODELS_DIR / "baseline_logreg_cc.joblib"
    scaler_path = MODELS_DIR / "scaler_cc.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to: {model_path}")

    # 9) Build combined metrics record
    metrics = {
        "timestamp": time.time(),
        "train_time_sec": float(train_time_sec),
        "accuracy": float(acc),
        "log_loss": float(ll),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "model": "LogisticRegression",
        "notes": "digits + CodeCarbon",
        "emissions_kg": float(emissions_kg) if emissions_kg is not None else None,
        "energy_kwh": float(energy_kwh) if energy_kwh is not None else None,
        "cc_duration_s": float(cc_duration_s) if cc_duration_s is not None else None,
    }

    # 10) Append to baseline_metrics.csv
    dfm = pd.DataFrame([metrics])
    if METRICS_CSV.exists():
        dfm.to_csv(METRICS_CSV, mode='a', header=False, index=False)
    else:
        dfm.to_csv(METRICS_CSV, index=False)

    # 11) Save JSON summary (easy to load in Streamlit / log as artifact)
    with open(DATA_DIR / "last_emissions.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 12) MLflow logging
    if USE_MLFLOW:
        mlflow.set_experiment("digits_with_codecarbon")
        with mlflow.start_run():
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("log_loss", float(ll))
            mlflow.log_metric("train_time_sec", float(train_time_sec))
            if emissions_kg is not None:
                mlflow.log_metric("emissions_kg", float(emissions_kg))
            if energy_kwh is not None:
                mlflow.log_metric("energy_kwh", float(energy_kwh))
            if cc_duration_s is not None:
                mlflow.log_metric("cc_duration_s", float(cc_duration_s))
            mlflow.log_artifact(str(emissions_csv))
            mlflow.log_artifact(str(DATA_DIR / "last_emissions.json"))
            mlflow.log_artifact(str(model_path))
        print("Logged run to MLflow.")

    print("Done.")
    return metrics

if __name__ == "__main__":
    main()
