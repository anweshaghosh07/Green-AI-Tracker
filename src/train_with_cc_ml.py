#!/usr/bin/env python3
"""
Baseline + CodeCarbon + MLflow Integration
- Dataset: sklearn digits
- Model: LogisticRegression
- Tracks: accuracy, log_loss, train_time_sec
- Adds: energy_kwh, emissions_kg from CodeCarbon
- Logs everything into MLflow + local CSV/JSON
"""

import os
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

# MLflow
USE_MLFLOW = True
try:
    if USE_MLFLOW:
        import mlflow
except Exception:
    USE_MLFLOW = False
    print("MLflow not available; running without MLflow logging.")

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
EMISSIONS_DIR = DATA_DIR / "emissions"
METRICS_CSV = DATA_DIR / "baseline_metrics.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    dataset_path = ROOT / "dataset" / "digits.npz"
    if dataset_path.exists():
        print(f"Loading dataset from {dataset_path}")
        d = np.load(dataset_path)
        X = d["data"]
        y = d["target"]
    else:
        print("Dataset file not found, using sklearn.load_digits() fallback")
        digits = load_digits()
        X = digits.data
        y = digits.target
    return X, y

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

    # 4) Model
    model = LogisticRegression(max_iter=1000, solver="saga", multi_class="multinomial", n_jobs=-1)

    # ---- CodeCarbon tracker ----
    tracker = EmissionsTracker(
        project_name="digits_mlflow_cc",
        output_dir=str(EMISSIONS_DIR),
        measure_power_secs=1,
        save_to_file=True,
        log_level="error",
        tracking_mode="process"
    )
    tracker.start()

    # 5) Train
    t0 = time.time()
    model.fit(X_train_s, y_train)
    train_time_sec = time.time() - t0

    emissions_kg: dict = tracker.stop()

    # 6) Metrics
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    # 7) Read the CSV to get the extra sustainability metrics.
    emissions_csv = EMISSIONS_DIR / "emissions.csv"
    energy_kwh = None
    cc_duration_s = None
    if emissions_csv.exists():
        df_cc = pd.read_csv(emissions_csv)
        if not df_cc.empty:
            last_run_stats = df_cc.iloc[-1]
            energy_kwh = last_run_stats.get("energy_consumed")
            cc_duration_s = last_run_stats.get("duration")

    print(f"Training time (s): {train_time_sec:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {ll:.4f}")
    print(f"Emissions (kg CO2eq): {emissions_kg:.6f}")
    print("Classification report:\n", classification_report(y_test, y_pred))

    # 8) Save model + scaler
    model_path = MODELS_DIR / "logreg_digits_mlflow_cc.joblib"
    scaler_path = MODELS_DIR / "scaler_digits_mlflow_cc.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # 9) Combine metrics
    metrics = {
        "timestamp": time.time(),
        "train_time_sec": float(train_time_sec),
        "accuracy": float(acc),
        "log_loss": float(ll),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "model": "LogisticRegression",
        "notes": "digits + CodeCarbon + MLflow",
        "emissions_kg": float(emissions_kg) if emissions_kg is not None else 0,
        "energy_kwh": float(energy_kwh) if energy_kwh is not None else 0,
        "cc_duration_s": float(cc_duration_s) if cc_duration_s is not None else 0,
    }

    # 10) Save CSV + JSON
    dfm = pd.DataFrame([metrics])
    dfm.to_csv(METRICS_CSV, mode="a", header=not METRICS_CSV.exists(), index=False)
    with open(DATA_DIR / "last_run_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 11) MLflow logging
    if USE_MLFLOW:
        # Check if running in GitHub Actions CI environment
        is_ci_environment = os.getenv("CI") == "true"

        if not is_ci_environment:
            # Only set the tracking URI if running locally and server is active
            try:
                mlflow.set_tracking_uri("http://127.0.0.1:5000")
                # Verify connection to avoid long timeouts if server isn't running
                client = mlflow.tracking.MlflowClient()
                client.search_experiments() 
            except Exception as e:
                print(f"Warning: Could not connect to MLflow server at http://127.0.0.1:5000. Logging to local './mlruns' instead.")
        # Fallback to default local logging path
        print("MLflow tracking to:", mlflow.get_tracking_uri())
        mlflow.set_experiment("digits_codecarbon_mlflow")

        with mlflow.start_run():
            # Params
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("solver", "saga")
            mlflow.log_param("max_iter", 500)

            # Metrics from the dict (only nemeric)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            # Artifacts
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(scaler_path))
            mlflow.log_artifact(str(DATA_DIR / "last_run_metrics.json"))
            if (EMISSIONS_DIR / "emissions.csv").exists():
                mlflow.log_artifact(str(EMISSIONS_DIR / "emissions.csv"))

        print("Logged run to MLflow.")

    print("Done.")
    return metrics

if __name__ == "__main__":
    main()
