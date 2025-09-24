#!/usr/bin/env python3
"""
Baseline + CodeCarbon + MLflow Integration (extended)
- Dataset: sklearn digits (default)
- Models supported: LogisticRegression (default), RandomForest (rf), SVM (svm), MLP (mlp), small CNN (cnn)
- Tracks: accuracy, log_loss (when proba available), train_time_sec
- Adds: energy_kwh, emissions_kg from CodeCarbon
- Logs: MLflow + local CSV/JSON (data/baseline_metrics.csv, data/last_run_metrics.json)
Usage examples:
  python src/train_with_cc_ml.py --model rf --n_runs 3
  python src/train_with_cc_ml.py --model cnn --cnn_dataset mnist --epochs 3 --n_runs 2
"""

import argparse
import os
import time
import json
import uuid
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

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

def build_small_cnn_for_digits(input_shape, num_classes):
    # Lazy import to avoid TF when not using CNN
    from tensorflow.keras import layers, models
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_mnist_like(which="mnist"):
    import tensorflow as tf

    if which == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif which == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError("cnn_dataset must be one of: digits,mnist,fashion_mnist")

    # normalize and add channel
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, x_test, y_train, y_test


def append_to_csv(csv_path: Path, row: dict):
    df_row = pd.DataFrame([row])
    df_row.to_csv(csv_path, index=False, mode="a", header=not csv_path.exists())


def extract_emissions_info(tracker_stop_return, emissions_csv_path: Path):
    """
    tracker_stop_return might be a float (kg) or an object.
    emissions_csv_path points to codecarbon's emissions.csv where extra fields exist.
    """
    co2_kg = None
    energy_kwh = None
    cc_duration_s = None

    # Try direct parse
    try:
        if isinstance(tracker_stop_return, (int, float)):
            co2_kg = float(tracker_stop_return)
        else:
            # object: try common attributes
            co2_kg = float(getattr(tracker_stop_return, "emissions", None) or getattr(tracker_stop_return, "emissions_kg", None))
    except Exception:
        co2_kg = None

    # Try emissions.csv for energy and duration and last-run row
    if emissions_csv_path.exists():
        try:
            df_cc = pd.read_csv(emissions_csv_path)
            if not df_cc.empty:
                last = df_cc.iloc[-1]
                # CodeCarbon column names differ by version; try a few options:
                energy_kwh = last.get("energy_consumed") or last.get("energy_kwh") or last.get("energy")
                cc_duration_s = last.get("duration") or last.get("runtime_seconds") or last.get("duration_s")
                # If co2_kg missing, try csv too
                if co2_kg is None:
                    co2_kg = last.get("emissions") or last.get("emissions_kg") or last.get("co2_kg")
        except Exception:
            pass

    # cast numeric if possible
    try:
        co2_kg = float(co2_kg) if co2_kg is not None and co2_kg != "" else None
    except Exception:
        co2_kg = None
    try:
        energy_kwh = float(energy_kwh) if energy_kwh is not None and energy_kwh != "" else None
    except Exception:
        energy_kwh = None
    try:
        cc_duration_s = float(cc_duration_s) if cc_duration_s is not None and cc_duration_s != "" else None
    except Exception:
        cc_duration_s = None

    return co2_kg, energy_kwh, cc_duration_s


def train_one_run(model_name: str, X_train, X_test, y_train, y_test, scaler, args, seed: int):
    """
    Train a single model run wrapped by CodeCarbon and MLflow logging.
    Returns metrics dict.
    """
    run_uuid = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    start_wall = time.time()
    start_perf = time.perf_counter()
    n_train_samples, n_test_samples = 0, 0

    # Start CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="digits_mlflow_cc",
        output_dir=str(EMISSIONS_DIR),
        measure_power_secs=1,
        save_to_file=True,
        log_level="error",
        tracking_mode="process",
    )
    tracker.start()

    # Instantiate model
    model = None
    model_fname = None
    fit_time = None
    y_proba = None

    try:
        if model_name == "logreg":
            model = LogisticRegression(max_iter=1000, solver="saga", multi_class="multinomial", n_jobs=-1)
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            model_fname = MODELS_DIR / f"logreg_{run_uuid}.joblib"
            joblib.dump(model, model_fname)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

        elif model_name == "rf":
            model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=seed, n_jobs=-1)
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            model_fname = MODELS_DIR / f"rf_{run_uuid}.joblib"
            joblib.dump(model, model_fname)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)

        elif model_name == "svm":
            # enable probability=True to allow log_loss calculation (slower)
            model = SVC(probability=True, random_state=seed)
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            model_fname = MODELS_DIR / f"svm_{run_uuid}.joblib"
            joblib.dump(model, model_fname)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)

        elif model_name == "cnn":
            # Build and train small CNN. Support digits (8x8) or mnist/fashion_mnist
            cnn_dataset = args.cnn_dataset.lower()
            if cnn_dataset == "digits":
                # digits images are 8x8 flattened
                n_samples = X_train.shape[0] + X_test.shape[0]
                img_side = 8
                # reconstruct arrays from flattened X_train/X_test (they are already passed)
                # In our pipeline we will pass X_train/X_test as flattened arrays; reshape here
                X_train_img = X_train.reshape((-1, img_side, img_side, 1)).astype("float32") / 16.0
                X_test_img = X_test.reshape((-1, img_side, img_side, 1)).astype("float32") / 16.0
                num_classes = int(np.max(y_train)) + 1
                model_tf = build_small_cnn_for_digits((img_side, img_side, 1), num_classes)
                t0 = time.time()
                history = model_tf.fit(
                    X_train_img,
                    y_train,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    validation_split=0.1,
                    verbose=0,
                )
                fit_time = time.time() - t0  # record actual training time
                eval_res = model_tf.evaluate(X_test_img, y_test, verbose=0)
                acc_val = float(eval_res[1]) if len(eval_res) > 1 else None
                y_pred = np.argmax(model_tf.predict(X_test_img), axis=1)
                y_proba = model_tf.predict(X_test_img)
                model_fname = MODELS_DIR / f"cnn_digits_{run_uuid}.h5"
                model_tf.save(model_fname)
            else:
                # mnist / fashion_mnist
                x_train_img, x_test_img, y_train_c, y_test_c = load_mnist_like(cnn_dataset)
                n_train_samples = x_train_img.shape[0]
                n_test_samples = x_test_img.shape[0]
                X_train = x_train_img   # just so metrics dict doesn’t break
                X_test = x_test_img
                y_train = y_train_c
                y_test = y_test_c

                # If original X_train/X_test were the digits arrays, we prefer the dataset from TF loader
                num_classes = int(np.max(y_train_c)) + 1
                model_tf = build_small_cnn_for_digits(x_train_img.shape[1:], num_classes)
                t0 = time.time()
                history = model_tf.fit(
                    x_train_img,
                    y_train_c,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    validation_split=0.1,
                    verbose=0,
                )
                fit_time = time.time() - t0  # record actual training time
                eval_res = model_tf.evaluate(x_test_img, y_test_c, verbose=0)
                acc_val = float(eval_res[1]) if len(eval_res) > 1 else None
                y_pred = np.argmax(model_tf.predict(x_test_img), axis=1)
                y_proba = model_tf.predict(x_test_img)
                # override y_test so metrics calculation works
                y_test = y_test_c

                model_fname = MODELS_DIR / f"cnn_{cnn_dataset}_{run_uuid}.h5"
                model_tf.save(model_fname)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    finally:
        # Always stop tracker
        emissions_obj = tracker.stop()
        co2_kg, energy_kwh, cc_duration_s = extract_emissions_info(emissions_obj, EMISSIONS_DIR / "emissions.csv")

    end_wall = time.time()
    total_runtime = end_wall - start_wall

    # Calculate metrics
    acc = None
    ll = None
    try:
        if y_test is not None:
            acc = float(accuracy_score(y_test, y_pred))
        if y_proba is not None:
            ll = float(log_loss(y_test, y_proba))
    except Exception as e:
        print("⚠️Metric calculation failed:", e)

    # Save scaler and model artifact path info
    scaler_path = None
    if scaler is not None and model_name != "cnn":  # we saved joblib models for tabular
        scaler_path = MODELS_DIR / f"scaler_{model_name}_{run_uuid}.joblib"
        joblib.dump(scaler, scaler_path)

    # Build metrics dict
    metrics = {
        "timestamp": datetime.utcnow().timestamp(),
        "human_timestamp": timestamp,
        "run_uuid": run_uuid,
        "model": model_name,
        "accuracy": acc,
        "log_loss": float(ll) if ll is not None else "",
        "train_time_sec": float(fit_time) if fit_time is not None else float(total_runtime),
        "runtime_sec": float(total_runtime),
        "n_train": int(X_train.shape[0]) if X_train is not None else n_train_samples,
        "n_test": int(X_test.shape[0]) if X_test is not None else n_test_samples,
        "seed": int(seed),
        "notes": f"{model_name} + CodeCarbon + MLflow",
        "emissions_kg": float(co2_kg) if co2_kg is not None else "",
        "energy_kwh": float(energy_kwh) if energy_kwh is not None else "",
        "cc_duration_s": float(cc_duration_s) if cc_duration_s is not None else "",
        "model_artifact": str(model_fname) if model_fname is not None else "",
        "scaler_artifact": str(scaler_path) if scaler_path is not None else "",
    }

    # save last_run_metrics.json
    with open(DATA_DIR / "last_run_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"[{model_name} run] acc={acc:.4f} ll={ll} co2_kg={metrics['emissions_kg']} runtime_s={metrics['runtime_sec']:.2f}")

    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train models with CodeCarbon + MLflow")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf", "svm", "cnn"], help="Which model to train")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_estimators", type=int, default=100, help="RF n_estimators")
    parser.add_argument("--epochs", type=int, default=5, help="CNN epochs (if using cnn)")
    parser.add_argument("--batch_size", type=int, default=128, help="CNN batch size")
    parser.add_argument("--cnn_dataset", type=str, default="digits", choices=["digits", "mnist", "fashion_mnist"])
    parser.add_argument("--mlflow_experiment", type=str, default="digits_codecarbon_mlflow")  
    parser.add_argument("--hidden_layers", type=int, default=1, help="Number of hidden layers for MLP (future use)")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for solvers like Logistic Regression")
  
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    print("Dataset shape:", X.shape, "Labels:", np.unique(y).shape)

    # If using CNN and dataset selected is mnist/fashion_mnist we will re-load inside train_one_run
    # For tabular models
    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

    # 3) Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4) MLflow logging
    if USE_MLFLOW:
        # Get the MLflow URI from the environment variable if it exists, otherwise default to localhost
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()
            client.search_experiments()  # quick ping
            print(f"Successfully connected to MLflow server at {mlflow_tracking_uri}")
        except Exception:
            print(f"Warning: Could not connect to MLflow server at {mlflow_tracking_uri}. Using local mlruns/ instead.")
        mlflow.set_experiment(args.mlflow_experiment)

    # Run loop
    for i in range(args.n_runs):
        seed = args.seed + i
        set_seeds(seed)

        # For CNN with TF using mnist/fashion_mnist, load_mnist_like inside train_one_run will be used.
        # For cnn with digits, pass the flattened arrays and reshape inside train_one_run.
        if args.model == "cnn" and args.cnn_dataset in ("mnist", "fashion_mnist"):
            # We don't need the tabular X_train/X_test in that case, pass placeholders
            X_train_in, X_test_in, y_train_in, y_test_in = None, None, None, None
        else:
            X_train_in, X_test_in, y_train_in, y_test_in = X_train_s, X_test_s, y_train, y_test

        # Start MLflow run (if available)
        if USE_MLFLOW:
            with mlflow.start_run():
                mlflow.log_param("model_type", args.model)
                mlflow.log_param("seed", seed)
                # Model-specific params
                mlflow.log_param("n_estimators", args.n_estimators)
                mlflow.log_param("hidden_layers", args.hidden_layers)
                mlflow.log_param("max_iter", args.max_iter)
                mlflow.log_param("epochs", args.epochs)
                mlflow.log_param("cnn_dataset", args.cnn_dataset)

                metrics = train_one_run(args.model, X_train_in, X_test_in, y_train_in, y_test_in, scaler, args, seed)

                # log numeric metrics to MLflow
                numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(numeric_metrics)

                # log artifacts
                if metrics.get("model_artifact"):
                    try:
                        mlflow.log_artifact(metrics["model_artifact"], artifact_path="models")
                    except Exception:
                        pass
                if metrics.get("scaler_artifact"):
                    try:
                        mlflow.log_artifact(metrics["scaler_artifact"], artifact_path="models")
                    except Exception:
                        pass
                # log CodeCarbon csv if available
                if (EMISSIONS_DIR / "emissions.csv").exists():
                    try:
                        mlflow.log_artifact(str(EMISSIONS_DIR / "emissions.csv"), artifact_path="emissions")
                    except Exception:
                        pass

        else:
            # Not using MLflow: just run and get metrics and append
            metrics = train_one_run(args.model, X_train_in, X_test_in, y_train_in, y_test_in, scaler, args, seed)

        # Append to CSV
        metrics_for_csv = {
            "run_id": metrics.get("run_uuid"),
            "timestamp": metrics.get("timestamp"),
            "human_timestamp": metrics.get("human_timestamp"), 
            "model": metrics.get("model"),
            "accuracy": metrics.get("accuracy"),
            "log_loss": metrics.get("log_loss"),
            "train_time_sec": metrics.get("train_time_sec"),
            "runtime_sec": metrics.get("runtime_sec"),
            "n_train": metrics.get("n_train"),
            "n_test": metrics.get("n_test"),
            "seed": metrics.get("seed"),
            "emissions_kg": metrics.get("emissions_kg"),
            "energy_kwh": metrics.get("energy_kwh"),
            "cc_duration_s": metrics.get("cc_duration_s"),
            "model_artifact": metrics.get("model_artifact"),
            "scaler_artifact": metrics.get("scaler_artifact"),
            "notes": metrics.get("notes"),
        }
        dfm = pd.DataFrame([metrics_for_csv])
        dfm.to_csv(METRICS_CSV, mode="a", header=not METRICS_CSV.exists(), index=False)

    print("All runs complete. Appended results to", METRICS_CSV)

if __name__ == "__main__":
    main()
