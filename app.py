import json
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib

# ML imports for fallback training
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# CodeCarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    CODECARBON_AVAILABLE = False

# MLflow import 
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# Paths
ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EMISSIONS_DIR = DATA_DIR / "emissions"
METRICS_CSV = DATA_DIR / "baseline_metrics.csv"
EMISSIONS_CSV = EMISSIONS_DIR / "emissions.csv"
LAST_JSON = DATA_DIR / "last_run_metrics.json"

# Ensure dirs exist
for p in [DATA_DIR, MODELS_DIR, EMISSIONS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Fallback training function (runs only if no metrics exist)
# -------------------------------------------------------------------
def run_fallback_training():
    st.warning("⚠️ No metrics found. Running a quick fallback training...")

    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200, solver="saga", multi_class="multinomial", n_jobs=-1)

    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name="fallback_digits",
            output_dir=str(EMISSIONS_DIR),
            measure_power_secs=1,
            save_to_file=True,
            log_level="error",
            tracking_mode="process"
        )
        tracker.start()

    t0 = time.time()
    model.fit(X_train_s, y_train)
    train_time_sec = time.time() - t0

    emissions_kg = tracker.stop() if tracker else 0.0

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    # Read CodeCarbon outputs if present
    energy_kwh, cc_duration_s = 0, 0
    if EMISSIONS_CSV.exists():
        df_cc = pd.read_csv(EMISSIONS_CSV)
        if not df_cc.empty:
            last = df_cc.iloc[-1]
            energy_kwh = float(last.get("energy_consumed", 0))
            cc_duration_s = float(last.get("duration", 0))

    # Save artifacts
    model_path = MODELS_DIR / "logreg_fallback.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "timestamp": time.time(),
        "train_time_sec": train_time_sec,
        "accuracy": acc,
        "log_loss": ll,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "model": "LogisticRegression",
        "notes": "fallback run (auto-triggered)",
        "emissions_kg": emissions_kg,
        "energy_kwh": energy_kwh,
        "cc_duration_s": cc_duration_s,
    }

    # Save CSV + JSON
    dfm = pd.DataFrame([metrics])
    dfm.to_csv(METRICS_CSV, mode="a", header=not METRICS_CSV.exists(), index=False)
    with open(LAST_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    st.success("✅ Fallback training complete. Data generated!")

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Green AI Tracker", page_icon="🌍", layout="centered")
st.title("Sustainability & Green-AI🌍Usage Tracker")
st.markdown(
    """
    Visualize the performance and environmental impact of machine learning model training runs.
    If no data is available, a quick fallback training will be executed automatically.
    """
)

# Sidebar controls
st.sidebar.header("🔹Data & Options")
use_mlflow = st.sidebar.checkbox("Use MLflow (if available)", value=False and MLFLOW_AVAILABLE)
st.sidebar.markdown("If MLflow is enabled, the app will try to fetch runs from the local tracking server.")

# Trigger fallback training if no metrics exist
if not METRICS_CSV.exists() or not LAST_JSON.exists():
    run_fallback_training()

# Load Data
@st.cache_data
def load_metrics_csv(path: Path):
    if not path.exists():
        return None
    # Read CSV safely, skip malformed lines
    df = pd.read_csv(path, on_bad_lines="skip")
    # Define the superset of expected columns
    expected_cols = [
        "timestamp", "train_time_sec", "accuracy", "log_loss", "n_train", "n_test", "model", "notes", "emissions_kg", "energy_kwh", "cc_duration_s",
    ]
    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None  # fill missing columns (for older runs)
    # Add run index
    df = df.reset_index(drop=True)
    df["run_idx"] = np.arange(1, len(df) + 1)
    # Add readable timestamp if present
    if "timestamp" in df.columns:
        try:
            df["ts_readable"] = pd.to_datetime(df["timestamp"], unit="s")
        except Exception:
            pass
    # Return only the standardized set of columns (plus helpers)
    return df[expected_cols + ["run_idx", "ts_readable"] if "ts_readable" in df else expected_cols + ["run_idx"]]

@st.cache_data
def load_last_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

metrics_df = load_metrics_csv(METRICS_CSV)
last_json = load_last_json(LAST_JSON)

# Option: load MLflow runs
mlflow_df = None
if use_mlflow and MLFLOW_AVAILABLE:
    try:
        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        # list experiments, choose one if available
        exps = client.search_experiments()
        if exps:
            # pick the first experiment (or show selectbox in UI)
            exp = st.sidebar.selectbox("Select experiment", [e.name for e in exps])
            exp_id = [e.experiment_id for e in exps if e.name == exp][0]
            runs = client.search_runs(exp_id, order_by=["attributes.start_time DESC"], max_results=50)
            rows = []
            for r in runs:
                d = r.data
                row = {
                    "run_id": r.info.run_id,
                    "start_time": r.info.start_time,
                }
                # copy over common metrics if present
                for k, v in d.metrics.items():
                    row[k] = v
                for k, v in d.params.items():
                    row[f"param_{k}"] = v
                rows.append(row)
            if rows:
                mlflow_df = pd.DataFrame(rows)
                mlflow_df["run_idx"] = np.arange(1, len(mlflow_df) + 1)
    except Exception as e:
        st.sidebar.error(f"MLflow fetch failed: {e}")

# Show data status
st.subheader("Data status")
col1, col2, col3 = st.columns(3)
with col1:
    if metrics_df is None:
        st.warning(f"No CSV metrics found at: {METRICS_CSV}")
    else:
        st.success(f"Loaded metrics CSV — {len(metrics_df)} runs")
with col2:
    if EMISSIONS_CSV.exists():
        st.success(f"Found CodeCarbon CSV: {EMISSIONS_CSV.name}")
    else:
        st.info("No CodeCarbon CSV found yet.")
with col3:
    if last_json:
        st.success("Found last_run_metrics.json")
    else:
        st.info("No last_run_metrics.json found.")

# Choose source (CSV preferred, then MLflow)
source = "csv"
if mlflow_df is not None and st.sidebar.checkbox("Prefer MLflow data", value=False):
    source = "mlflow"
elif metrics_df is None and mlflow_df is not None:
    source = "mlflow"
elif metrics_df is None:
    source = "none"

# Main visualizations
st.markdown("---")
st.header("📈 Metrics Visualizations")

if source == "none":
    st.info("No metrics available yet. Run the training script to produce metrics (baseline_metrics.csv & emissions.csv).")
else:
    if source == "csv":
        df = metrics_df.copy()
    else:
        df = mlflow_df.copy()

    # --- Clean numeric columns globally ---
    numeric_cols_to_fix = ["accuracy", "log_loss", "train_time_sec", "energy_kwh", "emissions_kg", "cc_duration_s",]
    for col in numeric_cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # select numeric columns available
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    st.subheader("Available numeric metrics")
    st.write(numeric_cols)

    # Line chart: Accuracy vs Run
    st.subheader("Accuracy vs Run")
    if "accuracy" in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df["run_idx"], df["accuracy"], marker="o")
        ax.set_xlabel("Run (index)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy per Run")
        st.pyplot(fig)
    else:
        st.info("No 'accuracy' column available in the selected source.")

    # Bar chart: Emissions vs Run
    st.subheader("Emissions (kg CO₂ eq.) vs Run")
    if "emissions_kg" in df.columns:
        # ensure numeric and no None values
        df["emissions_kg"] = pd.to_numeric(df["emissions_kg"], errors="coerce").fillna(0)
        
        if df["emissions_kg"].sum() == 0:
            st.info("No emissions data available yet.")
        else:
            fig2, ax2 = plt.subplots()
            ax2.bar(df["run_idx"], df["emissions_kg"])
            ax2.set_xlabel("Run (index)")
            ax2.set_ylabel("Emissions (kg CO₂ eq.)")
            ax2.set_title("Emissions per Run")
            st.pyplot(fig2)
    else:
        st.info("No 'emissions_kg' column available.")

    # Scatter: Accuracy vs Emissions
    st.subheader("Accuracy vs Emissions (trade-off)")
    if "accuracy" in df.columns and "emissions_kg" in df.columns:
        fig3, ax3 = plt.subplots()
        ax3.scatter(df["emissions_kg"], df["accuracy"])
        for i, txt in enumerate(df["run_idx"].astype(str)):
            ax3.annotate(txt, (df["emissions_kg"].iloc[i], df["accuracy"].iloc[i]))
        ax3.set_xlabel("Emissions (kg CO2eq)")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Accuracy vs Emissions")
        st.pyplot(fig3)
    else:
        st.info("Need both 'accuracy' and 'emissions_kg' to show this plot.")

    # Train time vs Energy
    st.subheader("Train time vs Energy (kWh)")
    if "train_time_sec" in df.columns and "energy_kwh" in df.columns:
        fig4, ax4 = plt.subplots()
        ax4.scatter(df["train_time_sec"], df["energy_kwh"])
        ax4.set_xlabel("Train time (s)")
        ax4.set_ylabel("Energy (kWh)")
        ax4.set_title("Train time vs Energy")
        st.pyplot(fig4)
    else:
        st.info("Need 'train_time_sec' and 'energy_kwh' to show this plot.")

    # Show data table and allow download
    st.subheader("Metrics table")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download metrics CSV", data=csv, file_name="metrics_export.csv", mime="text/csv")

# Show last run summary (from last_run_metrics.json)
st.markdown("---")
st.header("📊 Latest Run Summary")
if last_json:
    st.json(last_json)
else:
    st.info("No last_run_metrics.json found — run the training script to produce a summary.")

