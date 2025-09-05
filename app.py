import json
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
METRICS_CSV = DATA_DIR / "baseline_metrics.csv"
EMISSIONS_CSV = DATA_DIR / "emissions" / "emissions.csv"
LAST_JSON = DATA_DIR / "last_run_metrics.json"

st.set_page_config(page_title="Green AI Tracker", page_icon="🌍", layout="wide")

st.title("Sustainability & Green-AI🌍Usage Tracker")
st.markdown(
    """
    Visualize the performance and environmental impact of machine learning model training runs.
    Data is logged from experiments and updated here automatically upon refresh.
    """
)

# Sidebar controls
st.sidebar.header("🔹Data & Options")
use_mlflow = st.sidebar.checkbox("Use MLflow (if available)", value=False and MLFLOW_AVAILABLE)
st.sidebar.markdown("If MLflow is enabled, the app will try to fetch runs from the local tracking server.")

# Load CSV metrics (preferred)
@st.cache_data
def load_metrics_csv(path: Path):
    if not path.exists():
        return None
    # Read CSV safely, skip malformed lines
    df = pd.read_csv(path, on_bad_lines="skip")
    # Define the superset of expected columns
    expected_cols = [
        "timestamp",
        "train_time_sec",
        "accuracy",
        "log_loss",
        "n_train",
        "n_test",
        "model",
        "notes",
        "emissions_kg",
        "energy_kwh",
        "cc_duration_s",
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
    numeric_cols_to_fix = [
        "accuracy",
        "log_loss",
        "train_time_sec",
        "energy_kwh",
        "emissions_kg",
        "cc_duration_s",
    ]
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

