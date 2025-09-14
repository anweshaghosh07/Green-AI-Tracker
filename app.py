#!/usr/bin/env python3
"""
Streamlit Dashboard for Green-AI Usage Tracker
- Reads metrics from CSV/JSON/DVC/CodeCarbon
- If no metrics found, runs fallback training (LogisticRegression on digits)
- Shows accuracy vs CO₂, trends, filters, downloads, efficiency, and CO2 savings
"""
import io
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ML/metrics for fallback
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import joblib
from codecarbon import EmissionsTracker

# -------- CONFIG --------
ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
METRICS_CSV = DATA_DIR / "baseline_metrics.csv"
EMISSIONS_DIR = DATA_DIR / "emissions" 
LAST_JSON = DATA_DIR / "last_run_metrics.json"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Green AI Tracker", page_icon="🌍", layout="wide")

# ---------- FALLBACK TRAINING ----------
def run_fallback_training():
    st.warning("No metrics found — Running fallback baseline training...")

    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200, solver="saga", multi_class="multinomial")

    tracker = EmissionsTracker(
        project_name="fallback_digits",
        output_dir=str(EMISSIONS_DIR),
        log_level="error",
        tracking_mode="process"
    )
    tracker.start()
    t0 = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0
    emissions_kg = tracker.stop()

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    metrics = {
        "timestamp": time.time(),
        "train_time_sec": float(train_time),
        "accuracy": float(acc),
        "log_loss": float(ll),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "model": "LogisticRegression",
        "notes": "fallback run (digits)",
        "emissions_kg": float(emissions_kg) if emissions_kg else None,
    }

    # Save artifacts
    joblib.dump(model, MODELS_DIR / "fallback_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "fallback_scaler.joblib")

    pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)
    with open(LAST_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    return pd.DataFrame([metrics])

# -------- LOAD METRICS --------
@st.cache_data(show_spinner=False)
def load_metrics():
    """
    Loads metrics from baseline_metrics.csv (preferred).
    If not present, tries last_run_metrics.json.
    Also attempts to load codecarbon emissions CSV and merge by timestamp if possible.
    """
    df = None
    # 1) Try baseline_metrics.csv
    if METRICS_CSV.exists():
        df = pd.read_csv(METRICS_CSV)
    elif LAST_JSON.exists():
        df = pd.read_json(LAST_JSON, orient="records")
        # ensure consistent columns if single dict
        if isinstance(df, dict):
            df = pd.DataFrame([df])
    else:
        df = pd.DataFrame()

    # normalize column names / add missing columns
    if not df.empty:
        # Some scripts name emissions column 'emissions' or 'emissions_kg'
        if "emissions_kg" in df.columns:
            df["emissions_kg"] = pd.to_numeric(df["emissions_kg"], errors="coerce")
        elif "emissions" in df.columns:
            df["emissions_kg"] = pd.to_numeric(df["emissions"], errors="coerce")
        else:
            df["emissions_kg"] = np.nan

        # energy may be named 'energy_consumed' or 'energy_kwh' or 'energy'
        for c in ("energy_kwh", "energy_consumed", "energy"):
            if c in df.columns:
                df["energy_kwh"] = pd.to_numeric(df[c], errors="coerce")
                break
        if "energy_kwh" not in df.columns:
            df["energy_kwh"] = np.nan

        # timestamp normalization: supports unix epoch (seconds) or ISO strings
        if "timestamp" in df.columns:
            # some timestamps are floats in seconds
            try:
                # if floats -> epoch
                if np.issubdtype(df["timestamp"].dtype, np.number):
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
                else:
                    df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            # fallback: try to parse a "date" or "datetime" column
            for c in ("date", "datetime"):
                if c in df.columns:
                    df["datetime"] = pd.to_datetime(df[c], errors="coerce")
                    break
            else:
                # if nothing exists, create a fake incremental datetime for display
                df["datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df)).to_pydatetime()
    else:
        # empty df - produce empty columns we expect
        df = pd.DataFrame(columns=[
            "timestamp", "train_time_sec", "accuracy", "log_loss", "n_train",
            "n_test", "model", "notes", "emissions_kg", "energy_kwh", "datetime"
        ])
    # attempt to merge with CodeCarbon csv (gives energy_kwh, duration etc.)
    if EMISSIONS_DIR.exists():
        try:
            df_cc = pd.read_csv(EMISSIONS_DIR)
            # normalize cc emissions column names
            if "emissions" in df_cc.columns and "emissions_kg" not in df.columns:
                df_cc = df_cc.rename(columns={"emissions": "emissions_kg"})
            # find a way to join: many times we'll simply append latest emissions if timestamps are near
            # We'll create a df_cc_summary keyed by nearest timestamp (if both contain datetime)
            if "timestamp" in df_cc.columns:
                try:
                    df_cc["datetime"] = pd.to_datetime(df_cc["timestamp"], unit="s")
                except Exception:
                    df_cc["datetime"] = pd.to_datetime(df_cc["timestamp"], errors="coerce")
            merged = pd.merge_asof(
                df.sort_values("datetime"),
                df_cc.sort_values("datetime"),
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta("1m")  # only merge if timestamps within 1 minute
            )
            # prefer emissions_kg from baseline if present else from merged
            merged["emissions_kg"] = merged["emissions_kg_x"].fillna(merged.get("emissions_kg_y"))
            merged["energy_kwh"] = merged["energy_kwh_x"].fillna(merged.get("energy_consumed"))
            # cleanup duplicated columns
            merged = merged.rename(columns={c: c.replace("_x", "") for c in merged.columns})
            return merged
        except Exception:
            return df
    return df


def format_datetime_col(df):
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        df["date"] = pd.NaT
    return df

# -------- UI --------
st.title("🌱 Green-AI Usage Tracker")

with st.expander("📖 Project Description", expanded=True):
    st.markdown("""
    This dashboard tracks **Green-AI usage metrics**:
    - Accuracy vs CO₂ trade-offs across models, datasets, and tasks.
    - Aggregate KPIs (mean, median, efficiency).
    - Trend analysis over time.
    - CO₂ savings when switching to smaller/efficient models.
    """)

# Load data
df = load_metrics()
if df.empty or "datetime" not in df.columns or df["datetime"].dropna().empty:
    df = run_fallback_training()
df = format_datetime_col(df)

# Sidebar filters
st.sidebar.header("🔹Filters & Settings")
models = sorted(df["model"].dropna().unique().tolist())
selected_models = st.sidebar.multiselect("Select Model(s)", options=models, default=models if models else [])

datasets = df["dataset"].dropna().unique().tolist() if "dataset" in df.columns else []
selected_datasets = st.sidebar.multiselect("Select Dataset(s)", options=datasets, default=datasets)

tasks = df["task"].dropna().unique().tolist() if "task" in df.columns else []
selected_tasks = st.sidebar.multiselect("Select Task(s)", options=tasks, default=tasks)

if "datetime" in df.columns and not df["datetime"].dropna().empty:
    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()
else:
    # fallback to today
    min_date = datetime.now().date()
    max_date = datetime.now().date()

date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# apply filters
filtered = df.copy()
if selected_models:
    filtered = filtered[filtered["model"].isin(selected_models)]
if selected_datasets:
    filtered = filtered[filtered["dataset"].isin(selected_datasets)]
if selected_tasks:
    filtered = filtered[filtered["task"].isin(selected_tasks)]

if "datetime" in filtered.columns and not filtered["datetime"].dropna().empty:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[(filtered["datetime"] >= start_dt) & (filtered["datetime"] <= end_dt)]
else:
    # ensure datetime exists for later plots
    filtered["datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=len(filtered)).to_pydatetime()

# Main layout: KPIs and plots
k1, k2, k3, k4 = st.columns(4)
mean_acc = filtered["accuracy"].mean() if not filtered["accuracy"].dropna().empty else np.nan
median_acc = filtered["accuracy"].median() if not filtered["accuracy"].dropna().empty else np.nan
mean_co2 = filtered["emissions_kg"].mean() if not filtered["emissions_kg"].dropna().empty else np.nan
total_co2 = filtered["emissions_kg"].sum() if not filtered["emissions_kg"].dropna().empty else np.nan

efficiency = (filtered["accuracy"] / filtered["emissions_kg"].replace(0, np.nan))
best_idx = efficiency.idxmax() if efficiency.notna().any() else None
best_run = filtered.loc[best_idx] if best_idx is not None else None

k1.metric("Mean accuracy", f"{mean_acc:.3f}" if not np.isnan(mean_acc) else "N/A")
k2.metric("Median accuracy", f"{median_acc:.3f}" if not np.isnan(median_acc) else "N/A")
k3.metric("Mean CO₂ (kg/run)", f"{mean_co2:.6f}" if not np.isnan(mean_co2) else "N/A")
k4.metric("Total CO₂ (kg)", f"{total_co2:.6f}" if not np.isnan(total_co2) else "N/A")

if best_run is not None:
    st.markdown(f"**Best efficiency run:** {best_run['model']} — "
                f"Acc {best_run['accuracy']:.2f}% / CO₂ {best_run['emissions_kg']:.4f} kg")

st.subheader("📊 Accuracy vs CO₂ (per run)")
if filtered.empty:
    st.info("No runs match filters. Run training or pull data via DVC to populate.")
else:
    # scatter plot: accuracy vs emissions
    scatter_fig = px.scatter(
        filtered,
        x="emissions_kg",
        y="accuracy",
        color="model",
        size="train_time_sec" if "train_time_sec" in filtered.columns else None,
        hover_data=["datetime", "train_time_sec", "log_loss", "n_train", "n_test", "notes"],
        labels={"emissions_kg": "CO₂ (kg)", "accuracy": "Accuracy"}
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(scatter_fig, use_container_width=True)
    with col2:
        model_summary = filtered.groupby("model")["accuracy"].mean().reset_index()
        bar_fig = px.bar(model_summary, x="model", y="accuracy", color="model",
                         title="Mean Accuracy per Model")
        st.plotly_chart(bar_fig, use_container_width=True)

    # time series: accuracy & CO2 over time (two panels)
    st.subheader("📈 Trends over time")
    fig_acc = px.line(filtered.sort_values("datetime"), x="datetime", y="accuracy", color="model", markers=True)
    fig_co2 = px.line(filtered.sort_values("datetime"), x="datetime", y="emissions_kg", color="model", markers=True)
    st.plotly_chart(fig_acc, use_container_width=True)
    st.plotly_chart(fig_co2, use_container_width=True)

    # Train time vs Energy plot
    st.subheader("⚡ Training Time vs Energy")
    if "train_time_sec" in filtered.columns and "energy_kwh" in filtered.columns:
        fig_eff = px.scatter(filtered, x="train_time_sec", y="energy_kwh", color="model",
                             hover_data=["datetime", "accuracy", "emissions_kg"],
                             labels={"train_time_sec": "Training Time (s)", "energy_kwh": "Energy Consumed (kWh)"},
                             title="Training Time vs Energy Usage"
                            )
        st.plotly_chart(fig_eff, use_container_width=True)
    else:
        st.info("No energy data available — run CodeCarbon-enabled training to populate this chart.")

    # Runs Table
    st.subheader("📋 Runs Table")
    show_cols = ["datetime", "model", "dataset", "task", "accuracy", "log_loss", "train_time_sec", "emissions_kg", "energy_kwh", "notes"]

    # Only keep columns that actually exist in the dataframe
    available_cols = [c for c in show_cols if c in filtered.columns]
    if available_cols:
        display_df = filtered[available_cols].sort_values("datetime", ascending=False, na_position="last").reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True)

        # Download buttons
        csv = display_df.to_csv(index=False).encode("utf-8")
        json_str = display_df.to_json(orient="records", date_format="iso")
        st.download_button("Download CSV", csv, file_name="filtered_metrics.csv", mime="text/csv")
        st.download_button("Download JSON", json_str, file_name="filtered_metrics.json", mime="application/json")
    else:
        st.info("No valid columns found to display runs table.")

# CO2 saved analysis
st.subheader("🌍 CO₂ Saved Analysis: Compare two models")
if len(models) < 2:
    st.info("Need at least two different models in the dataset to compute CO₂ savings.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        big_model = st.selectbox("Select larger/reference model (higher CO₂)", options=models, index=0)
    with col_b:
        small_model = st.selectbox("Select smaller model (compare to reference)", options=[m for m in models if m != big_model], index=0 if len(models) > 1 else 0)

    # compute mean emissions per model
    model_stats = filtered.groupby("model")["emissions_kg"].mean().dropna()
    if big_model in model_stats.index and small_model in model_stats.index:
        mean_big = model_stats.loc[big_model]
        mean_small = model_stats.loc[small_model]
        saved_per_run = mean_big - mean_small
        saved_pct = (saved_per_run / mean_big) * 100 if mean_big != 0 else np.nan

        st.metric("Mean CO₂ (kg) — reference", f"{mean_big:.6f}")
        st.metric("Mean CO₂ (kg) — comparison", f"{mean_small:.6f}")
        st.metric("Mean CO₂ saved (kg)", f"{saved_per_run:.6f}" if not np.isnan(saved_per_run) else "N/A")
        st.metric("Percent saved", f"{saved_pct:.2f}%" if not np.isnan(saved_pct) else "N/A")

        # simple bar chart visual
        compare_df = pd.DataFrame({
            "model": [big_model, small_model],
            "mean_emissions_kg": [mean_big, mean_small]
        })
        fig_cmp = px.bar(compare_df, x="model", y="mean_emissions_kg", text="mean_emissions_kg")
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.warning("Selected models don't have emissions data in the filtered set. Try widening the date range or selecting different models.")

st.markdown("---")
st.caption("Tip: If metrics are stale or missing, run training ('dvc repro') locally and push artifacts, then refresh this page.")
