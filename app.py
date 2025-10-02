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
from datetime import timezone, datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt

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
        cc_files = list(EMISSIONS_DIR.glob("*.csv"))
        if cc_files:
            try:
                df_cc = pd.read_csv(cc_files[-1])
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

# ---------- INSIGHTS HELPERS ----------
try:
    from fpdf2 import FPDF
    _HAS_FPDF = True
    _FPDF_VERSION = getattr(FPDF,"__version__","unknown")
except Exception:
    _HAS_FPDF = False
    _FPDF_VERSION = None

@st.cache_data
def _normalize_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """If accuracy values are in 0..1 convert to percent (0..100)."""
    df = df.copy()
    if "accuracy" not in df.columns or df["accuracy"].dropna().empty:
        return df
    # if max <= 1 assume fraction
    if df["accuracy"].max() <= 1.0:
        df["accuracy"] = df["accuracy"] * 100.0
    return df

@st.cache_data
def aggregate_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-model aggregated stats (means, std, run counts)."""
    df = _normalize_accuracy(df)
    group = df.groupby("model", dropna=True)
    stats = group.agg(
        mean_accuracy = pd.NamedAgg(column="accuracy", aggfunc="mean"),
        median_accuracy = pd.NamedAgg(column="accuracy", aggfunc="median"),
        mean_emissions = pd.NamedAgg(column="emissions_kg", aggfunc="mean"),
        std_emissions = pd.NamedAgg(column="emissions_kg", aggfunc="std"),
        runs = pd.NamedAgg(column="model", aggfunc="count")
    ).reset_index()
    # Efficiency: accuracy per kg CO2 (higher = better), protect divide-by-zero
    stats["efficiency"] = stats.apply(lambda r: (r["mean_accuracy"] / r["mean_emissions"]) if r["mean_emissions"] and r["mean_emissions"] > 0 else float("nan"), axis=1)
    return stats

@st.cache_data
def pareto_frontier(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Return non-dominated models (maximize accuracy, minimize emissions).
    A model is dominated if there exists another model with >= accuracy and <= emissions
    and strictly better in at least one criterion.
    """
    if stats.empty:
        return stats
    records = stats.to_dict("records")
    nondom = []
    for a in records:
        dominated = False
        for b in records:
            if (b["mean_accuracy"] >= a["mean_accuracy"] and b["mean_emissions"] <= a["mean_emissions"]) and (b["mean_accuracy"] > a["mean_accuracy"] or b["mean_emissions"] < a["mean_emissions"]):
                dominated = True
                break
        if not dominated:
            nondom.append(a)
    return pd.DataFrame(nondom)

def compute_savings(stats: pd.DataFrame, from_model: str, to_model: str):
    """
    Compute mean CO2 savings and relative % change in accuracy if you switch from -> to.
    Returns dict: {mean_from, mean_to, saved_kg, saved_pct, acc_from, acc_to, acc_delta_pct}
    """
    if from_model not in stats["model"].values or to_model not in stats["model"].values:
        return None
    a = stats[stats["model"]==from_model].iloc[0]
    b = stats[stats["model"]==to_model].iloc[0]
    mean_from = a["mean_emissions"]
    mean_to = b["mean_emissions"]
    saved_kg = mean_from - mean_to if (mean_from is not None and mean_to is not None) else None
    saved_pct = (saved_kg / mean_from * 100.0) if mean_from and mean_from != 0 else None
    acc_from = a["mean_accuracy"]
    acc_to = b["mean_accuracy"]
    acc_delta_pct = (acc_to - acc_from)  # positive means to_model is more accurate
    return {
        "mean_from": mean_from,
        "mean_to": mean_to,
        "saved_kg": saved_kg,
        "saved_pct": saved_pct,
        "acc_from": acc_from,
        "acc_to": acc_to,
        "acc_delta_pct": acc_delta_pct
    }

def generate_insights_markdown(date_str: str, stats: pd.DataFrame, pareto_df: pd.DataFrame, recommendations: list, comparison_text: str) -> bytes:
    """Create a markdown report (returned as bytes)"""
    md = []
    md.append(f"🌍 Green-AI Insights — {date_str}\n")
    md.append("📋Per-model summary (mean accuracy and mean CO₂)\n")
    if not stats.empty:
        try:
            md.append(stats[["model","mean_accuracy","mean_emissions","efficiency","runs"]].to_markdown(index=False))
        except Exception:
            # fallback plain csv-style
            md.append(stats[["model","mean_accuracy","mean_emissions","efficiency","runs"]].to_csv(index=False))
    else:
        md.append("_No stats available_\n")
    md.append("\n Pareto frontier (recommended trade-offs)\n")
    if not pareto_df.empty:
        for _, r in pareto_df.iterrows():
            md.append(f"- **{r['model']}** — accuracy {r['mean_accuracy']:.2f}%, mean CO₂ {r['mean_emissions']:.6f} kg, efficiency {r['efficiency']:.2f} (%/kg)")
    else:
        md.append("_No pareto models found_\n")
    md.append("\n Automated recommendations\n")
    for rec in recommendations:
        md.append(f"- {rec}")
    md.append("\n Example comparison\n")
    md.append(comparison_text or "No comparisons available.")
    return ("\n\n".join(md)).encode("utf-8")

def markdown_to_pdf_bytes(md_bytes: bytes) -> bytes:
    """Render a simple text-like PDF from markdown using fpdf/fpdf2."""
    if not _HAS_FPDF:
        raise RuntimeError("fpdf not available")
    text = md_bytes.decode("utf-8")
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in text.splitlines():
        # FPDF's multi_cell handles wrapping
        pdf.multi_cell(0, 6, line)
    # fpdf2 supports output(dest="S") which returns str/bytes
    try:
        pdf_bytes = pdf.output(dest="S").encoder("latin1")
    except TypeError:
        pdf_bytes = pdf.output(dest="S")
    return pdf_bytes

# -------- UI --------
st.title("Green-AI🌱 Usage Tracker")

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

if st.sidebar.button("Refresh Data (clear cache )"):
    st.cache_data.clear()
    st.experimental_rerun()

# apply filters
filtered = df.copy()
# --- ensure datetime column always exists ---
if "datetime" in filtered.columns:
    filtered["datetime"] = pd.to_datetime(filtered["datetime"], errors="coerce").dt.tz_localize(None)
elif "timestamp" in filtered.columns:
    filtered["datetime"] = pd.to_datetime(filtered["timestamp"], errors="coerce").dt.tz_localize(None)
else:
    # fallback: create synthetic dates if missing
    filtered["datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=len(filtered)).to_pydatetime()

# apply user-selected filters
if selected_models:
    filtered = filtered[filtered["model"].isin(selected_models)]
if selected_datasets:
    filtered = filtered[filtered["dataset"].isin(selected_datasets)]
if selected_tasks:
    filtered = filtered[filtered["task"].isin(selected_tasks)]

# apply date range filter if available
if "datetime" in filtered.columns and not filtered["datetime"].dropna().empty:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[(filtered["datetime"] >= start_dt) & (filtered["datetime"] <= end_dt)]
else:
    # fallback again, just to be safe
    filtered["datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=len(filtered)).to_pydatetime()

# ----- Add a view selector so user can switch between Dashboard & Insights -----
view = st.sidebar.radio("🔹View", ["Dashboard", "Insights"], index=0)

if view == "Insights":
    # Build insights using the already-filtered DataFrame (respects sidebar filters)
    stats = aggregate_model_stats(filtered)
    pareto = pareto_frontier(stats)

    st.header("🔎 Insights — Best trade-offs & recommendations")
    st.markdown("Below we compute per-model averages and find the Pareto frontier (maximize accuracy, minimize CO₂).")

    # Show aggregated table
    if stats.empty:
        st.info("No data available for insights. Ensure `baseline_metrics.csv` has runs with emissions_kg populated.")
        st.stop()
    st.subheader("Per-model summary")
    st.dataframe(stats.sort_values("mean_accuracy", ascending=False).reset_index(drop=True), use_container_width=True)

    # Pareto
    st.subheader("Pareto frontier — best trade-offs")
    if not pareto.empty:
        st.table(pareto.sort_values(["mean_emissions","mean_accuracy"], ascending=[True, False]).reset_index(drop=True))
    else:
        st.info("Pareto frontier could not be computed (insufficient data).")

    # Recommend top-by-efficiency (accuracy per CO2)
    top_eff = stats.sort_values("efficiency", ascending=False).head(3)
    st.subheader("🎯Top models by efficiency (accuracy per kg CO₂)")
    st.table(top_eff[["model","mean_accuracy","mean_emissions","efficiency","runs"]])

    # Automated pairwise comparison: recommend swaps that save CO2 with small accuracy loss
    st.subheader("💡Actionable recommendations")
    recs = []
    # Conservative threshold: allow up to 1.0 percentage point accuracy drop
    max_allowed_acc_drop_pct = st.slider("Max allowed accuracy drop (%) when recommending a lower-CO₂ model", 0.0, 5.0, 1.0, 0.1)
    # Evaluate all model pairs (from higher emissions to lower)
    for a in stats.itertuples():
        for b in stats.itertuples():
            if a.model == b.model:
                continue
            # only consider switching from a -> b if b has lower emissions
            if b.mean_emissions is None or a.mean_emissions is None:
                continue
            if b.mean_emissions < a.mean_emissions:
                savings = compute_savings(stats, a.model, b.model)
                if savings is None:
                    continue
                # If accuracy loss within threshold (negative acc_delta_pct means drop)
                acc_drop = savings["acc_delta_pct"]
                # acc_drop is acc_to - acc_from; we want acceptable negative drop >= -max_allowed_acc_drop_pct
                if acc_drop >= -max_allowed_acc_drop_pct:
                    pct_str = f"{savings['saved_pct']:.2f}%" if savings['saved_pct'] is not None else "N/A"
                    acc_change = f"{savings['acc_delta_pct']:+.2f}%"
                    recs.append(f"Replace **{a.model}** (mean CO₂ {a.mean_emissions:.6f} kg) with **{b.model}** (mean CO₂ {b.mean_emissions:.6f} kg) → saves {pct_str} CO₂; accuracy change: {acc_change}.")

    if recs:
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.info("No safe swaps found under the current accuracy-drop threshold.")

    # Example: interactive compare two models (reuse earlier UI)
    st.subheader("📋Compare two specific models")
    m_from = st.selectbox("Model (from)", options=stats["model"].tolist(), index=0)
    m_to = st.selectbox("Model (to)", options=[m for m in stats["model"].tolist() if m != m_from], index=0)
    cmp = compute_savings(stats, m_from, m_to)
    if cmp:
        st.metric(f"Mean CO₂ (kg) - {m_from}", f"{cmp['mean_from']:.6f} kg")
        st.metric(f"Mean CO₂ (kg) - {m_to}", f"{cmp['mean_to']:.6f} kg")
        if cmp["saved_kg"] is not None:
            st.metric(f"CO₂ Saved per run (switch {m_from} → {m_to})", f"{cmp['saved_kg']:.6f} kg")
            st.metric(f"CO₂ Saved (%)", f"{cmp['saved_pct']:.2f}%")
        st.write(
            f"Accuracy change if you switch from **{m_from}** → **{m_to}**: "
            f"{cmp['acc_delta_pct']:+.2f} percentage points."
            )

    # Export insights to Markdown and (optionally) PDF
    date_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
    comparison_text = (
        f"Example comparison: {m_from} -> {m_to} saved {cmp['saved_pct']:.2f}% CO₂ (if data present)"
        if cmp and cmp.get("saved_pct") is not None else ""
   )
    md_bytes = generate_insights_markdown(date_str, stats, pareto, recs, comparison_text)

    st.download_button("📥Download insights (Markdown)", md_bytes, file_name=f"insights_{datetime.utcnow().date()}.md", mime="text/markdown")

    if _HAS_FPDF:
        try:
            pdf_bytes = markdown_to_pdf_bytes(md_bytes)
            st.download_button("📥Download insights (PDF)", pdf_bytes, file_name=f"insights_{datetime.utcnow().date()}.pdf", mime="application/pdf")
        except Exception as e:
            st.warning("Could not create PDF: " + str(e))
    # stop here so the Dashboard view doesn't render below
    st.stop()

# Main layout: KPIs and plots
k1, k2, k3, k4 = st.columns(4)
mean_acc = filtered["accuracy"].mean() if not filtered["accuracy"].dropna().empty else np.nan
median_acc = filtered["accuracy"].median() if not filtered["accuracy"].dropna().empty else np.nan
mean_co2 = filtered["emissions_kg"].mean() if not filtered["emissions_kg"].dropna().empty else np.nan
total_co2 = filtered["emissions_kg"].sum() if not filtered["emissions_kg"].dropna().empty else np.nan

efficiency = (filtered["accuracy"] / filtered["emissions_kg"].replace(0, np.nan))
best_idx = efficiency.idxmax() if efficiency.notna().any() else None
best_run = filtered.loc[best_idx] if best_idx is not None else None

k1.metric("Mean accuracy", f"{mean_acc:.2f}" if not np.isnan(mean_acc) else "N/A")
k2.metric("Median accuracy", f"{median_acc:.3f}" if not np.isnan(median_acc) else "N/A")
k3.metric("Mean CO₂ (kg/run)", f"{mean_co2:.6f}" if not np.isnan(mean_co2) else "N/A")
k4.metric("Total CO₂ (kg)", f"{total_co2:.6f}" if not np.isnan(total_co2) else "N/A")

if best_run is not None:
    st.markdown(f"🏆Best Efficiency Run: {best_run['model']} — "
                f"Acc {best_run['accuracy']:.2f}% / CO₂ {best_run['emissions_kg']:.4f} kg")

st.subheader("📊 Accuracy vs CO₂ (per run)")
if filtered.empty:
    st.info("No runs match filters. Run training or pull data via DVC to populate.")
else:
    # scatter plot: accuracy vs emissions
    scatter = alt.Chart(filtered).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X('emissions_kg:Q', title='CO₂ (kg)'),
        y=alt.Y('accuracy:Q', title='Accuracy (%)'),
        color=alt.Color('model:N', title='Model', scale=alt.Scale(scheme='tableau10')),
        tooltip=[
            alt.Tooltip('run_id:N', title='Run ID'),
            alt.Tooltip('model:N'),
            alt.Tooltip('dataset:N'),
            alt.Tooltip('task:N'),
            alt.Tooltip('accuracy:Q', format='.2f'),
            alt.Tooltip('emissions_kg:Q', format='.6f'),
            alt.Tooltip('date:T')
        ]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1, st.spinner("Building mean accuracy per model chart…"):
        model_summary = filtered.groupby("model")["accuracy"].mean().reset_index()
        bar_fig = alt.Chart(model_summary).mark_bar().encode(
            x=alt.X('model:N', title='Model'),
            y=alt.Y('accuracy:Q', title='Mean Accuracy (%)'),
            color=alt.Color('model:N', legend=None),
            tooltip=[alt.Tooltip('model:N'), alt.Tooltip('accuracy:Q', format='.2f')]
        ).interactive()
        st.altair_chart(bar_fig, use_container_width=True)
    
    # time series: accuracy & CO2 over time (two panels)
    st.subheader("📈 Trends over time")
    if not filtered.empty:
        # Accuracy trend
        acc_trend = alt.Chart(filtered.sort_values("datetime")).mark_line(point=True).encode(
            x=alt.X('datetime:T', title='Date'),
            y=alt.Y('accuracy:Q', title='Accuracy (%)'),
            color=alt.Color('model:N', title='Model', scale=alt.Scale(scheme='tableau10')),
            tooltip=['datetime:T', 'model:N', 'accuracy:Q', 'emissions_kg:Q']
        ).interactive()
    
        # Emissions trend
        co2_trend = alt.Chart(filtered.sort_values("datetime")).mark_line(point=True).encode(
            x=alt.X('datetime:T', title='Date'),
            y=alt.Y('emissions_kg:Q', title='CO₂ (kg)'),
            color=alt.Color('model:N', title='Model', scale=alt.Scale(scheme='tableau10')),
            tooltip=['datetime:T', 'model:N', 'accuracy:Q', 'emissions_kg:Q']
        ).interactive()
        
        st.altair_chart(acc_trend, use_container_width=True)
        st.altair_chart(co2_trend, use_container_width=True)
    else:
        st.info("No data available for trends over time.")

    # Train time vs Energy plot
    st.subheader("⚡ Training Time vs Energy")
    if "train_time_sec" in filtered.columns and "energy_kwh" in filtered.columns:
        eff_chart = alt.Chart(filtered).mark_circle(size=80, opacity=0.7).encode(
            x=alt.X("train_time_sec:Q", title="Training Time (s)"),
            y=alt.Y("energy_kwh:Q", title="Energy Consumed (kWh)"),
            color=alt.Color("model:N", title="Model", scale=alt.Scale(scheme='tableau10')),
            tooltip=[
                alt.Tooltip("datetime:T"),
                alt.Tooltip("model:N"),
                alt.Tooltip("accuracy:Q", format=".2f"),
                alt.Tooltip("emissions_kg:Q", format=".6f")
            ]
        ).interactive()
        st.altair_chart(eff_chart, use_container_width=True)
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