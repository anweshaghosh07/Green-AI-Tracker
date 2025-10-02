#!/usr/bin/env python3
"""
Validate exported filtered metrics against baseline.
Usage:
 python scripts/validate_export.py --export tests/exports/filtered_metrics.csv \
    --models logreg rf --start 2024-01-01 --end 2025-10-01
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--export", required=True, help="Path to exported CSV/JSON")
parser.add_argument("--baseline_csv", default="data/baseline_metrics.csv", help="Path to baseline_metrics.csv")
parser.add_argument("--models", nargs="+", help="Filter by models (optional)")
args = parser.parse_args()

# Load data
export_df = pd.read_csv(args.export)
base_df = pd.read_csv(args.baseline_csv)

# ---- Ensure 'date' column in baseline ----
if "date" not in base_df.columns:
    if "human_timestamp" in base_df.columns:
        base_df["date"] = pd.to_datetime(base_df["human_timestamp"], errors="coerce")
    elif "timestamp" in base_df.columns:
        base_df["date"] = pd.to_datetime(base_df["timestamp"], unit="s", errors="coerce")
    else:
        print("⚠️ WARNING: baseline has no date/timestamp columns.")

# ---- Ensure 'date' column in export ----
if "date" not in export_df.columns:
    if "human_timestamp" in export_df.columns:
        export_df = export_df.rename(columns={"human_timestamp": "date"})
    elif "datetime" in export_df.columns:
        export_df = export_df.rename(columns={"datetime": "date"})
    else:
        print("⚠️ WARNING: export has no date column; skipping date validation.")

# ---- Ensure run_id exists ----
if "run_id" not in export_df.columns:
    print("⚠️ WARNING: export missing 'run_id'. Validation will proceed without it.")

# ---- Normalize model names (case-insensitive) ----
if "model" in base_df.columns:
    base_df["model"] = base_df["model"].astype(str).str.lower()
if "model" in export_df.columns:
    export_df["model"] = export_df["model"].astype(str).str.lower()

# If user supplied --models, normalize them too
if args.models:
    models = [m.lower() for m in args.models]
    base_df = base_df[base_df["model"].isin(models)]
    export_df = export_df[export_df["model"].isin(models)]

print(f"Filtered baseline rows: {len(base_df)} Export rows: {len(export_df)}")

# ---- Compare columns ----
missing_cols = set(base_df.columns) - set(export_df.columns)
extra_cols = set(export_df.columns) - set(base_df.columns)

if missing_cols:
    print(f"⚠️ Export missing cols: {missing_cols}")
if extra_cols:
    print(f"ℹ️ Export has extra cols: {extra_cols}")

# ---- Compare counts ----
if len(export_df) == 0:
    print("❌ ERROR: No rows in export after filtering")
else:
    print("✅ Export validation completed (with warnings above if any).")