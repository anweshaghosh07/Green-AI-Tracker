#!/usr/bin/env python3
import pandas as pd
from pprint import pprint

# Read CSV
df = pd.read_csv("data/baseline_metrics.csv")

# parse timestamp if needed
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # timestamp is in seconds
elif 'human_timestamp' in df.columns:
    df['human_timestamp'] = pd.to_datetime(df['human_timestamp'])

# Function to compute stats
def stats_for(df):
    df2 = df.copy()
    if 'accuracy' in df2.columns and df2['accuracy'].max() <= 1.0:
        df2['accuracy'] = df2['accuracy'] * 100
    res = {
        'mean_accuracy': df2['accuracy'].mean() if 'accuracy' in df2.columns else None,
        'median_accuracy': df2['accuracy'].median() if 'accuracy' in df2.columns else None,
        'mean_emissions_kg': pd.to_numeric(df2['emissions_kg'], errors='coerce').mean() if 'emissions_kg' in df2.columns else None,
        'total_runs': len(df2)
    }
    return res

# Overall stats
print("Overall stats:")
pprint(stats_for(df))

# Per-model stats
print("\nPer-model stats:")
if 'model' in df.columns:
    pprint(df.groupby('model').apply(stats_for).to_dict())
else:
    print("No 'model' column in CSV.")