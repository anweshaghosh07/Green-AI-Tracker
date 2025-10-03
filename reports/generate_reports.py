#!/usr/bin/env python3
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from pathlib import Path

Path("reports").mkdir(exist_ok=True)
Path("docs/screenshots").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/baseline_metrics.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
# normalize accuracy if in 0..1
if df['accuracy'].max() <= 1.0:
    df['accuracy'] = df['accuracy'] * 100.0

# ensure numeric emissions
df['emissions_kg'] = pd.to_numeric(df['emissions_kg'], errors='coerce')

summary = df.groupby('model').agg(
    mean_accuracy=('accuracy','mean'),
    std_accuracy=('accuracy','std'),
    mean_emissions=('emissions_kg','mean'),
    runs=('run_id','count')
).reset_index()
summary['efficiency'] = summary['mean_accuracy'] / summary['mean_emissions']
summary.to_csv("reports/summary_by_model.csv", index=False)
print("Wrote reports/summary_by_model.csv")

# Scatter: accuracy vs co2
plt.figure(figsize=(8,6))
for m, g in df.groupby('model'):
    plt.scatter(g['emissions_kg'], g['accuracy'], label=m, alpha=0.8)
plt.xlabel("CO2 (kg)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs CO2 (per run)")
plt.legend()
plt.grid(True)
plt.savefig("reports/accuracy_vs_co2.png", dpi=150, bbox_inches='tight')
plt.close()
print("Wrote reports/accuracy_vs_co2.png")

# Emissions trend (daily)
if 'timestamp' in df.columns:
    t = df.groupby(df['timestamp'].dt.date).emissions_kg.sum().reset_index()
    plt.figure(figsize=(8,4))
    plt.plot(t['timestamp'], t['emissions_kg'], marker='o')
    plt.xlabel("Date")
    plt.ylabel("Total CO2 (kg)")
    plt.title("Total CO2 over time")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig("reports/emissions_trend.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Wrote reports/emissions_trend.png")

# Efficiency bar chart
plt.figure(figsize=(8,4))
summary_sorted = summary.sort_values('efficiency', ascending=False)
plt.bar(summary_sorted['model'], summary_sorted['efficiency'])
plt.ylabel("Efficiency (accuracy%/kg CO2)")
plt.title("Model Efficiency")
plt.xticks(rotation=45)
plt.savefig("reports/efficiency_by_model.png", dpi=150, bbox_inches='tight')
plt.close()
print("Wrote reports/efficiency_by_model.png")