#!/usr/bin/env bash
set -euo pipefail
echo "1) Install deps"
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt

echo "2) DVC pull"
dvc pull || echo "No dvc remote or no files tracked."

echo "3) Start MLflow UI in background"
mlflow ui --port 5000 &>/tmp/mlflow.log & echo $! > /tmp/mlflow.pid
sleep 2

echo "4) Run a minimal training (one fast run)"
python src/train_with_cc_ml.py

echo "5) Start Streamlit in background (or run interactively)"
streamlit run app.py &>/tmp/streamlit.log & echo $! > /tmp/streamlit.pid
sleep 5

echo "6) Smoke test http"
curl -s -I http://localhost:8501 | head -n 5

echo "7) Check outputs"
ls -la data || true
tail -n 5 data/baseline_metrics.csv || true

echo "8) Clean up background processes"
kill $(cat /tmp/mlflow.pid) || true
kill $(cat /tmp/streamlit.pid) || true