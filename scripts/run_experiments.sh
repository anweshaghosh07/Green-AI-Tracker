#!/usr/bin/env bash
set -euo pipefail

ROOT="$(dirname "$0")/.."
cd "$ROOT"

# change these if your script uses different argument names
TRAIN_SCRIPT="python src/train_with_cc_ml.py"

# array of commands to run: model args, n_runs=1 each loop for seed control
declare -a JOBS=(
  # logreg 3 seeds
  "$TRAIN_SCRIPT --model logreg --n_runs 1 --seed 0"
  "$TRAIN_SCRIPT --model logreg --n_runs 1 --seed 1"
  "$TRAIN_SCRIPT --model logreg --n_runs 1 --seed 2"

  # random forest 3 runs with 200 estimators
  "$TRAIN_SCRIPT --model rf --n_runs 1 --seed 0 --n_estimators 200"
  "$TRAIN_SCRIPT --model rf --n_runs 1 --seed 1 --n_estimators 200"
  "$TRAIN_SCRIPT --model rf --n_runs 1 --seed 2 --n_estimators 200"

  # svm 2 runs
  "$TRAIN_SCRIPT --model svm --n_runs 1 --seed 0"
  "$TRAIN_SCRIPT --model svm --n_runs 1 --seed 1"

  # cnn on mnist 2 runs, 3 epochs
  "$TRAIN_SCRIPT --model cnn --cnn_dataset mnist --epochs 3 --n_runs 1 --seed 0"
  "$TRAIN_SCRIPT --model cnn --cnn_dataset mnist --epochs 3 --n_runs 1 --seed 1"
)

LOGDIR="./logs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

i=0
for cmd in "${JOBS[@]}"; do
  i=$((i+1))
  echo "=== RUN $i/${#JOBS[@]}: $cmd ==="
  # make per-run log so you can inspect details later
  logfile="$LOGDIR/run_${i}.log"
  # Run the command and tee output
  bash -c "$cmd" 2>&1 | tee "$logfile"
  echo "Finished run $i — log: $logfile"
  # small sleep to allow CodeCarbon/emissions.csv writer to flush
  sleep 3
done

echo "All runs complete. Logs: $LOGDIR"