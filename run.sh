#!/usr/bin/env bash
set -euo pipefail

# Accept arguments, fall back to defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "========================================"
echo "AdRevenue Forecasting Pipeline"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Model path:     $MODEL_PATH"
echo "Output path:    $OUTPUT_PATH"
echo "========================================"

# Resolve Python executable (prefer .venv, then python, then python3)
if [ -f ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "❌ Python is not available. Install Python 3 or ensure 'python'/'python3' is on PATH." >&2
  exit 1
fi

# Step 1: Generate features from input data
echo ""
echo "Step 1/2: Generating features..."
FEATURES_PATH="features.csv"
"$PYTHON_BIN" src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out "$FEATURES_PATH"

# Step 2: Load model and produce predictions
# Pass --data-dir explicitly so predict.py can find historical CSVs for bootstrap/OOD.
# This arg is forwarded through the features CSV's directory context.
echo ""
echo "Step 2/2: Generating predictions..."
"$PYTHON_BIN" src/predict.py \
    --features "$FEATURES_PATH" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --data-dir "$DATA_DIR"

echo ""
echo "========================================"
echo " Done! Predictions written to $OUTPUT_PATH"
echo "========================================"
