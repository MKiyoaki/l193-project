#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJ_ROOT="$SCRIPT_DIR/.."

ENV_NAME="sfc"
export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"

cd $PROJ_ROOT

echo "Starting Global Evaluation for SAE Features (SFC)..."
conda run -n $ENV_NAME --no-capture-output python experiments/eval_global.py \
    --method sfc \
    --experiment_name political

echo "Starting Global Evaluation for Dense Neurons (Baseline)..."
conda run -n $ENV_NAME --no-capture-output python experiments/eval_global.py \
    --method dense \
    --experiment_name political
    
cd $SCRIPT_DIR