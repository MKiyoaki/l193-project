#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJ_ROOT="$SCRIPT_DIR/.."

ENV_NAME="${ENV_NAME:-sfc}"
export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"

if command -v conda &> /dev/null && conda info --envs | grep -q -w "$ENV_NAME"; then
    RUN_CMD="$RUN_CMD"
else
    RUN_CMD="python"
fi


cd $PROJ_ROOT

echo "Starting Global Evaluation for SAE Features (SFC)..."
$RUN_CMD experiments/eval_global.py \
    --method sfc \
    --experiment_name nlp

# echo "Starting Global Evaluation for Dense Neurons (Baseline)..."
# $RUN_CMD experiments/eval_global.py \
#     --method dense \
#     --experiment_name nlp
    
cd $SCRIPT_DIR