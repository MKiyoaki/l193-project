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
$RUN_CMD tools/generate_neuropedia.py \
    --file effects/extracted_dense_features_nlp.json \
    --top_k 15
    
cd $SCRIPT_DIR