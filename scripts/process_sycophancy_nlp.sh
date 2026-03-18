#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJ_ROOT="$SCRIPT_DIR/.."


ENV_NAME="sfc"
export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"

cd $PROJ_ROOT
conda run -n $ENV_NAME --no-capture-output python tools/process_dataset.py \
    --dataset "EleutherAI/sycophancy" \
    --subset "sycophancy_on_nlp_survey" \
    --output "sycophancy_nlp.jsonl"
    
cd $SCRIPT_DIR