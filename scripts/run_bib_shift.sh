#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJ_ROOT="$SCRIPT_DIR/.."


ENV_NAME="sfc"
export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"

cd $PROJ_ROOT
conda run -n $ENV_NAME --no-capture-output python experiments/run_bib_shift.py
    
cd $SCRIPT_DIR