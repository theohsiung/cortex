#!/bin/bash
source /home/theo/projects/cortex/.venv/bin/activate
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
uv run -m example 2>&1 | tee /home/theo/projects/cortex/logs/run_${TIMESTAMP}.log