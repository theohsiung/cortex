#!/usr/bin/env bash
# Run LLM-planning evaluation on cortex GAIA results.
#
# Usage:
#   ./scripts/eval/gaia/eval.sh [results.jsonl]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_PLANNING_DIR="/home/theo/projects/LLM-planning"
DEFAULT_INPUT="${SCRIPT_DIR}/cortex_gaia_results.jsonl"
INPUT="${1:-$DEFAULT_INPUT}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

OUTPUT_CSV="${INPUT%.jsonl}_per_sample.csv"
OUTPUT_SUMMARY="${INPUT%.jsonl}_summary.json"

echo "=== GAIA Evaluation ==="
echo "Input:   $INPUT"
echo "CSV:     $OUTPUT_CSV"
echo "Summary: $OUTPUT_SUMMARY"
echo ""

cd "$LLM_PLANNING_DIR"
python -m src.evaluation.runner \
    --input "$INPUT" \
    --output_csv "$OUTPUT_CSV" \
    --output_summary "$OUTPUT_SUMMARY"

echo ""
echo "=== Summary ==="
python -c "import json; d=json.load(open('$OUTPUT_SUMMARY')); print(json.dumps(d, indent=2))"
