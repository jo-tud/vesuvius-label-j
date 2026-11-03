#!/bin/bash
# VES-2321 attempt 2 scoring: attempt-1 infer4 + score4 unchanged on run_a2/; score4 writes attempt4/results.json in
# the code folder, so the attempt-1 file is kept aside and restored, and the attempt-2 result goes to attempt2/.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_1667=${RUN_1667:-$(cd .. && pwd)/data/run1667}
PY=${CPU_PY:-python}
# attempt4/results.json (the attempt-1 result) is not in the release tree: keep it aside only if it exists
KEEP=""
if [ -f attempt4/results.json ]; then KEEP=$(mktemp); cp attempt4/results.json "$KEEP"; fi
$PY attempt4/infer4.py --threads 12
$PY attempt4/score4.py
mv attempt4/results.json attempt2/results.json
if [ -n "$KEEP" ]; then cp "$KEEP" attempt4/results.json; fi
