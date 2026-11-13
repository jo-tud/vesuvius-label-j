#!/bin/bash
# VES-2578 step 1 attempt 2 CPU scoring, after run_a2_infer.sh: the frozen rules (scorer 5b066d30) -> step1/results.json.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves2578/step1a2}
${CPU_PY:-python} attempt4/score4.py
