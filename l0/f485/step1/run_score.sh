#!/bin/bash
# VES-2578 step 1 CPU scoring, after run_infer.sh: the frozen rules -> step1/results.json.
set -euo pipefail
cd "$(dirname "$0")"
${CPU_PY:-python} attempt4/score4.py
