#!/bin/bash
# VES-2578 step 1 attempt 2 GPU training (VES-2741), only after the delta R9 and run_a2_setup.sh: P1, then the six folds
# x (J0, J1, D0, D1), one run at a time, patch 128, 12 000 steps, prefetched batches. One unit <= 6 h: the runner
# stops at --unit-h (UNIT CAP) and the next unit resumes; budget guard 60 GPU-h over all units (BUDGET STOP).
# FOLDS (optional, comma list) = the runner's --folds order; from unit 2: w028 last (its fold is VOID, README 11-03).
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves2578/step1a2}
${GPU_PY:-python} heldout/run_folds.py --steps 12000 --budget-h "${BUDGET_H:-60}" --unit-h "${UNIT_H:-5.75}" ${FOLDS:+--folds "$FOLDS"}
