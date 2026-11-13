#!/bin/bash
# VES-2578 step 1 smoke test (code path only, no result read): VES2578_SMOKE=4 (4 old-draw label + 4 N1 sites per
# segment), 2 blocks per segment, own data root. cpu: prep, sites4, xseg, score4 --check. gpu: P0/P1 + fold w028 at
# 30 steps, infer4 on w028, score4 (output to a log, not read for any result).
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
export VES2578_SMOKE=4 VES2578_RUN=$LABELJ_DATA/ves2578/smoke
mkdir -p "$VES2578_RUN"
case "${1:-}" in
  cpu) PY=${CPU_PY:-python}
       $PY prep.py --workers 8 --limit-blocks 2
       $PY attempt4/sites4.py --workers 8 --threads 4
       $PY xseg.py
       $PY attempt4/score4.py --check > "$VES2578_RUN/score4_check.log" 2>&1; echo "score4 --check exit $?" ;;
  gpu) PY=${GPU_PY:-python}
       $PY heldout/run_folds.py --steps 30 --folds w028 --budget-h 3
       $PY attempt4/infer4.py --folds w028 --workers 8
       ${CPU_PY:-python} attempt4/score4.py > "$VES2578_RUN/score4.log" 2>&1; echo "score4 exit $?" ;;
  *) echo "usage: $0 cpu|gpu"; exit 2 ;;
esac
