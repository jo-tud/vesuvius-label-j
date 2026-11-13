#!/bin/bash
# VES-3226 smoke (code path only, no result read): F-485's smoke data root (30-step nets, 4 label + 4 N1 old-draw
# sites per segment), fold w028 only, outputs to LABELJ_DATA/ves3226/smoke. infer.py, then score3226.py on w028 only.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
export VES2578_SMOKE=4 VES2578_RUN=$LABELJ_DATA/ves2578/smoke VES3226_OUT=$LABELJ_DATA/ves3226/smoke
mkdir -p "$VES3226_OUT"
${GPU_PY:-python} infer.py --folds w028 --workers 8 --cap-h 0.2
${CPU_PY:-python} - > "$VES3226_OUT/score_smoke.log" 2>&1 <<'PY'
import sys, json
from pathlib import Path
import score3226 as X
X.FOLDS = ('w028',)
X.RESULTS = Path(__import__('os').environ['LABELJ_DATA'] + '/ves3226/smoke/results_smoke.json')
sys.argv = ['score3226.py']
X.main()
PY
echo "score smoke exit $?"
