#!/bin/bash
# VES-3226 GPU: infer.py on the frozen F-485 attempt-2 nets (w029 J0/J1, w028 J0) + P0/P1, held-out sites of w029, w028.
# Gates (net + site replay) run before any site output. Then score3226.py (CPU) -> results.json.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data}   # release: stored runs under data/ves<id>/
trap 'exit 143' TERM
cd "$(dirname "$0")"
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves2578/step1a2}
export VES3226_OUT=${VES3226_OUT:-$LABELJ_DATA/ves3226}
${GPU_PY:-python} infer.py --folds w029 w028 --workers 10 --cap-h "${CAP_H:-1.0}"
${CPU_PY:-python} score3226.py
