#!/bin/bash
# VES-3199 (F-538): inference only, ~5 min on one local card; then the readings.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves2578/step1a2}
export OUT=${OUT:-$LABELJ_DATA/ves3199}
P=${GPU_PY:-python}
$P score.py
$P make_results.py
