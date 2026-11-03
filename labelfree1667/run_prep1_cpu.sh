#!/bin/bash
# VES-2369 step-1 prep, CPU part: candidates (CT peak rule) + level-0 renders. Private 15 GiB chunk cache.
set -euo pipefail
cd "$(dirname "$0")"
P=${CPU_PY:-python}
export OMP_NUM_THREADS=8 INKCHECK_CACHE=${INKCHECK_CACHE:-$(cd .. && pwd)/data/labelfree1667/chunkcache} INKCHECK_CACHE_MAX_GB=15
$P -I prep1.py cands
$P -I prep1.py render
