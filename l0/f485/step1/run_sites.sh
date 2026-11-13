#!/bin/bash
# VES-2578 step 1 CPU, only after the R9: per-point arrays and hecate's level-2 output for every step-0 site
# (attempt4/sites4.py -> points.npz + points.sha256, commit the sha file), then score4 --check (no fold-net output).
set -euo pipefail
cd "$(dirname "$0")"
PY=${CPU_PY:-python}
$PY attempt4/sites4.py --workers 10 --threads 10
$PY attempt4/score4.py --check
