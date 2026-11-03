#!/bin/bash
# VES-2321 CPU prep, model-free: F-401 prep.py (32-plane site tiles + training blocks) then attempt3/prep3.py (44 planes).
set -euo pipefail
cd "$(dirname "$0")"
PY=${CPU_PY:-python}
mkdir -p "${RUN_1667:-../data/run1667}"
$PY prep.py --workers 10
$PY attempt3/prep3.py --workers 10
