#!/bin/bash
# VES-2321 CPU, after run_prep.sh: xseg reading, then attempt-4 sites (new draws, power rule, per-point arrays and
# hecate's 3D forward output for old + new sites -> points_old.npz + points_old.sha256, points4.npz). Only after the R9.
set -euo pipefail
cd "$(dirname "$0")"
PY=${CPU_PY:-python}
$PY xseg.py
$PY attempt4/sites4.py --workers 10 --threads 10
