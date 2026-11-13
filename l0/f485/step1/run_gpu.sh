#!/bin/bash
# VES-2578 step 1 GPU training, only after the R9 and run_sites.sh: P0, P1, then the six folds x (J0, J1, D0, D1),
# 2 runs at a time, 12 000 steps; budget guard over training + inference reserve (BUDGET_H, default 7.5 GPU-h, VES-2644).
set -euo pipefail
cd "$(dirname "$0")"
${GPU_PY:-python} heldout/run_folds.py --steps 12000 --budget-h "${BUDGET_H:-7.5}"
