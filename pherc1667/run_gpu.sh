#!/bin/bash
# VES-2321 GPU (gpu1 via vc-heavy-jobs), only after the R9 passes: known-artefact control P0, P1 on all six segments,
# then the six held-out folds (J0, J1, D0, D1 each, 4 parallel processes), budget guard 3.0 h.
set -euo pipefail
cd "$(dirname "$0")"
PY=${GPU_PY:-python}
for s in 0 1; do [ -f "${RUN_1667:-../data/run1667}/train_P$s.json" ] || $PY train.py --arm P --seed $s; done
$PY heldout/run_folds.py --budget-h 3.0
