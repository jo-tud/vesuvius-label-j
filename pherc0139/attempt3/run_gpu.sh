#!/bin/bash
# VES-2251 / VES-2241 attempt 3 GPU stage (gpu1): frame-shift outputs of the known-artefact control nets (P0, P1) and the
# attempt-1 D nets (reading), then arm J (P + +-6 vx frame jitter) x 2 seeds. Needs attempt3/prep3.py outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU_PY:-python}
for n in P0 P1 D0 D1; do "$GPU" attempt3/shift_out.py --name "$n"; done
for s in 0 1; do "$GPU" train.py --arm J --seed "$s" "$@"; done
