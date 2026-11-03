#!/bin/bash
# VES-2241 GPU stage: 2 arms x 2 seeds, sequential, one GPU. Usage: run_gpu.sh [extra train.py args]
set -euo pipefail
cd "$(dirname "$0")"
GPU=${GPU_PY:-python}
for s in 0 1; do for a in S D; do "$GPU" train.py --arm "$a" --seed "$s" "$@"; done; done
