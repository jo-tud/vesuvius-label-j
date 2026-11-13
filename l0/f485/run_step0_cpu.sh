#!/bin/bash
# VES-2578 step 0, CPU (one unit): G1 + G2 frame check, then G3 + S1 level-0 labels and scaled site draws.
set -euo pipefail
cd "$(dirname "$0")"
PY=${CPU_PY:-python}
$PY step0_frame.py
$PY step0_labels.py
