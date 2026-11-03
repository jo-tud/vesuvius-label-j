#!/bin/bash
# VES-2321 CPU, after run_gpu.sh: new-site outputs of every fold net (replay first), then the scorer.
set -euo pipefail
cd "$(dirname "$0")"
PY=${CPU_PY:-python}
$PY attempt4/infer4.py --threads 12
$PY attempt4/score4.py
