#!/bin/bash
# VES-2578 step 1 GPU inference, after run_gpu.sh: replay gates, then every trained fold's sites (tiles rendered on
# the fly by 10 CPU workers), J/D fold nets + P0/P1.
set -euo pipefail
cd "$(dirname "$0")"
${GPU_PY:-python} attempt4/infer4.py --workers 10
