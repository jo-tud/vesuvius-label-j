#!/bin/bash
# VES-2369 step-1 prep, GPU part: ink_canonical_2um on every rendered tile (waits for the renders if needed).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 INKCHECK_ALLOW_LOCAL_GPU=1
"${GPU_PY:-python}" -I prep1.py infer
