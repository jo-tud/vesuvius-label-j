#!/bin/bash
# VES-2578 step 0, attempt 2 (CPU): G1 frame check, fresh draw, near window on the argmax (attempt2/README.md).
set -euo pipefail
cd "$(dirname "$0")"
${CPU_PY:-python} step0_frame.py --attempt 2
