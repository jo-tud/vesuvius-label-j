#!/bin/bash
# VES-2578 step 1 attempt 2 GPU inference, after every run_a2_gpu.sh unit: replay gates, then every trained fold's
# sites (attempt4/infer4.py, unchanged; the nets are fully convolutional, so the 384 px tiles are as in attempt 1).
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves2578/step1a2}
${GPU_PY:-python} attempt4/infer4.py --workers 10
