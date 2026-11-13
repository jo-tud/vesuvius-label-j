#!/bin/bash
# VES-2578 step 1 CPU prep, model-free (allowed before the R9, as F-418's run_prep.sh): training blocks on 176 planes
# + replay tiles (prep.py), then the cross-segment reading (xseg.py). ~30 GB of blocks in $VES2578_RUN.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
PY=${CPU_PY:-python}
mkdir -p "${VES2578_RUN:-$LABELJ_DATA/ves2578/step1}"
$PY prep.py --workers 10
$PY xseg.py
