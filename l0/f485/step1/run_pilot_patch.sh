#!/bin/bash
# VES-2578 step 1 attempt-2 pilot (README "Step 1, attempt 2 pilot"): the known-artefact control P0 (all six segments,
# no held-out site read) at in-plane patch 64 and 128, 12 000 steps. Args = patches to train on this unit's GPU at once
# (default "64 128"; patch 64 alone fits gpu0, 128 needs gpu1). Reads only the training-patch fit AUC in
# DATA/train_P0p<patch>.json.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
PY=${GPU_PY:-python}
D=$LABELJ_DATA/ves2578/step1
PATCHES=("${@:-64 128}"); [ $# -eq 0 ] && PATCHES=(64 128)
for p in "${PATCHES[@]}"; do
  VES2578_PATCH=$p $PY train.py --arm P --seed 0 --tag p$p --threads 6 > "$D/log_P0p$p.txt" 2>&1 &
done
rc=0; for p in "${PATCHES[@]}"; do wait -n || rc=1; done
for p in "${PATCHES[@]}"; do ${CPU_PY:-python} -c "import json; d=json.load(open('$D/train_P0p$p.json')); print('p$p', d['patch'], round(d['fit_auc_shell'], 3), d['seconds'])" || true; done
exit $rc
