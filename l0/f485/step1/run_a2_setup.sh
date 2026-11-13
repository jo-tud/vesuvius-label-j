#!/bin/bash
# VES-2578 step 1 attempt 2 run directory (VES-2741): $VES2578_RUN (default .../ves2578/step1a2) links the model-free
# step-1 artefacts (blocks, prep, replay tiles, sites) from step1/ and starts from the pilot's patch-128 P0 (same code,
# seed 0, 12 000 steps; copied, not retrained). Attempt 1's nets stay in step1/.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
S=$LABELJ_DATA/ves2578/step1
R=${VES2578_RUN:-$LABELJ_DATA/ves2578/step1a2}
mkdir -p "$R/heldout" "$R/attempt4"
for f in train176.npy train_ok.npy train_pos.npy train_sup.npy prep.json replay.npz points.npz xseg.json; do
  ln -sfn "$S/$f" "$R/$f"
done
ln -sfn "$S/attempt4/sites4.json" "$R/attempt4/sites4.json"
for src_dst in "net_P0p128.pt net_P0.pt" "out_P0p128_replay.npz out_P0_replay.npz" "train_P0p128.json train_P0.json" \
               "log_P0p128.txt log_P0.txt"; do
  set -- $src_dst
  [ -e "$R/$2" ] || cp -p "$S/$1" "$R/$2"
done
${CPU_PY:-python} -c "import json; d=json.load(open('$R/train_P0.json')); assert (d['arm'], d['seed'], d['steps'], d['patch']) == ('P', 0, 12000, 128), d; print('P0 from the pilot: fit AUC', round(d['fit_auc_shell'], 3))"
ls -la "$R"
