#!/bin/bash
# VES-3380 attempt 3 (gpu1, one unit, only after R9 PASS) = ../2026-11-07-ves3347-f485-jd-folds/run_gpu.sh with the
# one registered change: D0 is trained for 24 000 steps (was 12 000). Order: (0) the CPU check first (C1 scorer, C2
# hecate positive control, C4 J = F-545, synthetic rules and gates): check_ok false -> VOID, no training; (1) train
# D0 on fold w029 with F-485's heldout/train_fold.py, unchanged (arm D, seed 0, patch 128, batch 16, --steps 24000);
# (2) gate1 (fit AUC >= the w029 F-538 bar 0.6961); (3) only if gate1 passes, the one held-out D0 read on w029
# (infer3380.py), then score3380.py. Cap 6.0 GPU-h: training is killed at 5.4 h (then D0 is NOT-RUN, no read);
# inference gets the rest of the 6.0 h as its guard cap.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves3380/run}
export VES3380_OUT=${VES3380_OUT:-$LABELJ_DATA/ves3380/out}
P=${GPU_PY:-python}
mkdir -p "$VES3380_OUT"
./setup_data.sh > /dev/null
$P score3380.py --check > /dev/null
$P -c "import json, sys; sys.exit(0 if json.load(open('check.json'))['check_ok'] is True else 4)" \
  || { echo "CHECK FAIL (hecate PC / scorer / J reproduction): VOID, no training"; exit 4; }
TJ=$VES2578_RUN/heldout/train_D0_hw029.json
if [ ! -f "$TJ" ]; then
  rc=0
  timeout 5.4h $P ../f485/step1/heldout/train_fold.py --fold w029 --arm D --seed 0 \
    --steps 24000 --threads 6 2>&1 | tee -a "$VES3380_OUT/log_D0_hw029.txt" || rc=$?
  [ "$rc" = 0 ] || { echo "TRAIN NOT-RUN (rc $rc): D0 not trained inside the cap, no held-out read"; exit 3; }
fi
rc=0; $P score3380.py --gate1 || rc=$?
if [ "$rc" = 10 ]; then
  echo "GATE1 FAIL: D0 (24 000 steps) under the w029 fit bar; no held-out read (STOP-FIT)"
  $P score3380.py
  exit 0
fi
[ "$rc" = 0 ] || exit "$rc"
CAP=$($P -c "import json; print(round(6.0 - json.load(open('$TJ'))['seconds'] / 3600, 3))")
$P infer3380.py --workers 10 --cap-h "$CAP"
$P score3380.py
