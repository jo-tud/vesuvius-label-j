#!/bin/bash
# VES-3663 (F-567 control) one GPU unit on gpu1: ./unit.sh T (train D1_hw029: arm D, seed 1, 24 000 steps, F-485's
# heldout/train_fold.py unchanged) or ./unit.sh R (the one held-out w029 read of D12s0 = F-560's D0_hw029 and D24s1,
# then the scorer). R is queued only after the R9 PASS; no unit is cancelled on a result. Every unit first re-runs the
# CPU check (controls, frozen J0 / J1 / D24s0 = F-545 / F-567, synthetic rules): check_ok not true -> exit 4, nothing
# trained or read. Budget: the task ledger (OUT/ledger.jsonl + a GPU smoke) may not pass 7.0 GPU-h: a unit starts only
# if ledger + its expected cost <= 7.0, and is killed at min(stage cap, 7.0 - ledger - 0.05). T killed -> D24s1
# NOT-RUN (the R unit then reads D12s0 alone); R killed at its cap -> infer.json not_run (NOT-RUN); any other R
# failure -> VOID.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
STAGE=$1
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves3663/run}
export VES3663_OUT=${VES3663_OUT:-$LABELJ_DATA/ves3663/out}
P=${GPU_PY:-python}
TF=../f485/step1/heldout/train_fold.py
mkdir -p "$VES3663_OUT"
case "$STAGE" in
  T) EST=4.4; CAP=5.0 ;;     # F-567 D0 24 000 steps: 15 025 s (w029), 14 543-15 527 s (step 2)
  R) EST=1.3; CAP=2.0 ;;     # F-567 D0 w029 read 4 147 s (render bound; 2 nets share the render)
  *) echo "stage T|R"; exit 2 ;;
esac
T0=$(date +%s)
log_unit() {
  local rc=$?
  $P -c "import json,sys,time; t0=int(sys.argv[1]); open(sys.argv[2],'a').write(json.dumps(dict(stage=sys.argv[3], start=t0, end=int(time.time()), seconds=int(time.time())-t0, rc=int(sys.argv[4])))+'\n')" \
    "$T0" "$VES3663_OUT/ledger.jsonl" "$STAGE" "$rc"
}
trap log_unit EXIT
trap 'exit 143' TERM          # a SIGTERM (systemctl stop) is ledgered with rc 143, not 0 (R9 VES-3668)
mark_not_run() {
  $P -c "import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; sys.exit(0) if d.get('done') is True else None; d.update(fold='w029', done=False, not_run=True, stop=sys.argv[2]); p.write_text(json.dumps(d, indent=1))" \
    "$VES3663_OUT/infer.json" "$1"
}
./setup_data.sh > /dev/null
$P score3663.py --check > "$VES3663_OUT/check_${STAGE}.log" 2>&1
$P -c "import json, sys; sys.exit(0 if json.load(open('check.json'))['check_ok'] is True else 4)" \
  || { echo "CHECK FAIL (controls / frozen outputs / rules): nothing trained or read"; exit 4; }
USED=$($P -c "import score3663 as S; print(S.ledger()['gpu_h'])")
USED=$($P -c "print(round($USED + ($(date +%s) - $T0) / 3600, 4))")
LEFT=$($P -c "print(round(7.0 - $USED, 3))")
if ! $P -c "import sys; sys.exit(0 if $USED + $EST <= 7.0 else 1)"; then
  echo "BUDGET GUARD: ledger $USED GPU-h + $EST > 7.0: $STAGE NOT-RUN"
  [ "$STAGE" = R ] && mark_not_run "budget guard: ledger $USED + $EST > 7.0" && $P score3663.py
  exit 0
fi
KILL=$($P -c "print(round(min($CAP, $LEFT - 0.05), 4))")
echo "$STAGE: ledger $USED GPU-h, kill after $KILL h"
case "$STAGE" in
  T)
    [ -f "$VES2578_RUN/heldout/train_D1_hw029.json" ] && { echo "D1_hw029 already trained"; exit 0; }
    rc=0
    timeout "${KILL}h" $P "$TF" --fold w029 --arm D --seed 1 --steps 24000 --threads 6 2>&1 \
      | tee -a "$VES3663_OUT/log_D1_hw029.txt" || rc=$?
    [ "$rc" = 0 ] || { echo "TRAIN NOT-RUN (rc $rc): D1_hw029 not trained inside its cap"; exit 3; }
    ;;
  R)
    TR=$(date +%s); KS=$($P -c "print(int($KILL * 3600))")
    rc=0; timeout -k 60 "${KILL}h" $P infer3663.py --workers 10 --cap-h "$KILL" || rc=$?
    CAPKILL=0
    if [ "$rc" = 124 ] || { [ "$rc" = 137 ] && [ $(( $(date +%s) - TR )) -ge "$KS" ]; }; then CAPKILL=1; fi
    if [ "$CAPKILL" = 1 ]; then echo "READ KILLED at $KILL h: NOT-RUN"; mark_not_run "killed at $KILL h"
    elif [ "$rc" != 0 ]; then echo "INFERENCE CRASH (rc $rc): VOID"; fi
    $P score3663.py
    [ "$rc" = 0 ] || [ "$CAPKILL" = 1 ] || exit "$rc"
    ;;
esac
