#!/bin/bash
# VES-3416 (F-567 step 2) one GPU unit: ./unit.sh FOLD STAGE, STAGE = J (train J0, 12 000 steps), D (train D0,
# 24 000 steps) or R (gate 1, then the one held-out read of the fold and the scorer). Queued in the registered order
# (w013, w028, w018, w023; J, D, R per fold; w028 has no J stage) right after the R9 PASS; no unit is cancelled on a
# result (no optional stopping). Every unit first re-runs the CPU check (positive controls, scorer, synthetic rules):
# check_ok not true -> exit 4, nothing trained or read. Budget: the task ledger (bars unit + every unit's wall time,
# OUT/ledger.jsonl) may not pass 35 GPU-h: a unit starts only if ledger + its expected cost <= 35 h, and its kill time
# is min(stage cap, 35 h - ledger - this unit's time so far). A unit not started or killed leaves its net NOT-RUN
# (the scorer reads that); an R unit stopped by the guard or killed at its cap writes infer_<f>.json with not_run
# true, so the scorer reads the fold as NOT-RUN (inference budget stop), not VOID (R9 VES-3419: RS, BUD). A SIGKILL
# before the cap (e.g. a MemoryMax OOM kill, rc 137) is a crash: no mark, VOID (R9 VES-3420). The kill time keeps
# 0.05 h back for the 60 s kill grace and the post-read scorer, so the ledger cannot pass 35 GPU-h.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
cd "$(dirname "$0")"
F=$1; STAGE=$2
export VES2578_RUN=${VES2578_RUN:-$LABELJ_DATA/ves3416/run}
export VES3416_OUT=${VES3416_OUT:-$LABELJ_DATA/ves3416/out}
P=${GPU_PY:-python}
TF=../../f485/step1/heldout/train_fold.py
HO=$VES2578_RUN/heldout
mkdir -p "$VES3416_OUT"
case "$STAGE" in
  J) EST=2.4; CAP=3.0 ;;     # F-485 J0 w029 8 360 s, w028 7 422 s
  D) EST=4.4; CAP=5.6 ;;     # F-567 D0 w029 15 025 s
  R) EST=1.3; CAP=2.0 ;;     # F-567 D0 read 4 147 s, F-545 w029 4 223 s (render bound)
  *) echo "stage J|D|R"; exit 2 ;;
esac
T0=$(date +%s)
log_unit() {
  local rc=$?
  $P -c "import json,sys,time; t0=int(sys.argv[1]); open(sys.argv[2],'a').write(json.dumps(dict(fold=sys.argv[3], stage=sys.argv[4], start=t0, end=int(time.time()), seconds=int(time.time())-t0, rc=int(sys.argv[5])))+'\n')" \
    "$T0" "$VES3416_OUT/ledger.jsonl" "$F" "$STAGE" "$rc"
}
trap log_unit EXIT
mark_not_run() {   # R stage: the read did not run to its end inside the budget -> NOT-RUN (scorer read_state)
  $P -c "import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; sys.exit(0) if d.get('done') is True else None; d.update(fold=sys.argv[2], done=False, not_run=True, stop=sys.argv[3]); p.write_text(json.dumps(d, indent=1))" \
    "$VES3416_OUT/infer_${F}.json" "$F" "$1"
}
./setup_data.sh > /dev/null
$P score_s2.py --check > "$VES3416_OUT/check_${F}_${STAGE}.log" 2>&1
$P -c "import json, sys; sys.exit(0 if json.load(open('check.json'))['check_ok'] is True else 4)" \
  || { echo "CHECK FAIL (positive controls / scorer / rules): nothing trained or read"; exit 4; }
USED=$($P -c "import score_s2 as S; print(S.ledger()['gpu_h'])")
USED=$($P -c "print(round($USED + ($(date +%s) - $T0) / 3600, 4))")   # + this unit's setup and check time
LEFT=$($P -c "print(round(35.0 - $USED, 3))")
if $P -c "import sys; sys.exit(0 if $USED + $EST <= 35.0 else 1)"; then :; else
  echo "BUDGET GUARD: ledger $USED GPU-h + $EST > 35: $F $STAGE NOT-RUN"
  [ "$STAGE" = R ] && [ ! -f "$VES3416_OUT/infer_${F}.json" ] && mark_not_run "budget guard: ledger $USED + $EST > 35"
  exit 0
fi
KILL=$($P -c "print(round(min($CAP, $LEFT - 0.05), 4))")
echo "$F $STAGE: ledger $USED GPU-h, kill after $KILL h"
case "$STAGE" in
  J|D)
    ARM=$STAGE; STEPS=$([ "$ARM" = J ] && echo 12000 || echo 24000)
    [ "$F" = w028 ] && [ "$ARM" = J ] && { echo "w028 J0 is F-485's frozen net: no J stage"; exit 0; }
    [ -f "$HO/train_${ARM}0_h${F}.json" ] && { echo "${ARM}0_h$F already trained"; exit 0; }
    if [ "$ARM" = D ] && [ ! -f "$HO/train_J0_h${F}.json" ]; then echo "J0_h$F not trained: D0_h$F not started (fold NOT-RUN)"; exit 0; fi
    rc=0
    timeout "${KILL}h" $P "$TF" --fold "$F" --arm "$ARM" --seed 0 --steps "$STEPS" --threads 6 2>&1 \
      | tee -a "$VES3416_OUT/log_${ARM}0_h${F}.txt" || rc=$?
    [ "$rc" = 0 ] || { echo "TRAIN NOT-RUN (rc $rc): ${ARM}0_h$F not trained inside its cap"; exit 3; }
    ;;
  R)
    rc=0; $P score_s2.py --gate1 "$F" || rc=$?
    if [ "$rc" = 3 ]; then echo "$F: a net is NOT-RUN, no read"; $P score_s2.py; exit 0; fi
    if [ "$rc" = 10 ]; then echo "GATE1 FAIL $F: fold not read (VOID)"; $P score_s2.py; exit 0; fi
    [ "$rc" = 0 ] || exit "$rc"
    TR=$(date +%s); KS=$($P -c "print(int($KILL * 3600))")
    rc=0; timeout -k 60 "${KILL}h" $P infer_s2.py --fold "$F" --workers 10 --cap-h "$KILL" || rc=$?
    CAPKILL=0   # 124 = TERM at the cap; 137 counts only if the read ran to its cap (else an outside SIGKILL = crash)
    if [ "$rc" = 124 ] || { [ "$rc" = 137 ] && [ $(( $(date +%s) - TR )) -ge "$KS" ]; }; then CAPKILL=1; fi
    if [ "$CAPKILL" = 1 ]; then
      echo "READ KILLED at $KILL h: $F NOT-RUN (inference budget stop)"; mark_not_run "killed at $KILL h"
    elif [ "$rc" != 0 ]; then echo "INFERENCE CRASH (rc $rc): $F VOID"; fi
    $P score_s2.py
    [ "$rc" = 0 ] || [ "$CAPKILL" = 1 ] || exit "$rc"
    ;;
esac
