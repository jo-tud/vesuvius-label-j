#!/bin/bash
# VES-2321 attempt 2 (VES-2360 decision a): the attempt-1 code unchanged, 12 000 training steps instead of 6000.
# Data root run_a2/ links attempt 1's prep, sites, points and P0/P1 (control, unchanged); new fold nets in run_a2/heldout.
# RAM (VES-2360 item 4): attempt 1 ran 4 train_fold.py processes at once and peaked at 32 of 32 GB; here the same
# unchanged train_fold.py runs 2 at a time (J0+J1, then D0+D1), 6 CPU threads each, so the job fits in 28 GB.
#   ./run_gpu_a2.sh gate    fold w031 only (gate: all 4 fit AUC >= 0.80, read by attempt2/fit_gate.py)
#   ./run_gpu_a2.sh rest    the other 5 folds in the registered order (only after the gate passed)
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_1667=${RUN_1667:-$(cd .. && pwd)/data/run1667}
PY=${GPU_PY:-python}
HO=$RUN_1667/heldout
mkdir -p "$HO"
fold() {
  local f=$1
  for pair in "J 0 J 1" "D 0 D 1"; do
    set -- $pair
    local pids=()
    for k in 0 2; do
      local arm=${@:$((k+1)):1} s=${@:$((k+2)):1}
      [ -f "$HO/train_${arm}${s}_h${f}.json" ] && continue
      $PY heldout/train_fold.py --fold "$f" --arm "$arm" --seed "$s" --steps 12000 --threads 6 \
        > "$HO/log_${arm}${s}_h${f}.txt" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
  done
  echo "fold $f done $(date -u +%H:%M:%S)"
}
case "${1:-}" in
  gate) fold w031 ;;
  rest) for f in w028 w029 w013 w018 w023; do fold "$f"; done ;;
  *) echo "usage: $0 gate|rest"; exit 2 ;;
esac
