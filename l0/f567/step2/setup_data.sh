#!/bin/bash
# VES-3416 (F-567 step 2): run dir RUN = a read-only overlay of F-485's step-1 run dir (step1a2) with its own heldout/
# dir, so that F-485's heldout/train_fold.py runs unchanged (it writes to C.DATA/heldout) and nothing is written into
# step1a2 or ves3380. Every top-level entry of step1a2 except heldout/ is a symlink; heldout/ holds symlinks to the
# frozen nets (F-485 J0 / J1 w029 and J0 w028 from step1a2, F-567 D0 24 000 steps from ves3380/run) and, after training, the
# step-2 nets (J0 + D0 on w013, w018, w023; D0 on w028 next to F-485's frozen J0_hw028). Idempotent.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/data}   # release: stored runs under data/ves<id>/
SRC=${SRC:-$LABELJ_DATA/ves2578/step1a2}
S1=${S1:-$LABELJ_DATA/ves3380/run/heldout}
RUN=${RUN:-$LABELJ_DATA/ves3416/run}
mkdir -p "$RUN/heldout"
for p in "$SRC"/*; do
  b=$(basename "$p")
  case "$b" in heldout|IN_USE.txt) continue ;; esac
  [ -e "$RUN/$b" ] || ln -s "$(readlink -f "$p")" "$RUN/$b"
done
for n in J0 J1; do
  for f in net_${n}_hw029.pt train_${n}_hw029.json out_${n}_hw029_replay.npz; do
    [ -e "$RUN/heldout/$f" ] || ln -s "$SRC/heldout/$f" "$RUN/heldout/$f"
  done
done
for f in net_J0_hw028.pt train_J0_hw028.json out_J0_hw028_replay.npz; do
  [ -e "$RUN/heldout/$f" ] || ln -s "$SRC/heldout/$f" "$RUN/heldout/$f"
done
for f in net_D0_hw029.pt train_D0_hw029.json out_D0_hw029_replay.npz; do
  [ -e "$RUN/heldout/$f" ] || ln -s "$(readlink -f "$S1/$f")" "$RUN/heldout/$f"
done
ls -l "$RUN/heldout"
