#!/bin/bash
# VES-3380 (attempt 3; copy of the VES-3347 overlay, own dir ves3380/run): run dir RUN = a read-only overlay of F-485's step-1 run dir (step1a2) with its own heldout/ dir, so that
# F-485's heldout/train_fold.py runs unchanged (it writes to C.DATA/heldout) and nothing is written into step1a2.
# Every top-level entry of step1a2 except heldout/ is a symlink; heldout/ holds symlinks to the frozen F-485 J nets of
# fold w029 (net, train json, replay) and, after training, our own D0_hw029 files (24 000 steps). Idempotent.
set -euo pipefail
export LABELJ_DATA=${LABELJ_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data}   # release: stored runs under data/ves<id>/
SRC=${SRC:-$LABELJ_DATA/ves2578/step1a2}
RUN=${RUN:-$LABELJ_DATA/ves3380/run}
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
ls -l "$RUN" "$RUN/heldout"
