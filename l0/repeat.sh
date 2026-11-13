#!/bin/bash
# One command from a fresh clone: re-score the stored 2.4 um (PHerc. 1667 level 0) results with the unchanged
# scorers and replay the released weights against their stored outputs.
#
#   bash l0/repeat.sh
#
# The two large files are on the Hugging Face model page https://huggingface.co/j6sp5r/vesuvius-label-j
# (fetched from .../resolve/main/<file>). To use a copy you already have, point the script at the folder that holds
# both .tar files (or at another URL; the file name is appended):
#
#   export LABELJ_ASSETS=/path/to/folder       (default: https://huggingface.co/j6sp5r/vesuvius-label-j/resolve/main)
#
# 0. fetches label-j-1667-l0-weights.tar (90 MB) and label-j-1667-l0-stored.tar (1.4 GB) into data/ (skipped if
#    present) and checks both against l0/archives.sha256;
# 1. unpacks them into weights/ and data/ves<id>/ and checks every file against l0/assets.sha256;
# 2. F-567 step 2 scorer (l0/f567/step2/score_s2.py) and F-606 scorer (l0/f606/score3663.py) rewrite their
#    results.json; both must be byte-equal to the committed files (git diff --exit-code);
# 3. l0/check_weights.py: every released net reruns its 4 stored replay tiles within 5e-3 (GPU if present, else CPU).
# CPU: about 4 min for 2; 3 takes a few minutes on one GPU, longer on CPU. Python: requirements.txt.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LABELJ_ASSETS=${LABELJ_ASSETS:-https://huggingface.co/j6sp5r/vesuvius-label-j/resolve/main}
PY=${PY:-python}
export LABELJ_DATA=$ROOT/data
fail() { echo "repeat: FAIL ($1)"; exit 1; }
[ $# -eq 0 ] || fail "no arguments; set LABELJ_ASSETS to use a local copy"
case "$LABELJ_ASSETS" in http://*|https://*|/*) ;; *) LABELJ_ASSETS=$PWD/$LABELJ_ASSETS ;; esac   # relative: to the caller
mkdir -p "$ROOT/data"
cd "$ROOT"
ARCHIVES="label-j-1667-l0-weights.tar label-j-1667-l0-stored.tar"
for a in $ARCHIVES; do
  [ -f "data/$a" ] && continue
  case "$LABELJ_ASSETS" in
    http://*|https://*)
      echo "0. fetching $LABELJ_ASSETS/$a"
      if ! curl -fL --retry 3 --connect-timeout 30 --max-time 7200 -o "data/$a.part" "$LABELJ_ASSETS/$a"; then
        rm -f "data/$a.part"
        fail "download of $a from $LABELJ_ASSETS; if you have the file, export LABELJ_ASSETS=<folder holding it>"
      fi
      mv "data/$a.part" "data/$a" ;;
    *)
      [ -f "$LABELJ_ASSETS/$a" ] || fail "LABELJ_ASSETS=$LABELJ_ASSETS holds no $a"
      cp "$LABELJ_ASSETS/$a" "data/$a.part" && mv "data/$a.part" "data/$a" ;;
  esac
done
if ! (cd data && sha256sum --quiet -c ../l0/archives.sha256); then
  fail "archive sha256; delete the data/*.tar named above and rerun"; fi
echo "0. archives: sha256 of both OK (source $LABELJ_ASSETS)"
for a in $ARCHIVES; do tar -xf "data/$a" -C "$ROOT"; done
if ! sha256sum --quiet -c l0/assets.sha256; then fail "asset sha256"; fi
echo "1. assets: sha256 of $(wc -l < l0/assets.sha256) files OK"
grep -v '^#' l0/net_paths.txt | while read -r name path; do      # the scorer checks that these nets exist
  mkdir -p "$(dirname "$LABELJ_DATA/$path")"; cp "weights/$name.pt" "$LABELJ_DATA/$path"
done

(cd l0/f567/step2 && VES2578_RUN=$LABELJ_DATA/ves3416/run VES3416_OUT=$LABELJ_DATA/ves3416/out \
   "$PY" score_s2.py > "$LABELJ_DATA/score_s2.log" 2>&1) || fail "score_s2.py exit (data/score_s2.log)"
(cd l0/f606 && VES2578_RUN=$LABELJ_DATA/ves3663/run VES3663_OUT=$LABELJ_DATA/ves3663/out \
   "$PY" score3663.py > "$LABELJ_DATA/score3663.log" 2>&1) || fail "score3663.py exit (data/score3663.log)"
if ! git diff --exit-code --stat -- l0/f567/step2/results.json l0/f606/results.json; then
  fail "a scorer's results.json differs from the committed file (git diff above)"; fi
echo "2. scorers: l0/f567/step2/results.json and l0/f606/results.json byte-equal to the committed files"
"$PY" - <<'PY'
import json
s2 = json.load(open('l0/f567/step2/results.json'))
f6 = json.load(open('l0/f606/results.json'))
r = s2['pooled']['runs']
print(f"   F-567 step 2: {s2['verdict']}, pooled L J0 {r['J0']['label']['L']:.3f} vs D0 {r['D0']['label']['L']:.3f}; F-606: {f6['november'][0]}")
PY
if ! "$PY" l0/check_weights.py; then fail "weight replay"; fi
echo "3. weights: every released net replays its stored outputs (l0/check_weights.json)"
echo "repeat: PASS"
