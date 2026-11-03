#!/bin/bash
# VES-2344 step 1 after VES-2321's WIN (README "Step 1 run"). selftest touches old sites only; g2, points, infer and
# score run only after the delta R9 of step1.py + the score_lf re-pin passes.
#   ./run_step1.sh selftest   points path + 26-net replay on VES-2321's old sites (CPU), no Will output
#   ./run_step1.sh g2         G2 on the 480 old sites (CPU)
#   ./run_step1.sh points     Will's 612 sites: tiles, per-point arrays, hecate (CPU, 8 render workers + 4 threads)
#   ./run_step1.sh infer      26 nets on Will's sites after the replay (GPU)
#   ./run_step1.sh score      score_lf --check, then the scored run -> results.json
set -euo pipefail
cd "$(dirname "$0")"
export RUN_1667=${RUN_1667:-$(cd .. && pwd)/data/run1667}
export CODE_1667=${CODE_1667:-$(cd ../pherc1667 && pwd)}
export OMP_NUM_THREADS=1
CPU=${CPU_PY:-python}
GPU=${GPU_PY:-python}
case "${1:-}" in
  selftest) $CPU -I step1.py selftest --device cpu --workers 6 --threads 6 ;;
  g2)       $CPU -I step1.py g2 ;;
  points)   $CPU -I step1.py points --workers 8 --threads 4 ;;
  infer)    $GPU -I step1.py infer --device cuda --threads 4 ;;
  score)    OMP_NUM_THREADS=8 $CPU -I score_lf.py --code "$CODE_1667" --check
            OMP_NUM_THREADS=8 $CPU -I score_lf.py --code "$CODE_1667" ;;
  *) echo "usage: $0 selftest|g2|points|infer|score"; exit 2 ;;
esac
