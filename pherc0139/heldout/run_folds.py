"""VES-2257 GPU stage: the 5 folds in the registered order (FOLDS), per fold the 4 runs J0, J1, D0, D1 as parallel
processes on the one GPU of this unit (3 CPU threads each, 12 in total), then the next fold. Budget guard: a fold
is started only if elapsed + 1.1 x the longest fold so far (first fold: 0.27 h assumed, the smoke-test rate) stays within --budget-h (default BUDGET_H); folds
not started are NOT-RUN (score_heldout.py). Resumable: a run whose train_*.json exists is skipped.

  python heldout/run_folds.py [--steps 6000] [--folds w035,...] [--budget-h 0.8]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

FOLDS = ('w035', 'w044', 'w040', 'w039', 'w041')     # registered order; w035 = the 1-fold fallback
RUNS = (('J', 0), ('J', 1), ('D', 0), ('D', 1))
BUDGET_H = 0.8
FIRST_GUESS_S = 0.27 * 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--folds', default=','.join(FOLDS))
    ap.add_argument('--budget-h', type=float, default=BUDGET_H)
    a = ap.parse_args()
    folds = a.folds.split(',')
    assert all(f in FOLDS for f in folds)
    T0, longest = time.time(), 0.0
    for f in folds:
        el = time.time() - T0
        need = 1.1 * (longest or FIRST_GUESS_S)
        if el + need > a.budget_h * 3600:
            print(f'budget guard: fold {f} not started (elapsed {el:.0f} s + {need:.0f} s > {a.budget_h} h)', flush=True)
            break
        t = time.time()
        procs = []
        for arm, s in RUNS:
            if (C.DATA / 'heldout' / f'train_{arm}{s}_h{f}.json').exists():
                continue
            log = open(C.DATA / 'heldout' / f'log_{arm}{s}_h{f}.txt', 'w')
            procs.append((f'{arm}{s}', subprocess.Popen(
                [sys.executable, str(HERE / 'train_fold.py'), '--fold', f, '--arm', arm, '--seed', str(s),
                 '--steps', str(a.steps), '--threads', '3'], stdout=log, stderr=subprocess.STDOUT)))
        rc = {n: p.wait() for n, p in procs}
        longest = max(longest, time.time() - t)
        print(f'fold {f}: {time.time() - t:.0f} s, exit codes {rc}', flush=True)
        if any(rc.values()):
            sys.exit(f'fold {f}: a run failed {rc}')
    print(f'total {time.time() - T0:.0f} s', flush=True)


if __name__ == '__main__':
    (C.DATA / 'heldout').mkdir(parents=True, exist_ok=True)
    main()
