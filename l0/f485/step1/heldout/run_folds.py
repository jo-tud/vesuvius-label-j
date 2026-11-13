"""VES-2257 GPU stage: the folds in the registered order (FOLDS), per fold the 4 runs J0, J1, D0, D1 on the one GPU of
this unit, then the next fold. Resumable: a run whose train_*.json exists is skipped; folds not started are NOT-RUN
(score4.py).
VES-2578 step 1 (level 0), changes against F-418's run_folds.py:
- first the known-artefact control P0, P1 (train.py --arm P, all six segments), then the folds; runs go 2 at a time
  (P0+P1, J0+J1, D0+D1, 6 CPU threads each), the schedule F-418 attempt 2 used and the step-0 cost model assumed;
- 12 000 steps;
- budget guard over training AND inference (at level 0 the site outputs need the GPU too, attempt4/infer4.py): a
  fold is started only if elapsed + 1.1 x the longest fold so far (first fold: 2 x the P pair) + the inference
  reserve of every started fold and of this one stays within --budget-h. Inference reserve per fold =
  WINDOWS_PER_SITE x its sites x S_WINDOW x 1.1 (S_WINDOW: bench_infer.json, 384 x 384 x 128 fp32 window).

Attempt 2 (VES-2741 grant, patch 128, 54.5 GB VRAM per run), changes against the attempt-1 runner above:
- one run at a time (two patch-128 runs do not fit the 96 GB card): P1, then per fold J0, J1, D0, D1; P0 is the
  pilot's patch-128 net (run_a2_setup.sh copies it; asserted: arm P, seed 0, 12 000 steps, patch 128), not retrained;
- budget guard 58 GPU-h over attempt 2's training (P1 and the fold runs; P0 was paid under VES-2644) plus the
  inference reserve, kept across units in DATA/heldout/run_log.jsonl (wall seconds of every finished run). A fold is
  started only if used + 1.1 x the longest run x its runs left + the inference reserve of every started fold and of
  this one stays within --budget-h; P1 likewise with its one run. A refusal stops the runner (BUDGET STOP: ask the
  Chief of staff, VES-2741 term 1); folds not started are NOT-RUN;
- unit cap (VES-2741 term 4, one unit <= 6 h): a run starts only if the unit's elapsed time + 1.1 x the longest run
  stays within --unit-h; otherwise the unit ends with "UNIT CAP" and the next unit resumes (finished runs are skipped).

  python heldout/run_folds.py [--steps 12000] [--folds w028,...] [--budget-h 58] [--unit-h 5.75]
Writes DATA/heldout/run_folds.json (this unit: runs, guard decisions, stop reason) and DATA/heldout/run_log.jsonl.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import train as T  # noqa: E402

FOLDS = C.FOLDS
RUNS = (('J', 0), ('J', 1), ('D', 0), ('D', 1))
BUDGET_H = 60.0                                      # VES-2741 58, raised to 60 by the CoS (VES-2634, 11-03); replaces 7.5
UNIT_H = 5.75                                        # VES-2741 term 4: one unit <= 6 h
S_WINDOW = json.loads((HERE.parent / 'bench_infer.json').read_text())['shapes'][0]['s_window']   # 384, fp32
WINDOWS_PER_SITE = 2 * 6 + 2 * 2 + 2 * 6             # J0, J1: k -16, 0, 16 x 2 phases; D0, D1: k 0; P0, P1: as J
MARGIN = 1.1


def infer_reserve_s(folds, keys):
    return sum(sum(k[0] == f for k in keys) for f in folds) * WINDOWS_PER_SITE * S_WINDOW * MARGIN


def jpath(name, fold=None):
    return (C.DATA / f'train_{name}.json') if fold is None else (C.DATA / 'heldout' / f'train_{name}_h{fold}.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--folds', default=','.join(FOLDS))
    ap.add_argument('--budget-h', type=float, default=BUDGET_H)
    ap.add_argument('--unit-h', type=float, default=UNIT_H)
    a = ap.parse_args()
    folds = a.folds.split(',')
    assert all(f in FOLDS for f in folds)
    keys = json.loads((C.DATA / 'prep.json').read_text())['site_keys']
    HO = C.DATA / 'heldout'
    p0 = json.loads(jpath('P0').read_text())                                  # the pilot's net (run_a2_setup.sh)
    assert (p0['arm'], p0['seed'], p0['steps'], p0['patch']) == ('P', 0, a.steps, T.PATCH), 'P0 is not the pilot net'
    RL = HO / 'run_log.jsonl'
    ledger = [json.loads(x) for x in RL.read_text().splitlines()] if RL.exists() else []
    log = dict(budget_h=a.budget_h, unit_h=a.unit_h, s_window=S_WINDOW, windows_per_site=WINDOWS_PER_SITE,
               steps=a.steps, patch=T.PATCH, p0_seconds=p0['seconds'], runs=[], guard=[])
    todo = [('P1', None, [sys.executable, str(HERE.parent / 'train.py'), '--arm', 'P', '--seed', '1', '--steps',
                          str(a.steps), '--threads', '6'])]
    todo += [(f'{arm}{s}', f, [sys.executable, str(HERE / 'train_fold.py'), '--fold', f, '--arm', arm, '--seed',
                               str(s), '--steps', str(a.steps), '--threads', '6']) for f in folds for arm, s in RUNS]
    T0, stop = time.time(), 'done'
    for name, f, cmd in todo:
        if jpath(name, f).exists():
            continue
        used = sum(r['wall_s'] for r in ledger)
        longest = max(r['wall_s'] for r in ledger) if ledger else p0['seconds']      # first run: the pilot P0's time
        started = sorted({r['fold'] for r in ledger if r['fold']} | ({f} if f else set()))
        left = 1 if f is None else sum(not jpath(f'{arm}{s}', f).exists() for arm, s in RUNS)
        need, res = MARGIN * longest * left, infer_reserve_s(started, keys)
        ok = used + need + res <= a.budget_h * 3600
        g = dict(run=name, fold=f, used_s=round(used), need_s=round(need), infer_reserve_s=round(res), start=ok)
        log['guard'].append(g)
        if not ok:
            stop = 'budget'
            print(f'BUDGET STOP: {name} {f or ""} not started: used {used:.0f} s + {need:.0f} s + inference '
                  f'{res:.0f} s > {a.budget_h} h. Ask the Chief of staff (VES-2741).', flush=True)
            break
        if time.time() - T0 + MARGIN * longest > a.unit_h * 3600:
            stop = 'unit_cap'
            break
        t = time.time()
        lg = (C.DATA / f'log_{name}.txt') if f is None else (HO / f'log_{name}_h{f}.txt')
        rc = subprocess.run(cmd, stdout=open(lg, 'w'), stderr=subprocess.STDOUT).returncode
        wall = time.time() - t
        r = dict(run=name, fold=f, wall_s=round(wall, 1), exit=rc, unit_t0=round(T0))
        log['runs'].append(r)
        (HO / 'run_folds.json').write_text(json.dumps(log, indent=1))
        print(f'{name} {f or ""}: {wall:.0f} s, exit {rc}', flush=True)
        if rc:
            sys.exit(f'{name} {f}: run failed (exit {rc}), see {lg}')
        ledger.append(r)
        with open(RL, 'a') as fh:
            fh.write(json.dumps(r) + '\n')
    left = [n + (f'_h{f}' if f else '') for n, f, _ in todo if not jpath(n, f).exists()]
    log.update(stop=stop, left=left, unit_seconds=round(time.time() - T0, 1),
               used_s=round(sum(r['wall_s'] for r in ledger)))
    (HO / 'run_folds.json').write_text(json.dumps(log, indent=1))
    msg = dict(done='ALL RUNS DONE', unit_cap=f'UNIT CAP: {len(left)} runs left, resubmit',
               budget=f'BUDGET STOP: {len(left)} runs left')[stop]
    print(f'{msg}; attempt-2 GPU time used {log["used_s"] / 3600:.2f} h of {a.budget_h} h', flush=True)


if __name__ == '__main__':
    (C.DATA / 'heldout').mkdir(parents=True, exist_ok=True)
    main()
