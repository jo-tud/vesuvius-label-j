"""Fresh-clone check: re-score one held-out fold from its stored net outputs with the leak measure and compare with
the committed results.json. No GPU, no CT read, no training: it reads the per-point arrays and fold outputs written by
steps 1-4 (data folders below) and calls the scorer of attempt4/score4.py unchanged.

  python check_fold.py --scroll 1667 --fold w028     # RUN_1667 = folder with prep.json, attempt4/, heldout/
  python check_fold.py --scroll 0139 --fold w040     # RUN_0139 and POINTS_0139 (points.npz)

Exit 0 when every run's L and L0 on the fold's label spots match the stored numbers to 1e-9.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORED = {'0139': ROOT / 'pherc0139/attempt4/results.json', '1667': ROOT / 'pherc1667/attempt2/results.json'}
TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scroll', choices=sorted(STORED), default='1667')
    ap.add_argument('--fold', default='w028')
    a = ap.parse_args()
    code = ROOT / f'pherc{a.scroll}' / 'attempt4' / 'score4.py'
    spec = importlib.util.spec_from_file_location('score4', code)
    M = importlib.util.module_from_spec(spec)
    sys.modules['score4'] = M
    spec.loader.exec_module(M)
    stored = json.loads(STORED[a.scroll].read_text())['folds'][a.fold]['runs']
    U = M.Sites()
    idx, O = U.fidx[a.fold], M.Out(U, a.fold)
    trs = {n: json.loads((M.HO / f'train_{n}_h{a.fold}.json').read_text()) for n in M.NAMES}
    runs, _ = M.score_set(M.NAMES, O.site, trs, idx, U)
    ok = True
    print(f'PHerc. {a.scroll}, held-out segment {a.fold}: leak L at labelled ink spots (no-depth baseline L0)')
    for n in M.NAMES:
        new, old = runs[n]['label'], stored[n]['label']
        same = all(abs(new[k] - old[k]) <= TOL for k in ('L', 'L0'))
        ok &= same
        label = {'J': 'shell label', 'D': 'extruded label'}[n[0]]
        print(f'  {label:15s} seed {n[1]}: L {100 * new["L"]:5.2f} %  L0 {100 * new["L0"]:5.2f} %   '
              f'stored L {100 * old["L"]:5.2f} %  L0 {100 * old["L0"]:5.2f} %   {"MATCH" if same else "DIFFERENT"}')
    print('PASS: matches the stored results.json' if ok else 'FAIL: differs from the stored results.json')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
