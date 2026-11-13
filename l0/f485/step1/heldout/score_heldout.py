"""VES-2257 scorer: held-out confirmation of F-399 (VES-2241 attempt 3), README "Criteria" of heldout/. CPU only.
Reuses score.py's L / L0 / ink AUC (score_run), frame-shift control (frame_control) and verdict rules (verdict_of)
unchanged; only adds the fold split, the per-fold controls and the fold rule.

  python heldout/score_heldout.py --check   scorer + hecate PC (pooled and per fold), P-type known-artefact control
                                            per fold, synthetic fold-rule cases; no held-out output read
  python heldout/score_heldout.py           heldout/results.json
VES-2578 step 1 (level 0): this file is the fold-rule library of attempt4/score4.py; unchanged rules (fold_status,
overall, synthetic_overall, MIN_WIN 5, PC_FOLD_RATIO 0.5, MASS_PER_SITE = 1000 / 200 = 5 per label site, the frozen
floor "label-site mass >= 5 x number of label sites"). Removed: main(), Loader and fold_controls, which read F-418's
old-site output layout (score4.py has its own fold_controls on the union sites).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import score as S  # noqa: E402

FOLDS = C.FOLDS                                      # VES-2321: six 1667 folds, registered order (= run_folds.FOLDS)
OUT = C.DATA / 'heldout'
PC_FOLD_RATIO = 0.5                                  # hecate L / L0 <= 0.5 on the fold's label sites (pooled 0.23)
MIN_WIN = 5                                          # VES-2321: J wins >= 5 of the 6 folds (F-401: 4 of 5)
MASS_PER_SITE = S.MIN_MASS / 200                     # attempt-3 mass floor 1000 over 200 label sites, scaled by sites


def sub(xs, idx):
    return [xs[i] for i in idx]


def fold_status(ctrl, runs, frame):
    """One trained fold. VOID: a fold control, a non-finite output or fit AUC < 0.80; NO-VERDICT: a J or D run under the
    ink AUC floor / mass floor / defined share; ARTEFACT: a J run ARTEFACT; else scored with win = mean J L < mean D L
    and follows = both J runs FOLLOWS."""
    if not ctrl['ok'] or not all(r['fit_ok'] for r in runs.values()):
        return dict(status='VOID')
    if not all(r['measurable'] for r in runs.values()):
        return dict(status='NO-VERDICT')
    if any(frame[f'J{s}'] == 'ARTEFACT' for s in C.SEEDS):
        return dict(status='ARTEFACT', scored=True, win=False, follows=False)
    LJ = float(np.mean([runs[f'J{s}']['label']['L'] for s in C.SEEDS]))
    LD = float(np.mean([runs[f'D{s}']['label']['L'] for s in C.SEEDS]))
    win, follows = LJ < LD, all(frame[f'J{s}'] == 'FOLLOWS' for s in C.SEEDS)
    st = 'WIN' if win and follows else 'LOSS' if not win else 'UNCLEAR'
    return dict(status=st, scored=True, win=bool(win), follows=bool(follows), L_J=LJ, L_D=LD)


def overall(checks_ok, folds, pooled_v):
    """README "Fold rule". folds: {f: fold_status | {'status': 'NOT-RUN'}}; pooled_v: verdict_of dict over the scored
    folds' sites (None if no scored fold)."""
    st = [folds[f]['status'] for f in FOLDS]
    n_win = st.count('WIN')
    scored = [folds[f] for f in FOLDS if folds[f].get('scored')]
    bad = sum(s in ('LOSS', 'UNCLEAR') for s in st)
    pooled_j = bool(pooled_v and pooled_v['verdict'] == 'LABEL-EFFECT' and pooled_v.get('lower_L_arm') == 'J')
    if not checks_ok:
        return 'VOID'
    if 'ARTEFACT' in st:
        return 'NOT-CONFIRMED'
    if n_win >= MIN_WIN and all(s['follows'] for s in scored) and pooled_j:
        return 'CONFIRMED'
    if n_win >= MIN_WIN:
        return 'NOT-CONFIRMED'                       # wins enough folds, but a scored fold not FOLLOWS or pooled fails
    if bad == 0 and n_win + st.count('NOT-RUN') >= MIN_WIN and 'NOT-RUN' in st:
        return 'INCOMPLETE'                          # GPU cut; CONFIRMED still reachable; F-399 stays candidate
    if bad == 0:
        return 'NO-VERDICT'                          # short of 4 wins only through VOID / NO-VERDICT / NOT-RUN folds
    return 'NOT-CONFIRMED'


def synthetic_overall():
    W = dict(status='WIN', scored=True, win=True, follows=True)
    L = dict(status='LOSS', scored=True, win=False, follows=True)
    U = dict(status='UNCLEAR', scored=True, win=True, follows=False)
    A = dict(status='ARTEFACT', scored=True, win=False, follows=False)
    V, N, R = dict(status='VOID'), dict(status='NO-VERDICT'), dict(status='NOT-RUN')
    good, badp = dict(verdict='LABEL-EFFECT', lower_L_arm='J'), dict(verdict='INCONCLUSIVE')
    cases = {'6win': ([W] * 6, good, 'CONFIRMED'), '5win1loss': ([W] * 5 + [L], good, 'CONFIRMED'),
             '5win1void': ([W] * 5 + [V], good, 'CONFIRMED'), '5win_pooled_fail': ([W] * 5 + [L], badp, 'NOT-CONFIRMED'),
             '5win1unclear': ([W] * 5 + [U], good, 'NOT-CONFIRMED'), '4win2loss': ([W] * 4 + [L] * 2, good, 'NOT-CONFIRMED'),
             '5win1artefact': ([W] * 5 + [A], good, 'NOT-CONFIRMED'), '4win2void': ([W] * 4 + [V, N], good, 'NO-VERDICT'),
             'fallback_1win': ([W] + [R] * 5, good, 'INCOMPLETE'), 'fallback_1loss': ([L] + [R] * 5, good, 'NOT-CONFIRMED'),
             'fallback_1artefact': ([A] + [R] * 5, None, 'NOT-CONFIRMED'), '4win1loss1notrun': ([W] * 4 + [L, R], good,
                                                                                               'NOT-CONFIRMED'),
             'pooled_D_lower': ([W] * 6, dict(verdict='LABEL-EFFECT', lower_L_arm='D'), 'NOT-CONFIRMED')}
    out = {}
    for name, (fs, pv, want) in cases.items():
        got = overall(True, dict(zip(FOLDS, fs)), pv)
        out[name] = dict(want=want, got=got, ok=got == want)
        print(f'synthetic fold rule {name}: {got} (want {want})')
    got = overall(False, dict(zip(FOLDS, [W] * 6)), good)
    out['checks_fail'] = dict(want='VOID', got=got, ok=got == 'VOID')
    return dict(cases=out, ok=all(c['ok'] for c in out.values()))


if __name__ == '__main__':
    print(synthetic_overall()['ok'])
