"""VES-3416 (F-567 step 2) scorer: label J vs D at level 0 on the 6 F-485 folds (README "Step 2 criteria"): new
J0 (12 000 steps) + D0 (24 000 steps) on w013, w018, w023; new D0 (24 000 steps) on w028 next to F-485's frozen J0
(F-545 outputs); w029 = step 1 (frozen outputs, not re-read); w031 VOID before training (its hecate positive control
F1 fails: L 0.1135 > 0.5 x L0 0.2215, check.json). Pooled rule = F-485's overall() over the 6 folds, unchanged. CPU only. Every frozen rule function is imported unchanged: F-485 step1/score.py (score_run, frame_control,
checks, verdict_of), attempt4/score4.py (Sites, fold_controls, score_set, auc_ci), heldout/score_heldout.py (overall,
MIN_WIN 5), F-545's score3226.py (w029 bar, loader, p_shift) and F-567 step 1's score3380.py (fold_status_jd,
score_fold for w029, keep_keys). New here, all written before any step-2 net exists:
  - gate1(f): J0_h<f> (12 000 steps) and D0_h<f> (24 000 steps) are the registered runs and both have fit AUC >= the
    fold's bar (bars.json: P0 on the fold's 5-segment draw - 0.03, F-538 rule); a fail leaves the fold unread (VOID);
  - read_state(f): NOT-RUN (a net not trained: kill or budget guard; or inference budget stop), VOID (gate 1 fail,
    inference crash or an inference gate fail), READ;
  - pooled_verdict: score.verdict_of on the J0 / D0 runs of the pooled sites with the seed noise fixed to F-545's
    measured w029 J0-J1 spread (one seed per arm: no own spread);
  - C6: w028's J0 recomputed here equals F-545 (L, L0, ink AUC, fit_ok, controls);
  - november_line: the registered decision rule for the November text.

  python score_s2.py --check       bars, gates, scorer + hecate PCs (pooled and per new fold), w029 = step 1 (C5),
                                   synthetic rule cases -> check.json; reads no step-2 output
  python score_s2.py --gate1 FOLD  gate 1 of one fold (after its J0 and D0 train) -> gate1_<f>.json; exit 0 / 10 / 3
  python score_s2.py               results.json (every fold in its current state)
"""
import argparse
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np

HERE = Path(__file__).resolve().parent
S1DIR = HERE.parent
STEP1 = S1DIR.parent / 'f485' / 'step1'
F545 = S1DIR.parent / 'f545'
for p in (STEP1, STEP1 / 'heldout', STEP1 / 'attempt4', F545, S1DIR):
    sys.path.insert(0, str(p))
import common as C  # noqa: E402
import score as S  # noqa: E402
import score_heldout as SH  # noqa: E402
import score4 as S4  # noqa: E402
import score3226 as SC  # noqa: E402
import score3380 as S1  # noqa: E402

OUT = Path(os.environ.get('VES3416_OUT', _VD + '/ves3416/out'))
HO = C.DATA / 'heldout'
NEW = ('w013', 'w028', 'w018', 'w023')                    # step-2 folds, run order (w028 replaces w031: README)
FOLDS6 = C.FOLDS                                          # F-485's 6 folds; overall() reads all six
NOT_TRAINED = {'w031': 'F1 hecate PC fails before training (L 0.1135 > 0.5 x L0 0.2215): fold VOID, not trained'}
NAMES = ('J0', 'D0')
NEW_NETS = {'w013': NAMES, 'w028': ('D0',), 'w018': NAMES, 'w023': NAMES}   # nets trained in step 2
STEPS = {'J': 12000, 'D': 24000}
SEED_NOISE = 0.011575894                                  # F-545 w029 |L(J0) - L(J1)| = 0.053863 - 0.042287
BARS = HERE / 'bars.json'
RESULTS, CHECK = HERE / 'results.json', HERE / 'check.json'
S1_RESULTS = S1DIR / 'results.json'
F545_RESULTS = F545 / 'results.json'
GPU_CAP_H = 35.0


def sub(xs, idx):
    return [xs[i] for i in idx]


def bars():
    """Per-fold bars: w029, w028 = F-538 (score3226.bars); w013, w018, w023, w031 = bars.json (bars_s2.py, B1, B2 pass)."""
    b = json.loads(BARS.read_text())
    assert b['ok'] and set(FOLDS6) <= set(b['folds'])
    out = {f: dict(P0_on_draw=b['folds'][f]['P0_on_draw'], bar=b['folds'][f]['bar']) for f in FOLDS6}
    for f in ('w029', 'w028'):
        out[f] = SC.bars()[f]
        assert abs(b['folds'][f]['bar'] - out[f]['bar']) <= 1e-9
    return out


def trained(f, n):
    return (HO / f'train_{n}_h{f}.json').exists() and (HO / f'net_{n}_h{f}.pt').exists()


def gate1(f):
    """J0 and D0 of fold f are the registered runs and both fit AUCs >= the fold's bar. None if a net is not trained."""
    if not all(trained(f, n) for n in NAMES):
        return None
    b = bars()[f]
    kept = sorted(set(C.FOLDS) - {f})
    nets = {}
    for n in NAMES:
        tr = json.loads((HO / f'train_{n}_h{f}.json').read_text())
        reg = bool((tr['fold'], tr['arm'], tr['seed'], tr['steps'], tr['patch'], tr['batch']) ==
                   (f, n[0], 0, STEPS[n[0]], 128, 16) and tr['kept_segments'] == kept and tr['limit_blocks'] is None)
        fit = float(tr['fit_auc_shell'])
        nets[n] = dict(fit_auc_shell=fit, margin=fit - b['bar'], registered=reg,
                       fit_ok=bool(np.isfinite(fit) and fit >= b['bar']), seconds=tr['seconds'])
    return dict(fold=f, bar=b['bar'], P0_on_draw=b['P0_on_draw'], nets=nets,
                passed=bool(all(x['registered'] and x['fit_ok'] for x in nets.values())))


def infer_gates(inf, f, replay):
    """G2 net replay (w013 / w018 / w023: P0, P1, J0_h<f>, D0_h<f>; w028: D0_hw028; each finite and <= tol), G3 site
    replay (every replay point; a tile for every replay site of fold f), G4 the registered nets, not a smoke."""
    rs = inf['replay_sites']
    n_tiles = sum(r['key'][0] == f for r in replay)
    want = ({f'D0_h{f}'}, ['D0']) if f == 'w028' else ({'P0', 'P1', f'J0_h{f}', f'D0_h{f}'}, ['J0', 'D0', 'P0', 'P1'])
    return dict(
        replay_nets=bool(set(inf['replay']) == want[0] and
                         all(np.isfinite(v) and v <= inf['replay_tol'] for v in inf['replay'].values())),
        replay_sites=bool(all(rs.values()) and sum(k.startswith('points_') for k in rs) == len(replay) and
                          sum(k.startswith('tile_') for k in rs) == n_tiles),
        nets_as_registered=bool(inf['fold'] == f and inf['nets'] == want[1] and not inf['smoke']))


def read_state(f, g1, inf, replay, n_sites):
    """NOT-RUN: a net not trained (kill / budget guard) or the inference budget stop; VOID: gate 1 fail (fold not
    read), inference crash or an inference gate fail; READ: outputs exist and every inference gate passes."""
    if g1 is None:
        return 'NOT-RUN', 'net not trained'
    if not g1['passed']:
        return 'VOID', 'gate1'
    if inf is None:
        return 'VOID', 'no infer json after gate 1 passed (crash)'
    if inf.get('not_run'):
        return 'NOT-RUN', 'inference budget stop'
    if not inf.get('done'):
        return 'VOID', 'inference not done (crash)'
    g = infer_gates(inf, f, replay)
    if not all(g.values()) or inf.get('sites') != n_sites:
        return 'VOID', f'inference gate {g}'
    return 'READ', None


def hecate_pc(U, idx):
    """F1: the hecate half of score4.fold_controls, verbatim (P outputs are not needed): L <= 0.5 L0 on the fold."""
    P, F = sub(U.pts, idx), sub(U.flags, idx)
    H = [d['p_hecate_fwd'].astype(np.float32) for d in P]
    sm = S.summarise([S.leak_site(h, d) for h, d in zip(H, P)], F, H, P)
    pc = sm['label']
    ok = bool(pc['L'] is not None and pc['L0'] and pc['L'] <= SH.PC_FOLD_RATIO * pc['L0'])
    return dict(L=pc['L'], L0=pc['L0'], ink_auc=sm['ink_auc'], ok=ok)


class Fold:
    """Outputs of fold f: w029 from F-545 (J0, J1, P) and F-567 step 1 (D0); w028 J0 and P from F-545, D0 from OUT;
    w013, w018, w023 from OUT."""
    FROZEN_J = ('w029', 'w028')

    def __init__(self, f):
        self.f, self.z = f, {}
        if f in self.FROZEN_J:
            self.jsite, self.jshift = SC.loader(f)

    def _ld(self, p):
        if p not in self.z:
            self.z[p] = np.load(p)
        return self.z[p]

    def site(self, n, u):
        if self.f in self.FROZEN_J and n[0] == 'J':
            return self.jsite(n, u)
        if self.f == 'w029':
            return self._ld(S1.OUT / f'out_{n}_hw029.npz')[f'p_{u}']
        return self._ld(OUT / f'out_{n}_h{self.f}.npz')[f'p_{u}']

    def shift(self, n):
        if self.f in self.FROZEN_J:
            return self.jshift(n)
        return lambda k, u: self._ld(OUT / f'out_{n}_h{self.f}_shift.npz')[f'p{k}_{u}']

    def pshift(self, n):
        if self.f in self.FROZEN_J:
            return lambda k, u: self._ld(SC.OUT / f'out_{n}_shift_h{self.f}.npz')[f'p{k}_{u}']
        return lambda k, u: self._ld(OUT / f'out_{n}_shift_h{self.f}.npz')[f'p{k}_{u}']


def score_fold(U, f, bar):
    """One read step-2 fold: controls (F1 hecate, F2 P0/P1 ARTEFACT), J0 / D0 runs with S.MIN_FIT = the fold's bar,
    J0 frame test, fold_status_jd (WIN = J0 L < D0 L and J0 FOLLOWS)."""
    idx = U.fidx[f]
    O = Fold(f)
    trs = {n: json.loads((HO / f'train_{n}_h{f}.json').read_text()) for n in NAMES}
    ctrl = S4.fold_controls(U, f, idx, O.pshift)
    keep, S.MIN_FIT = S.MIN_FIT, bar
    try:
        runs, floor = S4.score_set(list(NAMES), O.site, trs, idx, U)
    finally:
        S.MIN_FIT = keep
    fc = {n: S.frame_control(lambda k, j, g=O.shift(n): g(k, idx[j]), sub(U.pts, idx), sub(U.flags, idx))
          for n in NAMES if n[0] == 'J' and runs[n]['finite']}
    frame = {n: x['status'] for n, x in fc.items()}
    st = S1.fold_status_jd(ctrl, runs, frame) if len(frame) == 1 else dict(status='VOID')
    rd = {n: S4.auc_ci([O.site(n, u).mean(0) for u in idx], sub(U.pts, idx), sub(U.flags, idx)) for n in NAMES}
    return dict(st, fit_bar=bar, controls=ctrl, runs={n: S1.keep_keys(r) for n, r in runs.items()},
                frame={n: {k: v for k, v in x.items() if k != 'peaks'} for n, x in fc.items()}, mass_floor=floor,
                auc_ci=rd, D0_const=np.load(OUT / f'out_D0_h{f}_shift.npz')['const'].tolist())


def j_reproduces_w028(U):
    """C6: w028's frozen J0 recomputed here (bar 0.6283) equals F-545's results.json (ink AUC, L, L0 within 1e-9;
    fit_ok; fold controls ok)."""
    O, idx = Fold('w028'), U.fidx['w028']
    tr = {'J0': json.loads((HO / 'train_J0_hw028.json').read_text())}
    ctrl = S4.fold_controls(U, 'w028', idx, O.pshift)
    keep, S.MIN_FIT = S.MIN_FIT, SC.bars()['w028']['bar']
    try:
        runs, _ = S4.score_set(['J0'], O.site, tr, idx, U)
    finally:
        S.MIN_FIT = keep
    r = json.loads(F545_RESULTS.read_text())['folds']['w028']['runs']['J0']
    d = dict(ink_auc=abs(runs['J0']['ink_auc'] - r['ink_auc']), L=abs(runs['J0']['label']['L'] - r['label']['L']),
             L0=abs(runs['J0']['label']['L0'] - r['label']['L0']))
    return dict(abs_diff=d, fit_ok=runs['J0']['fit_ok'], controls=ctrl,
                ok=bool(all(v <= 1e-9 for v in d.values()) and runs['J0']['fit_ok'] == r['fit_ok'] is True and ctrl['ok']))


def score_w029(U):
    """w029 recomputed with step 1's own function (J0, J1 F-545; D0 F-567); C5: equals step 1's results.json."""
    idx, ctrl, runs, floor, fc, rd = S1.score_fold(U, with_d=True)
    frame = {n: x['status'] for n, x in fc.items()}
    st = S1.fold_status_jd(ctrl, runs, frame)
    r1 = json.loads(S1_RESULTS.read_text())['fold']
    d = {n: abs(runs[n]['label']['L'] - r1['runs'][n]['label']['L']) for n in ('J0', 'J1', 'D0')}
    c5 = dict(abs_diff_L=d, status=st['status'], status_step1=r1['status'],
              ok=bool(all(v <= 1e-9 for v in d.values()) and st['status'] == r1['status'] and ctrl['ok']))
    return dict(st, fit_bar=SC.bars()['w029']['bar'], controls=ctrl, runs={n: S1.keep_keys(r) for n, r in runs.items()},
                frame={n: {k: v for k, v in x.items() if k != 'peaks'} for n, x in fc.items()}, mass_floor=floor,
                auc_ci=rd), c5


def effect_chain(v):
    """score.verdict_of's effect rules after the gates, verbatim order (seed_stable, e1-e6 + frame, n1/n2)."""
    if not v['seed_stable']:
        return 'INCONCLUSIVE', False
    if all(v[k] for k in ('e1_ratio', 'e2_dL', 'e3_above_noise', 'e4_ci_apart', 'e5_N1_direction', 'e6_equal_auc')):
        if 'frame_follows_all' in v and not v['frame_follows_all']:
            return 'INCONCLUSIVE', True
        return 'LABEL-EFFECT', False
    if v['n1_small'] and v['n2_ratio']:
        return 'NO-LABEL-EFFECT', False
    return 'INCONCLUSIVE', False


def pooled_verdict(ck, runs, frame, seed_noise=SEED_NOISE):
    """score.verdict_of with one seed per arm: run on {J0, J1 := J0, D0, D1 := D0} (spread 0), then the seed-noise
    rules (seed_stable, e3) recomputed with the fixed seed_noise and the effect chain re-run in verdict_of's order."""
    r2 = dict(runs, J1=runs['J0'], D1=runs['D0'])
    arms, v = S.verdict_of(ck, r2, C.ATTEMPT_ARMS[3], {'J0': frame['J0'], 'J1': frame['J0']})
    if 'lower_L_arm' not in v:                                             # a gate decided (VOID / NO-VERDICT / FRAME-ARTEFACT)
        return arms, v
    v['seed_noise'] = seed_noise
    v['seed_noise_source'] = 'F-545 w029 |L(J0) - L(J1)| (fixed; one seed per arm in step 2)'
    v['seed_stable'] = bool(seed_noise <= S.MAX_SEED_PP)
    v['e3_above_noise'] = bool(v['dL'] > S.NOISE_K * seed_noise)
    v.pop('frame_downgrade', None)
    v['verdict'], dg = effect_chain(v)
    if dg:
        v['frame_downgrade'] = True
    return arms, v


def pooled(U, folds, F):
    """Pooled J0 vs D0 over the sites of every scored fold (S.MIN_FIT off: each pooled net passed its own fold bar)."""
    scored = [f for f in FOLDS6 if folds[f].get('scored')]
    if not scored:
        return None
    pidx = [u for f in scored for u in U.fidx[f]]
    fold_of = {u: F[U.keys[u][0]] for u in pidx}
    tr_pool = {n: [json.loads((HO / f'train_{n}_h{f}.json').read_text()) for f in scored] for n in NAMES}
    tr = {n: dict(fit_auc_shell=min(t['fit_auc_shell'] for t in tr_pool[n]), steps=sorted({t['steps'] for t in tr_pool[n]}),
                  seconds=sum(t['seconds'] for t in tr_pool[n]), gpu=tr_pool[n][0]['gpu'], n_params=tr_pool[n][0]['n_params'])
          for n in NAMES}
    keep, S.MIN_FIT = S.MIN_FIT, 0.0
    try:
        runs, floor = S4.score_set(list(NAMES), lambda n, u: fold_of[u].site(n, u), tr, pidx, U)
    finally:
        S.MIN_FIT = keep
    P, Fl = sub(U.pts, pidx), sub(U.flags, pidx)
    pfc = S.frame_control(lambda k, j: fold_of[pidx[j]].shift('J0')(k, pidx[j]), P, Fl)
    pctl = {n: S.frame_control(lambda k, j, n=n: fold_of[pidx[j]].pshift(n)(k, pidx[j]), P, Fl)['status']
            for n in S.FRAME_PC}
    chk = S.checks(sub(U.pts, U.old), sub(U.flags, U.old))
    ck = dict(scorer=chk['scorer']['ok'], pc_hecate=chk['pc_hecate']['ok'],
              frame_pc=all(v == 'ARTEFACT' for v in pctl.values()))
    arms, v = pooled_verdict(ck, runs, {'J0': pfc['status']})
    return dict(folds=scored, n_sites=len(pidx), mass_floor=floor,
                runs={n: {k: r.get(k) for k in ('ink_auc', 'label', 'N1', 'label_xseg_dropped', 'measurable')}
                      for n, r in runs.items()},
                frame={'J0': pfc['status']}, frame_e={'J0': pfc['e']}, p_control=pctl, arms=arms, verdict=v)


def overall6(checks_ok, folds, pooled_v):
    """score_heldout.overall (F-485, unchanged) over its 6 folds: CONFIRMED needs >= 5 of 6 WIN (MIN_WIN 5), every
    scored fold FOLLOWS and the pooled LABEL-EFFECT with J lower. w031 is VOID (not trained), so 5 of 5 others."""
    assert SH.MIN_WIN == 5 and tuple(SH.FOLDS) == tuple(FOLDS6) and set(folds) == set(FOLDS6)
    return SH.overall(checks_ok, folds, pooled_v)


def november_line(verdict, folds, pool):
    """Registered decision rule for the November text (README "Step 2 criteria", N1-N4)."""
    read = [f for f in FOLDS6 if folds[f].get('scored')]
    win = [f for f in read if folds[f]['status'] == 'WIN']
    k, n = len(win), len(read)
    if verdict == 'CONFIRMED':
        r = pool['runs']
        return ('N1', f'At 2.4 um (PHerc1667 level 0), a 3D ink net trained with label J keeps less of its output off '
                      f'the papyrus sheet than the same net trained on the standard slab label (D) on all 5 read '
                      f'held-out folds of 6 (the sixth failed its positive control) (pooled leak L {r["J0"]["label"]["L"]:.3f} vs {r["D0"]["label"]["L"]:.3f}); ink AUC is '
                      f'similar ({r["J0"]["ink_auc"]:.2f} vs {r["D0"]["ink_auc"]:.2f}). One seed per arm; D trained '
                      f'twice as long as J.')
    if verdict == 'VOID':
        return ('N4', 'No J vs D line at 2.4 um (a control failed); keep the F-545 J-only wording.')
    if verdict == 'NOT-CONFIRMED':
        return ('N3', f'At 2.4 um, J vs D is not confirmed: J kept less mass off the sheet on {k} of {n} read folds; no '
                      f'claim that J beats D at level 0.')
    return ('N2', f'At 2.4 um, J kept less mass off the sheet than D on {k} of {n} read folds ({6 - n} of 6 not read); '
                  f'not a confirmed result.')


def synthetic():
    """Rule cases: read_state, overall6, pooled seed-noise override, november_line (6 folds in C.FOLDS order
    w028, w029, w013, w018, w023, w031; w031 is VOID in every case but the first two)."""
    out = {}
    g_ok, g_bad = dict(passed=True), dict(passed=False)
    rep = [dict(key=['w013']), dict(key=['w029'])]
    good = dict(replay={'P0': 0, 'P1': 0, 'J0_hw013': 1e-4, 'D0_hw013': 0}, replay_tol=5e-3, fold='w013',
                nets=['J0', 'D0', 'P0', 'P1'], smoke=0, done=True, not_run=False, sites=320,
                replay_sites={'points_0': True, 'points_1': True, 'tile_0': True})
    cases = {'not_trained': (None, None, 'NOT-RUN'), 'gate1_fail': (g_bad, good, 'VOID'), 'crash': (g_ok, None, 'VOID'),
             'budget_stop': (g_ok, dict(good, done=False, not_run=True), 'NOT-RUN'),
             'not_done': (g_ok, dict(good, done=False), 'VOID'), 'read': (g_ok, good, 'READ'),
             'replay_over': (g_ok, dict(good, replay=dict(good['replay'], D0_hw013=6e-3)), 'VOID'),
             'replay_nan': (g_ok, dict(good, replay=dict(good['replay'], J0_hw013=float('nan'))), 'VOID'),
             'net_missing': (g_ok, dict(good, replay={'P0': 0, 'P1': 0, 'J0_hw013': 0}), 'VOID'),
             'tile_missing': (g_ok, dict(good, replay_sites={'points_0': True, 'points_1': True}), 'VOID'),
             'tile_differs': (g_ok, dict(good, replay_sites={'points_0': True, 'points_1': True, 'tile_0': False}), 'VOID'),
             'smoke': (g_ok, dict(good, smoke=8), 'VOID'), 'wrong_fold': (g_ok, dict(good, fold='w018'), 'VOID'),
             'few_sites': (g_ok, dict(good, sites=8), 'VOID')}
    for k, (g, inf, want) in cases.items():
        got = read_state('w013', g, inf, rep, 320)[0]
        out[f'read_{k}'] = dict(want=want, got=got, ok=got == want)
    rep8 = [dict(key=['w028']), dict(key=['w029'])]
    g28 = dict(good, fold='w028', replay={'D0_hw028': 0.0}, nets=['D0'])
    for k, (inf, want) in {'w028_read': (g28, 'READ'), 'w028_with_P': (dict(g28, replay={'P0': 0, 'P1': 0, 'D0_hw028': 0}), 'VOID'),
                           'w028_wrong_nets': (dict(g28, nets=['J0', 'D0']), 'VOID')}.items():
        got = read_state('w028', g_ok, inf, rep8, 320)[0]
        out[f'read_{k}'] = dict(want=want, got=got, ok=got == want)
    W, L, U, A, V, N, R = ({'status': s, 'scored': s in ('WIN', 'LOSS', 'UNCLEAR', 'ARTEFACT'), 'follows': s == 'WIN'}
                           for s in ('WIN', 'LOSS', 'UNCLEAR', 'ARTEFACT', 'VOID', 'NO-VERDICT', 'NOT-RUN'))
    pj, pd, pi = dict(verdict='LABEL-EFFECT', lower_L_arm='J'), dict(verdict='LABEL-EFFECT', lower_L_arm='D'), \
        dict(verdict='INCONCLUSIVE', lower_L_arm='J')
    oc = {'6win': ([W] * 6, pj, True, 'CONFIRMED'),
          '5win': ([W] * 5 + [V], pj, True, 'CONFIRMED'), '5win_pooled_inconcl': ([W] * 5 + [V], pi, True, 'NOT-CONFIRMED'),
          '5win_pooled_D': ([W] * 5 + [V], pd, True, 'NOT-CONFIRMED'), '4win1loss': ([W] * 4 + [L, V], pj, True, 'NOT-CONFIRMED'),
          '4win1unclear': ([W] * 4 + [U, V], pj, True, 'NOT-CONFIRMED'), '4win1artefact': ([W] * 4 + [A, V], pj, True, 'NOT-CONFIRMED'),
          '4win1gate1void': ([W] * 4 + [V, V], pj, True, 'NO-VERDICT'), '4win1nv': ([W] * 4 + [N, V], pj, True, 'NO-VERDICT'),
          '4win1notrun': ([W] * 4 + [R, V], pj, True, 'INCOMPLETE'), '3win1void1notrun': ([W] * 3 + [V, R, V], pj, True, 'NO-VERDICT'),
          '5win_checks_fail': ([W] * 5 + [V], pj, False, 'VOID'), 'w029win_only': ([V, W] + [V] * 4, pj, True, 'NO-VERDICT')}
    for k, (st, pv, ck, want) in oc.items():
        got = overall6(ck, dict(zip(FOLDS6, st)), pv)
        out[f'overall_{k}'] = dict(want=want, got=got, ok=got == want)
    # pooled seed-noise override: a synthetic verdict_of dict after the gates
    base = dict(lower_L_arm='J', dL=0.20, L_ratio=5.0, seed_noise=0.0, seed_stable=True, e1_ratio=True, e2_dL=True,
                e3_above_noise=True, e4_ci_apart=True, e5_N1_direction=True, e6_equal_auc=True, n1_small=False,
                n2_ratio=False, frame_follows_all=True)
    ec = {'effect': (base, 'LABEL-EFFECT'), 'frame_down': (dict(base, frame_follows_all=False), 'INCONCLUSIVE'),
          'unstable': (dict(base, seed_stable=False), 'INCONCLUSIVE'),
          'no_e3_null': (dict(base, e3_above_noise=False, n1_small=True, n2_ratio=True), 'NO-LABEL-EFFECT'),
          'no_e4': (dict(base, e4_ci_apart=False), 'INCONCLUSIVE')}
    for k, (v, want) in ec.items():
        got = effect_chain(v)[0]
        out[f'chain_{k}'] = dict(want=want, got=got, ok=got == want)
    # e3 with the fixed noise: dL 0.03 > 2 x 0.0116 -> e3 true; dL 0.02 -> false (e2 false anyway)
    out['e3_fixed_noise'] = dict(want=[True, False], got=[0.03 > S.NOISE_K * SEED_NOISE, 0.02 > S.NOISE_K * SEED_NOISE],
                                 ok=(0.03 > S.NOISE_K * SEED_NOISE) and not (0.02 > S.NOISE_K * SEED_NOISE))
    pool = dict(runs={n: dict(label=dict(L=x), ink_auc=0.8) for n, x in (('J0', 0.05), ('D0', 0.28))})
    for k, (vd, st, want) in {'conf': ('CONFIRMED', [W] * 5 + [V], 'N1'), 'void': ('VOID', [W] * 5 + [V], 'N4'),
                              'notconf': ('NOT-CONFIRMED', [W] * 4 + [L, V], 'N3'), 'nv': ('NO-VERDICT', [W] * 4 + [V, V], 'N2'),
                              'inc': ('INCOMPLETE', [W] * 4 + [R, V], 'N2')}.items():
        got = november_line(vd, dict(zip(FOLDS6, st)), pool)[0]
        out[f'november_{k}'] = dict(want=want, got=got, ok=got == want)
    for k, v in out.items():
        print(f'synthetic {k}: {v["got"]} (want {v["want"]})')
    return dict(cases=out, ok=all(v['ok'] for v in out.values()))


def pooled_identity(U):
    """pooled_verdict with seed_noise 0 equals score.verdict_of on duplicated seeds (w029 J0 / D0 runs, real data)."""
    O = Fold('w029')
    idx = U.fidx['w029']
    trs = {n: json.loads((HO / f'train_{n}_hw029.json').read_text()) for n in NAMES}
    keep, S.MIN_FIT = S.MIN_FIT, 0.0
    try:
        runs, _ = S4.score_set(list(NAMES), O.site, trs, idx, U)
    finally:
        S.MIN_FIT = keep
    ck = dict(scorer=True, pc_hecate=True, frame_pc=True)
    _, a = pooled_verdict(ck, runs, {'J0': 'FOLLOWS'}, seed_noise=0.0)
    _, b = S.verdict_of(ck, dict(runs, J1=runs['J0'], D1=runs['D0']), C.ATTEMPT_ARMS[3], {'J0': 'FOLLOWS', 'J1': 'FOLLOWS'})
    _, c = pooled_verdict(ck, runs, {'J0': 'FOLLOWS'})
    return dict(fixed0=a['verdict'], verdict_of=b['verdict'], registered=c['verdict'], dL=c['dL'],
                ok=bool(a['verdict'] == b['verdict'] and a['e3_above_noise'] == b['e3_above_noise']))


def ledger():
    """GPU hours of this task: bars unit + inference smoke + every unit.sh unit (OUT/ledger.jsonl)."""
    h = json.loads(BARS.read_text())['seconds'] / 3600
    sm = OUT.parent / 'smoke' / 'infer_w029.json'
    h += json.loads(sm.read_text())['total_seconds'] / 3600 if sm.exists() else 0.0
    lf = OUT / 'ledger.jsonl'
    rows = [json.loads(x) for x in lf.read_text().splitlines() if x.strip()] if lf.exists() else []
    return dict(bars_h=round(h, 3), units=rows, gpu_h=round(h + sum(r['seconds'] for r in rows) / 3600, 3), cap_h=GPU_CAP_H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--gate1')
    a = ap.parse_args()
    if a.gate1:
        assert a.gate1 in NEW
        g = gate1(a.gate1)
        if g is None:
            print(f'GATE1 {a.gate1}: a net is not trained (NOT-RUN)')
            sys.exit(3)
        (HERE / f'gate1_{a.gate1}.json').write_text(json.dumps(g, indent=1))
        print(json.dumps(g, indent=1))
        sys.exit(0 if g['passed'] else 10)
    U = S4.Sites()
    b = bars()
    res = dict(task='VES-3416', finding='F-567', step=2, scan='PHerc1667 20251217075048 level 0 (2.403 um, 77 keV, 0.2 m)',
               folds_order=list(NEW), steps=STEPS, bars={f: b[f] for f in FOLDS6})
    res['gates'] = dict(power=bool(U.info['power']['ok']),
                        full_draw=bool(U.info['limit'] is None and U.prep['limit_blocks'] is None and not U.prep['smoke']
                                       and not C.SMOKE))
    res['checks'] = S.checks(sub(U.pts, U.old), sub(U.flags, U.old))      # C1 scorer + C2 pooled hecate PC
    res['n_sites'] = {f: dict(label=sum(U.flags[u] for u in U.fidx[f]), N1=sum(not U.flags[u] for u in U.fidx[f]))
                      for f in FOLDS6}
    w029, c5 = score_w029(U)
    res['c5_w029_equals_step1'] = c5
    if a.check:
        res['hecate_pc_new'] = {f: hecate_pc(U, U.fidx[f]) for f in ('w013', 'w018', 'w023', 'w031')}
        res['c6_w028_J0_equals_F545'] = j_reproduces_w028(U)
        res['w031_f1_fails'] = bool(not res['hecate_pc_new']['w031']['ok'])           # the reason w031 is not trained
        res['hecate_pc_w029_equals_fold_controls'] = bool(abs(hecate_pc(U, U.fidx['w029'])['L'] - w029['controls']['pc_hecate']['L']) <= 1e-12)
        res['bars_controls'] = {k: json.loads(BARS.read_text())[k] for k in ('B1', 'B2', 'ok')}
        res['synthetic'] = synthetic()
        res['pooled_identity'] = pooled_identity(U)
        res['seed_noise_from_F545'] = abs(SEED_NOISE - abs(w029['runs']['J0']['label']['L'] - w029['runs']['J1']['label']['L'])) <= 1e-8
        res['ledger'] = ledger()
        res['check_ok'] = bool(all(res['gates'].values()) and res['checks']['ok'] and c5['ok']
                               and all(res['hecate_pc_new'][f]['ok'] for f in ('w013', 'w018', 'w023'))
                               and res['c6_w028_J0_equals_F545']['ok'] and res['w031_f1_fails']
                               and res['hecate_pc_w029_equals_fold_controls'] and res['bars_controls']['ok']
                               and res['synthetic']['ok'] and res['pooled_identity']['ok'] and res['seed_noise_from_F545'])
        CHECK.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: res[k] for k in ('gates', 'c5_w029_equals_step1', 'c6_w028_J0_equals_F545', 'hecate_pc_new',
                                              'w031_f1_fails', 'pooled_identity',
                                              'seed_noise_from_F545', 'check_ok')}, indent=1, default=float))
        return
    folds, F = {'w029': w029, 'w031': dict(status='VOID', reason=NOT_TRAINED['w031'])}, {'w029': Fold('w029')}
    res['gate1'], res['read_state'] = {}, {}
    for f in NEW:
        g = gate1(f)
        lf = OUT / f'infer_{f}.json'
        inf = json.loads(lf.read_text()) if lf.exists() else None
        state, why = read_state(f, g, inf, U.prep['replay'], len(U.fidx[f]))
        res['gate1'][f], res['read_state'][f] = g, dict(state=state, why=why)
        if state == 'READ':
            folds[f], F[f] = score_fold(U, f, b[f]['bar']), Fold(f)
        else:
            assert why != 'gate1' or not (OUT / f'out_D0_h{f}.npz').exists(), f'{f}: site output exists although gate 1 failed'
            folds[f] = dict(status=state if state == 'NOT-RUN' else 'VOID', reason=why)
        print(f'fold {f}: {folds[f]["status"]}', flush=True)
    res['folds'] = folds
    pool = pooled(U, folds, F)
    res['pooled'] = pool
    pooled_v = pool['verdict'] if pool else None
    checks_ok = bool(all(res['gates'].values()) and res['checks']['ok'] and c5['ok']
                     and (pooled_v is None or pooled_v.get('frame_pc', True)))
    res['checks_ok'] = checks_ok
    res['verdict'] = overall6(checks_ok, folds, pooled_v)
    res['november'] = november_line(res['verdict'], folds, pool)
    read = [f for f in FOLDS6 if folds[f].get('scored')]
    res['readings'] = dict(
        n_read=len(read), n_win=sum(folds[f]['status'] == 'WIN' for f in read),
        per_fold={f: dict(status=folds[f]['status'], L_J=folds[f].get('L_J'), L_D=folds[f].get('L_D'),
                          J_minus_D=(folds[f]['L_J'] - folds[f]['L_D']) if folds[f].get('L_D') is not None else None,
                          D0_L_over_L0=(folds[f]['runs']['D0']['label']['L'] / folds[f]['runs']['D0']['label']['L0'])
                          if folds[f].get('runs') and folds[f]['runs']['D0']['label'] and folds[f]['runs']['D0']['label']['L0'] else None)
                  for f in read},
        baselines=dict(chance_auc=0.5, F418_L2_D_L='~L0 (0.19-0.24)', step1_w029=dict(J_L=0.048, D0_L=0.280)))
    res['ledger'] = ledger()
    RESULTS.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res.get(k) for k in ('read_state', 'readings', 'checks_ok', 'verdict', 'november')}, indent=1,
                     default=float))
    if pool:
        print('pooled:', {k: pooled_v.get(k) for k in ('verdict', 'lower_L_arm', 'dL', 'L_ratio', 'seed_noise')})


if __name__ == '__main__':
    main()
