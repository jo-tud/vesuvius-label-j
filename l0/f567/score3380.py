"""VES-3380 (F-485 J vs D, attempt 3) scorer = ../2026-11-07-ves3347-f485-jd-folds/score3347.py with one registered
change: D0_hw029 is trained for STEPS = 24 000 steps (was 12 000; README "Criteria"). Paths: VES3380_OUT, and C.DATA =
VES2578_RUN = the ves3380 overlay. Every rule below is score3347's, unchanged. CPU only.
Every frozen rule function is imported unchanged: F-485 step1/score.py (score_run, frame_control, checks),
attempt4/score4.py (Sites, fold_controls, score_set, auc_ci), heldout/score_heldout.py (the fold rule, mirrored here
for one D seed and checked against it), and F-545's score3226.py (bars = the F-538 per-fold fit bar, loader /
p_shift = the F-545 J and P outputs of w029, read from LABELJ_DATA/ves3226). New here, all written before the D0
net exists:
  - gate1: D0_hw029's fit AUC (train json fit_auc_shell) >= the w029 per-fold bar (F-538 P0_on_w029 - 0.03 =
    0.6961); a miss stops the task before any held-out output (no read);
  - fold_status_jd: score_heldout.fold_status with the D mean over the D nets that exist (D0 only on w029): WIN =
    mean J L < mean D L and both J FOLLOW (F-485 fold WIN rule). With D0 and D1 it equals fold_status (checked);
  - step1_outcome: GO-STEP2 only if gate1 passes and w029 is WIN (w028 has no D net under its bar (F-496 0.605 <
    0.628), so CONFIRMED (>= 5 of 6 WIN) needs every other fold to WIN; a non-WIN w029 makes it unreachable).

  python score3380.py --check   gates without outputs, scorer + hecate checks, J reproduction of F-545 (w029),
                                synthetic fold-rule / outcome cases -> check.json; no D0 output read
  python score3380.py --gate1   gate1 only (after training, before inference) -> gate1.json; exit 0 pass, 10 fail
  python score3380.py           results.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np

HERE = Path(__file__).resolve().parent
STEP1 = HERE.parent / 'f485' / 'step1'
F545 = HERE.parent / 'f545'
for p in (STEP1, STEP1 / 'heldout', STEP1 / 'attempt4', F545):
    sys.path.insert(0, str(p))
import common as C  # noqa: E402
import score as S  # noqa: E402
import score_heldout as SH  # noqa: E402
import score4 as S4  # noqa: E402
import score3226 as SC  # noqa: E402

OUT = Path(os.environ.get('VES3380_OUT', _VD + '/ves3380/out'))
STEPS = 24000                                             # attempt 3: the one registered change (was 12000)
HO = C.DATA / 'heldout'
FOLD = 'w029'
JN, DN = ('J0', 'J1'), ('D0',)
EXPECT_KEPT = sorted(set(C.FOLDS) - {FOLD})
RESULTS, CHECK, GATE1 = HERE / 'results.json', HERE / 'check.json', HERE / 'gate1.json'
F545_RESULTS = F545 / 'results.json'


def sub(xs, idx):
    return [xs[i] for i in idx]


def gate1():
    """D0_hw029 is the registered run (fold, arm, seed, steps, patch, batch, kept segments) and its fit AUC >= the
    fold's F-538 bar. Returns the gate dict (pass = registered and fit_ok)."""
    b = SC.bars()[FOLD]
    tr = json.loads((HO / f'train_D0_h{FOLD}.json').read_text())
    reg = bool((tr['fold'], tr['arm'], tr['seed'], tr['steps'], tr['patch'], tr['batch']) == (FOLD, 'D', 0, STEPS, 128, 16)
               and tr['kept_segments'] == EXPECT_KEPT and tr['limit_blocks'] is None)
    fit = float(tr['fit_auc_shell'])
    return dict(net='D0_hw029', fit_auc_shell=fit, P0_on_draw=b['P0_on_draw'], bar=b['bar'], margin=fit - b['bar'],
                registered=reg, fit_ok=bool(np.isfinite(fit) and fit >= b['bar']),
                passed=bool(reg and np.isfinite(fit) and fit >= b['bar']),
                baselines=dict(chance=0.5, F560_D0_w029_12k=0.6869, F496_D0_w028=0.605, J0_w029=0.737, J1_w029=0.713, P0_on_w029=b['P0_on_draw']),
                seconds=tr['seconds'], gpu=tr['gpu'])


def fold_status_jd(ctrl, runs, frame):
    """score_heldout.fold_status (F-485, frozen) with the J and D means over the J and D runs present."""
    if not ctrl['ok'] or not all(r['fit_ok'] for r in runs.values()):
        return dict(status='VOID')
    if not all(r['measurable'] for r in runs.values()):
        return dict(status='NO-VERDICT')
    J = sorted(n for n in runs if n[0] == 'J')
    D = sorted(n for n in runs if n[0] == 'D')
    assert J and D and set(frame) == set(J)
    if any(frame[n] == 'ARTEFACT' for n in J):
        return dict(status='ARTEFACT', scored=True, win=False, follows=False)
    LJ = float(np.mean([runs[n]['label']['L'] for n in J]))
    LD = float(np.mean([runs[n]['label']['L'] for n in D]))
    win, follows = LJ < LD, all(frame[n] == 'FOLLOWS' for n in J)
    st = 'WIN' if win and follows else 'LOSS' if not win else 'UNCLEAR'
    return dict(status=st, scored=True, win=bool(win), follows=bool(follows), L_J=LJ, L_D=LD)


def step1_outcome(checks_ok, g1, st):
    """STOP-FIT (D0 under its fit bar, no read) -> VOID (a gate / control fails) -> GO-STEP2 (w029 WIN) ->
    STOP-<status> (LOSS / UNCLEAR / ARTEFACT / NO-VERDICT / fold VOID / NOT-RUN)."""
    if not g1['passed']:
        return 'STOP-FIT'
    if not checks_ok:
        return 'VOID'
    return 'GO-STEP2' if st == 'WIN' else f'STOP-{st}'


def synthetic_rules():
    ok_c, bad_c = dict(ok=True), dict(ok=False)

    def run(L, fit=True, meas=True):
        return dict(fit_ok=fit, measurable=meas, label=dict(L=L, L0=0.25))
    F, A, U = 'FOLLOWS', 'ARTEFACT', 'UNCLEAR'
    cases = {'win': (ok_c, [0.05, 0.04], [0.20], [F, F], 'WIN'),
             'loss': (ok_c, [0.05, 0.04], [0.03], [F, F], 'LOSS'),
             'tie_loss': (ok_c, [0.05, 0.05], [0.05], [F, F], 'LOSS'),
             'mean_loss': (ok_c, [0.01, 0.30], [0.15], [F, F], 'LOSS'),
             'unclear': (ok_c, [0.05, 0.04], [0.20], [F, U], 'UNCLEAR'),
             'artefact': (ok_c, [0.05, 0.04], [0.20], [F, A], 'ARTEFACT'),
             'ctrl_fail': (bad_c, [0.05, 0.04], [0.20], [F, F], 'VOID'),
             'fit_fail_D': (ok_c, [0.05, 0.04], [(0.20, False, True)], [F, F], 'VOID'),
             'floor_fail_D': (ok_c, [0.05, 0.04], [(0.20, True, False)], [F, F], 'NO-VERDICT')}
    out = {}
    for k, (c, lj, ld, fr, want) in cases.items():
        runs = {f'J{i}': run(x) for i, x in enumerate(lj)}
        runs.update({f'D{i}': run(*(x if isinstance(x, tuple) else (x,))) for i, x in enumerate(ld)})
        got = fold_status_jd(c, runs, {f'J{i}': x for i, x in enumerate(fr)})['status']
        out[f'fold_{k}'] = dict(want=want, got=got, ok=got == want)
    # equivalence with the frozen fold_status when both D seeds exist (24 random cases)
    rng = np.random.default_rng(3347)
    eq = []
    for _ in range(24):
        runs = {n: run(float(rng.uniform(0, 0.3)), bool(rng.random() > 0.1), bool(rng.random() > 0.1))
                for n in ('J0', 'J1', 'D0', 'D1')}
        fr = {n: str(rng.choice([F, F, F, A, U])) for n in ('J0', 'J1')}
        c = ok_c if rng.random() > 0.1 else bad_c
        eq.append(fold_status_jd(c, runs, fr)['status'] == SH.fold_status(c, runs, fr)['status'])
    out['equiv_fold_status_2D'] = dict(want=24, got=int(sum(eq)), ok=all(eq))
    g_ok, g_bad = dict(passed=True), dict(passed=False)
    for k, (c, g, st, want) in {'go': (True, g_ok, 'WIN', 'GO-STEP2'), 'loss': (True, g_ok, 'LOSS', 'STOP-LOSS'),
                                'unclear': (True, g_ok, 'UNCLEAR', 'STOP-UNCLEAR'),
                                'nv': (True, g_ok, 'NO-VERDICT', 'STOP-NO-VERDICT'),
                                'fit': (True, g_bad, None, 'STOP-FIT'), 'void': (False, g_ok, 'WIN', 'VOID'),
                                'notrun': (True, g_ok, 'NOT-RUN', 'STOP-NOT-RUN')}.items():
        got = step1_outcome(c, g, st)
        out[f'outcome_{k}'] = dict(want=want, got=got, ok=got == want)
    for k, v in out.items():
        print(f'synthetic {k}: {v["got"]} (want {v["want"]})')
    return dict(cases=out, ok=all(v['ok'] for v in out.values()))


def infer_gates(inf, replay):
    """G3 / G2 / G4 on an infer.json dict (replay = prep.json 'replay' list). G3: every replay point, and a tile per
    replay site of the inferred folds; G2: D0_hw029 replayed, finite, <= tol; G4: either the full read (folds [w029],
    nets {w029: [D0]}) or the budget stop before the site pass (folds [], not_run [w029]): R9 VES-3351 1i, a budget
    stop gives STOP-NOT-RUN, not VOID (score4 / F-545 form)."""
    rs = inf['replay_sites']
    g = dict(replay_sites=bool(all(rs.values()) and sum(k.startswith('points_') for k in rs) == len(replay)
                               and sum(k.startswith('tile_') for k in rs) ==
                               sum(r['key'][0] in inf['folds'] for r in replay)))
    g['replay_nets'] = bool(set(inf['replay']) == {f'{n}_h{FOLD}' for n in DN} and
                            all(np.isfinite(v) and v <= inf['replay_tol'] for v in inf['replay'].values()))
    g['nets_as_registered'] = bool(inf['folds'] in ([FOLD], []) and inf['nets'] == {f: list(DN) for f in inf['folds']}
                                   and (inf['folds'] == [FOLD] or inf['not_run'] == [FOLD]))
    return g


def gates(U, inf):
    """G1 power and full draw; with infer.json: infer_gates (G2-G4)."""
    g = dict(power=bool(U.info['power']['ok']), full_draw=bool(U.info['limit'] is None and U.prep['limit_blocks'] is None
                                                               and not U.prep['smoke'] and not C.SMOKE))
    if inf is not None:
        g.update(infer_gates(inf, U.prep['replay']))
    return g


def synthetic_gates():
    """infer_gates on synthetic infer.json dicts (R9 VES-3351 1i): full read, budget stop, and the failure cases;
    the budget stop must then give outcome STOP-NOT-RUN."""
    replay = [dict(key=['w029']), dict(key=['w029']), dict(key=['w028'])]
    pts = {f'points_{i}': True for i in range(3)}
    tiles = {'tile_0': True, 'tile_1': True}
    rep = {'D0_hw029': 1e-4}

    def inf(folds, nets, not_run=(), rs=None, rp=None):
        return dict(replay_sites=dict(pts, **tiles) if rs is None else rs, replay=rep if rp is None else rp,
                    replay_tol=5e-3, folds=list(folds), nets=nets, not_run=list(not_run))
    full, stop = inf([FOLD], {FOLD: ['D0']}), inf([], {}, [FOLD], rs=pts)
    cases = {'full': (full, True), 'budget_stop': (stop, True),
             'stop_with_tiles': (inf([], {}, [FOLD]), False),
             'crash_no_not_run': (inf([], {}, rs=pts), False),
             'missing_tile': (inf([FOLD], {FOLD: ['D0']}, rs=dict(pts, tile_0=True)), False),
             'tile_differs': (inf([FOLD], {FOLD: ['D0']}, rs=dict(pts, tile_0=True, tile_1=False)), False),
             'over_tol': (inf([FOLD], {FOLD: ['D0']}, rp={'D0_hw029': 6e-3}), False),
             'nan': (inf([FOLD], {FOLD: ['D0']}, rp={'D0_hw029': float('nan')}), False),
             'extra_net': (inf([FOLD], {FOLD: ['D0']}, rp=dict(rep, D1_hw029=0.0)), False),
             'wrong_site_nets': (inf([FOLD], {FOLD: ['D0', 'J0']}), False),
             'wrong_fold': (inf(['w028'], {'w028': ['D0']}), False)}
    out = {}
    for k, (x, want) in cases.items():
        got = all(infer_gates(x, replay).values())
        out[k] = dict(want=want, got=got, ok=got == want)
    got = step1_outcome(all(infer_gates(stop, replay).values()), dict(passed=True), 'NOT-RUN')
    out['budget_stop_outcome'] = dict(want='STOP-NOT-RUN', got=got, ok=got == 'STOP-NOT-RUN')
    for k, v in out.items():
        print(f'synthetic gate {k}: {v["got"]} (want {v["want"]})')
    return dict(cases=out, ok=all(v['ok'] for v in out.values()))


def score_fold(U, with_d):
    """Runs, frame, controls of w029: J from F-545's outputs (ves3226), D0 from ours; S.MIN_FIT = the w029 bar."""
    idx = U.fidx[FOLD]
    jsite, jshift = SC.loader(FOLD)
    dz = np.load(OUT / f'out_D0_h{FOLD}.npz') if with_d else None
    get = lambda n, u: jsite(n, u) if n[0] == 'J' else dz[f'p_{u}']      # noqa: E731
    names = list(JN) + (list(DN) if with_d else [])
    trs = {n: json.loads((HO / f'train_{n}_h{FOLD}.json').read_text()) for n in names}
    ctrl = S4.fold_controls(U, FOLD, idx, SC.p_shift(U, [FOLD]))          # hecate L <= 0.5 L0; P0, P1 ARTEFACT
    keep, S.MIN_FIT = S.MIN_FIT, SC.bars()[FOLD]['bar']
    try:
        runs, floor = S4.score_set(names, get, trs, idx, U)
    finally:
        S.MIN_FIT = keep
    fc = {n: S.frame_control(lambda k, j, g=jshift(n): g(k, idx[j]), sub(U.pts, idx), sub(U.flags, idx))
          for n in JN if runs[n]['finite']}
    rd = {n: S4.auc_ci([get(n, u).mean(0) for u in idx], sub(U.pts, idx), sub(U.flags, idx)) for n in names}
    return idx, ctrl, runs, floor, fc, rd


def j_reproduces(runs, ctrl):
    """C4: the J numbers recomputed here equal F-545's results.json (w029): ink AUC, L, L0, fit_ok; controls ok."""
    r = json.loads(F545_RESULTS.read_text())['folds'][FOLD]
    d = {n: dict(ink_auc=abs(runs[n]['ink_auc'] - r['runs'][n]['ink_auc']),
                 L=abs(runs[n]['label']['L'] - r['runs'][n]['label']['L']),
                 L0=abs(runs[n]['label']['L0'] - r['runs'][n]['label']['L0'])) for n in JN}
    ok = all(v <= 1e-9 for x in d.values() for v in x.values()) and ctrl['ok'] == r['controls']['ok'] and \
        all(runs[n]['fit_ok'] == r['runs'][n]['fit_ok'] for n in JN)
    return dict(abs_diff=d, ok=bool(ok))


def keep_keys(r):
    return {k: r.get(k) for k in ('fit_auc_shell', 'finite', 'fit_ok', 'measurable', 'ink_auc', 'label', 'N1', 'phase',
                                  'label_xseg_dropped', 'share_def')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--gate1', action='store_true')
    a = ap.parse_args()
    if a.gate1:
        g1 = gate1()
        GATE1.write_text(json.dumps(g1, indent=1))
        print(json.dumps(g1, indent=1))
        sys.exit(0 if g1['passed'] else 10)
    U = S4.Sites()
    inf = json.loads((OUT / 'infer.json').read_text()) if (OUT / 'infer.json').exists() and not a.check else None
    res = dict(task='VES-3380', attempt=3, step=1, steps_D=STEPS, scan='PHerc1667 20251217075048 level 0 (2.403 um, 77 keV, 0.2 m)',
               fold=FOLD, gates=gates(U, inf))
    res['checks'] = S.checks(sub(U.pts, U.old), sub(U.flags, U.old))     # C1 scorer + C2 pooled hecate PC (old draw)
    res['n_sites'] = dict(label=sum(U.flags[u] for u in U.fidx[FOLD]), N1=sum(not U.flags[u] for u in U.fidx[FOLD]))
    if a.check:
        _, ctrl, runs, _, _, _ = score_fold(U, with_d=False)
        res['controls'] = ctrl
        res['j_reproduces_F545'] = j_reproduces(runs, ctrl)
        res['synthetic_rules'] = synthetic_rules()
        res['synthetic_gates'] = synthetic_gates()
        res['bar'] = SC.bars()[FOLD]
        res['check_ok'] = bool(res['checks']['ok'] and res['synthetic_rules']['ok'] and res['synthetic_gates']['ok']
                               and res['j_reproduces_F545']['ok']
                               and ctrl['ok'] and all(res['gates'].values()))
        CHECK.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(dict(gates=res['gates'], bar=res['bar'], controls=ctrl, j_reproduces=res['j_reproduces_F545'],
                              check_ok=res['check_ok']), indent=1, default=float))
        return
    g1 = gate1()
    res['gate1'] = g1
    if not g1['passed']:                                                   # no held-out output may exist
        assert not (OUT / f'out_D0_h{FOLD}.npz').exists(), 'D0 site output exists although gate1 failed'
        res['fold_status'] = 'NOT-SCORED'
        res['outcome'] = step1_outcome(True, g1, None)
        RESULTS.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: res[k] for k in ('gate1', 'outcome')}, indent=1, default=float))
        return
    assert inf is not None, 'no infer.json: run infer3380.py first'
    done = FOLD in inf['folds']
    gate_ok = all(res['gates'].values()) and len(res['gates']) == 5
    if not done:
        st = dict(status='NOT-RUN')
        c4 = dict(ok=True)
    else:
        idx, ctrl, runs, floor, fc, rd = score_fold(U, with_d=True)
        frame = {n: x['status'] for n, x in fc.items()}
        st = fold_status_jd(ctrl, runs, frame) if len(frame) == len(JN) else dict(status='VOID')
        c4 = j_reproduces(runs, ctrl)
        D = runs['D0']
        res['fold'] = dict(st, fit_bar=SC.bars()[FOLD]['bar'], controls=ctrl,
                           runs={n: keep_keys(r) for n, r in runs.items()},
                           frame={n: {k: v for k, v in x.items() if k != 'peaks'} for n, x in fc.items()},
                           mass_floor=floor, auc_ci=rd)
        res['readings'] = dict(
            D0_L=D['label']['L'], D0_L0=D['label']['L0'],
            D0_L_over_L0=(D['label']['L'] / D['label']['L0']) if D['label']['L'] is not None and D['label']['L0'] else None,
            J_minus_D_L=(st.get('L_J') - st.get('L_D')) if st.get('L_D') is not None else None,
            D0_const=np.load(OUT / f'out_D0_h{FOLD}_shift.npz')['const'].tolist(),
            baselines=dict(chance_auc=0.5, F418_L2_D_L='~L0 (0.19-0.24)', F418_L2_J_w029_L=[0.031, 0.020],
                           hecate_w029_L=ctrl['pc_hecate']['L'], hecate_w029_auc=ctrl['pc_hecate']['ink_auc']))
    res['j_reproduces_F545'] = c4
    checks_ok = bool(gate_ok and res['checks']['ok'] and c4['ok'])
    res['checks_ok'] = checks_ok
    res['fold_status'] = st['status']
    res['outcome'] = step1_outcome(checks_ok, g1, st['status'])
    res['gpu'] = dict(train_seconds=g1['seconds'], infer_seconds=inf['total_seconds'], gpu=inf['gpu'],
                      gpu_h=round((g1['seconds'] + inf['total_seconds']) / 3600, 3))
    RESULTS.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res.get(k) for k in ('gates', 'gate1', 'fold_status', 'readings', 'checks_ok', 'outcome')},
                     indent=1, default=float))


if __name__ == '__main__':
    main()
