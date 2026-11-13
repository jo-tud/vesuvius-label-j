"""VES-3226 scorer: F-485 attempt-2 folds w029, w028 re-gated with the F-538 per-fold fit bar, then the held-out J
read (README "Criteria"). CPU only. Every frozen F-485 rule function is imported unchanged from step1/: score.py
(score_run, frame_control, checks, summarise, leak_site), heldout/score_heldout.py (MASS_PER_SITE, PC_FOLD_RATIO),
attempt4/score4.py (Sites, fold_controls, score_set, auc_ci). New here, all written before any held-out output:
  - the per-fold fit bar (regate): S.MIN_FIT = P0's F-538 AUC on the fold's 5-segment draw - 0.03, set per fold
    (the only rule change; score_run reads S.MIN_FIT for fit_ok);
  - the J-only fold status and overall verdict (fold_status_j, overall_j): no D net passes the fit bar on these folds
    (w029: none trained; w028: F-496 D0 0.605 < 0.628), so J vs D is not tested; J is read against its own
    depth-blind null L0 (F-485 task ratio 0.5) and the frame test;
  - data access for infer.py's outputs (OUT) and its gates.

  python score3226.py --check   regate, gates without outputs, scorer check, pooled + per-fold hecate PC, synthetic
                            fold-rule / verdict / frame cases -> check.json; no J or P output read
  python score3226.py       results.json
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
for p in (STEP1, STEP1 / 'heldout', STEP1 / 'attempt4'):
    sys.path.insert(0, str(p))
import common as C  # noqa: E402
import score as S  # noqa: E402
import score_heldout as SH  # noqa: E402
import score4 as S4  # noqa: E402

OUT = Path(os.environ.get('VES3226_OUT', _VD + '/ves3226'))
HO = C.DATA / 'heldout'
FOLDS = ('w029', 'w028')
NETS = {'w029': ('J0', 'J1'), 'w028': ('J0',)}          # = infer.NETS (asserted against infer.json)
BAR_TOL = 0.03                                           # F-538: pass if within 0.03 below P0 on the fold's draw
F538 = HERE.parent / 'f538' / 'results.json'
F538_KEY = {'w029': 'P0_on_w029', 'w028': 'P0_on_w028'}
TASK_RATIO = S4.TASK_RATIO                               # 0.5: J L <= 0.5 x J L0 (F-485 task bar)
RESULTS, CHECK = HERE / 'results.json', HERE / 'check.json'


def sub(xs, idx):
    return [xs[i] for i in idx]


def bars():
    m = json.loads(F538.read_text())['matrix']
    return {f: dict(P0_on_draw=m[F538_KEY[f]], bar=m[F538_KEY[f]] - BAR_TOL) for f in FOLDS}


def regate():
    """Per-fold fit bar on the frozen nets' training json fit_auc_shell (known before these criteria: README)."""
    b, out = bars(), {}
    for f in FOLDS:
        nets = {}
        for n in NETS[f]:
            fit = json.loads((HO / f'train_{n}_h{f}.json').read_text())['fit_auc_shell']
            nets[n] = dict(fit_auc_shell=fit, pass_fold_bar=bool(fit >= b[f]['bar']), pass_old_080=bool(fit >= 0.80))
        out[f] = dict(b[f], nets=nets, all_pass=all(v['pass_fold_bar'] for v in nets.values()))
    return out


def fold_status_j(ctrl, runs, frame):
    """VOID: a fold control fails, an output is non-finite or a J net is under the fold's fit bar; NO-VERDICT: a J net
    under a floor (ink AUC 0.60, mass, defined share, L defined); ARTEFACT: a J net ARTEFACT; else J-PASS (every J
    net FOLLOWS and mean J L <= 0.5 x mean J L0), J-LEAK (mean J L > 0.5 x mean J L0), UNCLEAR (on the sheet but a
    J net NO-PEAK / UNCLEAR)."""
    if not ctrl['ok'] or not all(r['fit_ok'] for r in runs.values()):
        return dict(status='VOID')
    if not all(r['measurable'] for r in runs.values()):
        return dict(status='NO-VERDICT')
    if any(v == 'ARTEFACT' for v in frame.values()):
        return dict(status='ARTEFACT', scored=True)
    LJ = float(np.mean([r['label']['L'] for r in runs.values()]))
    L0J = float(np.mean([r['label']['L0'] for r in runs.values()]))
    on_sheet = bool(LJ <= TASK_RATIO * L0J)
    follows = all(v == 'FOLLOWS' for v in frame.values())
    st = 'J-PASS' if on_sheet and follows else 'J-LEAK' if not on_sheet else 'UNCLEAR'
    return dict(status=st, scored=True, on_sheet=on_sheet, follows=bool(follows), L_J=LJ, L0_J=L0J,
                L_over_L0=LJ / L0J if L0J else None)


def overall_j(checks_ok, st):
    """st: {fold: status}. VOID if a gate / pooled control fails; FAIL if a fold is ARTEFACT or J-LEAK; PASS if both
    folds are J-PASS; INCOMPLETE if a fold is NOT-RUN (and none FAIL); NO-VERDICT otherwise."""
    s = [st[f] for f in FOLDS]
    if not checks_ok:
        return 'VOID'
    if any(x in ('ARTEFACT', 'J-LEAK') for x in s):
        return 'FAIL'
    if all(x == 'J-PASS' for x in s):
        return 'PASS'
    if 'NOT-RUN' in s:
        return 'INCOMPLETE'
    return 'NO-VERDICT'


def synthetic_rules():
    ok_c, bad_c = dict(ok=True), dict(ok=False)

    def run(L, L0=0.2, fit=True, meas=True):
        return dict(fit_ok=fit, measurable=meas, label=dict(L=L, L0=L0))
    F, A, U, N = 'FOLLOWS', 'ARTEFACT', 'UNCLEAR', 'NO-PEAK'
    fcases = {'pass2': (ok_c, [run(0.02), run(0.03)], [F, F], 'J-PASS'),
              'pass1': (ok_c, [run(0.02)], [F], 'J-PASS'),
              'at_half': (ok_c, [run(0.10)], [F], 'J-PASS'),
              'leak': (ok_c, [run(0.1001)], [F], 'J-LEAK'),
              'leak_mean': (ok_c, [run(0.02), run(0.19)], [F, F], 'J-LEAK'),
              'artefact': (ok_c, [run(0.02), run(0.02)], [F, A], 'ARTEFACT'),
              'artefact_leak': (ok_c, [run(0.2)], [A], 'ARTEFACT'),
              'unclear': (ok_c, [run(0.02), run(0.02)], [F, U], 'UNCLEAR'),
              'nopeak': (ok_c, [run(0.02)], [N], 'UNCLEAR'),
              'nopeak_leak': (ok_c, [run(0.2)], [N], 'J-LEAK'),
              'ctrl_fail': (bad_c, [run(0.02)], [F], 'VOID'),
              'fit_fail': (ok_c, [run(0.02), run(0.02, fit=False)], [F, F], 'VOID'),
              'floor_fail': (ok_c, [run(0.02, meas=False)], [F], 'NO-VERDICT')}
    out = {}
    for k, (c, rs, fr, want) in fcases.items():
        got = fold_status_j(c, {f'J{i}': r for i, r in enumerate(rs)}, {f'J{i}': x for i, x in enumerate(fr)})['status']
        out[f'fold_{k}'] = dict(want=want, got=got, ok=got == want)
    vcases = {'pp': (True, ['J-PASS', 'J-PASS'], 'PASS'), 'pl': (True, ['J-PASS', 'J-LEAK'], 'FAIL'),
              'pa': (True, ['J-PASS', 'ARTEFACT'], 'FAIL'), 'pu': (True, ['J-PASS', 'UNCLEAR'], 'NO-VERDICT'),
              'pv': (True, ['J-PASS', 'VOID'], 'NO-VERDICT'), 'pn': (True, ['J-PASS', 'NO-VERDICT'], 'NO-VERDICT'),
              'pr': (True, ['J-PASS', 'NOT-RUN'], 'INCOMPLETE'), 'lr': (True, ['J-LEAK', 'NOT-RUN'], 'FAIL'),
              'checks_fail': (False, ['J-PASS', 'J-PASS'], 'VOID')}
    for k, (c, s, want) in vcases.items():
        got = overall_j(c, dict(zip(FOLDS, s)))
        out[f'verdict_{k}'] = dict(want=want, got=got, ok=got == want)
    for k, v in out.items():
        print(f'synthetic {k}: {v["got"]} (want {v["want"]})')
    return dict(cases=out, ok=all(v['ok'] for v in out.values()))


def net_gates(inf):
    """G2 net replay: exactly the five registered nets (P0, P1 and the J nets of BOTH folds; infer.py replays every
    requested fold's nets in pass 1, before its budget guard), each finite and <= tol. G4: the site-output nets of
    each inferred fold are the registered ones (R9 VES-3240 G2: a NOT-RUN fold must give INCOMPLETE, not VOID)."""
    want = {'P0', 'P1'} | {f'{n}_h{f}' for f in FOLDS for n in NETS[f]}
    return dict(replay_nets=bool(all(np.isfinite(v) and v <= inf['replay_tol'] for v in inf['replay'].values())
                                 and set(inf['replay']) == want),
                nets_as_registered=bool(inf['nets'] == {f: list(NETS[f]) for f in inf['folds']}))


def synthetic_gates():
    """net_gates on synthetic infer.json dicts (R9 VES-3240): full, w028 NOT-RUN, extra net, missing net, over tol,
    NaN, a wrong net in a fold's site outputs."""
    five = {n: 0.0 for n in ('P0', 'P1', 'J0_hw029', 'J1_hw029', 'J0_hw028')}

    def inf(rep, folds, nets=None):
        return dict(replay=rep, replay_tol=5e-3, folds=folds, nets=nets or {f: list(NETS[f]) for f in folds})
    cases = {'full': (inf(five, ['w029', 'w028']), True),
             'w028_not_run': (inf(five, ['w029']), True),
             'extra_net': (inf(dict(five, D0_hw028=0.0), ['w029', 'w028']), False),
             'missing_net': (inf({k: v for k, v in five.items() if k != 'J0_hw028'}, ['w029', 'w028']), False),
             'missing_net_not_run': (inf({k: v for k, v in five.items() if k != 'J0_hw028'}, ['w029']), False),
             'over_tol': (inf(dict(five, J1_hw029=6e-3), ['w029', 'w028']), False),
             'nan': (inf(dict(five, P1=float('nan')), ['w029', 'w028']), False),
             'wrong_site_nets': (inf(five, ['w029', 'w028'], {'w029': ['J0'], 'w028': ['J0']}), False)}
    out = {}
    for k, (x, want) in cases.items():
        got = all(net_gates(x).values())
        out[k] = dict(want=want, got=got, ok=got == want)
        print(f'synthetic gate {k}: {got} (want {want})')
    return dict(cases=out, ok=all(v['ok'] for v in out.values()))


def gates(U, inf):
    """G1 power and full draw; with infer.json: G3 site replay (every replay point; a tile per replay site of the
    inferred folds) and G2 / G4 (net_gates)."""
    g = dict(power=bool(U.info['power']['ok']), full_draw=bool(U.info['limit'] is None and U.prep['limit_blocks'] is None
                                                               and not U.prep['smoke'] and not C.SMOKE))
    if inf is not None:
        rs = inf['replay_sites']
        g['replay_sites'] = bool(all(rs.values()) and sum(k.startswith('points_') for k in rs) == len(U.prep['replay'])
                                 and sum(k.startswith('tile_') for k in rs) ==
                                 sum(r['key'][0] in inf['folds'] for r in U.prep['replay']))
        g.update(net_gates(inf))
    return g


def hecate_fold(U, idx):
    """score4.fold_controls' hecate half (no P output needed): L <= 0.5 L0 on the fold's label sites."""
    P, F = sub(U.pts, idx), sub(U.flags, idx)
    H = [d['p_hecate_fwd'].astype(np.float32) for d in P]
    sm = S.summarise([S.leak_site(h, d) for h, d in zip(H, P)], F, H, P)
    pc = sm['label']
    return dict(L=pc['L'], L0=pc['L0'], ink_auc=sm['ink_auc'],
                ok=bool(pc['L'] is not None and pc['L0'] and pc['L'] <= SH.PC_FOLD_RATIO * pc['L0']))


def loader(f):
    z = {}

    def ld(name):
        if name not in z:
            z[name] = np.load(OUT / name)
        return z[name]
    site = lambda n, u: ld(f'out_{n}_h{f}.npz')[f'p_{u}']                  # noqa: E731
    shift = lambda n: (lambda k, u: ld(f'out_{n}_h{f}_shift.npz')[f'p{k}_{u}'])   # noqa: E731
    return site, shift


def p_shift(U, folds):
    z = {(n, f): np.load(OUT / f'out_{n}_shift_h{f}.npz') for n in S.FRAME_PC for f in folds}
    return lambda n: (lambda k, u: z[(n, U.keys[u][0])][f'p{k}_{u}'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    U = S4.Sites()
    inf = json.loads((OUT / 'infer.json').read_text()) if (OUT / 'infer.json').exists() and not a.check else None
    res = dict(task='VES-3226', scan='PHerc1667 20251217075048 level 0 (2.403 um, 77 keV, 0.2 m)',
               regate=regate(), gates=gates(U, inf))
    res['checks'] = S.checks(sub(U.pts, U.old), sub(U.flags, U.old))     # scorer + pooled hecate PC, old-draw sites
    res['n_sites'] = {f: dict(label=sum(U.flags[u] for u in U.fidx[f]), N1=sum(not U.flags[u] for u in U.fidx[f]))
                      for f in FOLDS}
    res['hecate_folds'] = {f: hecate_fold(U, U.fidx[f]) for f in FOLDS}
    if a.check:
        res['synthetic_rules'] = synthetic_rules()
        res['synthetic_gates'] = synthetic_gates()
        res['synthetic_frame'] = S.synthetic_frame(sub(U.pts, U.fidx['w029']), sub(U.flags, U.fidx['w029']))
        res['check_ok'] = bool(res['checks']['ok'] and res['synthetic_rules']['ok'] and res['synthetic_gates']['ok'] and
                               all(v['ok'] for v in res['synthetic_frame'].values()) and all(res['gates'].values()))
        CHECK.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(dict(regate=res['regate'], gates=res['gates'], hecate_folds=res['hecate_folds'],
                              check_ok=res['check_ok']), indent=1, default=float))
        return
    assert inf is not None, 'no infer.json: run infer.py first'
    done = list(inf['folds'])
    pshift = p_shift(U, done)
    folds = {}
    for f in FOLDS:
        if f not in done:
            folds[f] = dict(status='NOT-RUN')
            continue
        idx = U.fidx[f]
        site, shift = loader(f)
        trs = {n: json.loads((HO / f'train_{n}_h{f}.json').read_text()) for n in NETS[f]}
        ctrl = S4.fold_controls(U, f, idx, pshift)                         # hecate L <= 0.5 L0; P0, P1 ARTEFACT
        keep, S.MIN_FIT = S.MIN_FIT, res['regate'][f]['bar']              # the re-gate: per-fold fit bar
        try:
            runs, floor = S4.score_set(list(NETS[f]), site, trs, idx, U)
        finally:
            S.MIN_FIT = keep
        fc = {n: S.frame_control(lambda k, j, g=shift(n): g(k, idx[j]), sub(U.pts, idx), sub(U.flags, idx))
              for n in NETS[f] if runs[n]['finite']}
        frame = {n: x['status'] for n, x in fc.items()}
        st = fold_status_j(ctrl, runs, frame) if len(frame) == len(NETS[f]) else dict(status='VOID')
        rd = {n: S4.auc_ci([site(n, u).mean(0) for u in idx], sub(U.pts, idx), sub(U.flags, idx)) for n in NETS[f]}
        folds[f] = dict(st, fit_bar=res['regate'][f]['bar'], controls=ctrl,
                        runs={n: {k: r.get(k) for k in ('fit_auc_shell', 'finite', 'fit_ok', 'measurable', 'ink_auc',
                                                        'label', 'N1', 'phase', 'label_xseg_dropped', 'share_def')}
                              for n, r in runs.items()},
                        frame={n: {k: v for k, v in x.items() if k != 'peaks'} for n, x in fc.items()},
                        mass_floor=floor, auc_ci=rd)
        print(f'fold {f}: {st}', flush=True)
    # C3: P0, P1 ARTEFACT on the pooled label sites of the SCORED folds (as score4.py; R9 VES-3240 C3); with no
    # scored fold there is no pooled P test, and the verdict comes from the fold statuses (NO-VERDICT)
    pidx = [u for f in FOLDS if folds[f].get('scored') for u in U.fidx[f]]
    pctl = {n: S.frame_control(lambda k, j, n=n: pshift(n)(k, pidx[j]), sub(U.pts, pidx), sub(U.flags, pidx))
            for n in S.FRAME_PC} if pidx else {}
    res['pooled_p_control'] = dict(folds=[f for f in FOLDS if folds[f].get('scored')],
                                   **{n: dict(status=x['status'], e=x['e']) for n, x in pctl.items()})
    gate_ok = all(res['gates'].values()) and len(res['gates']) == 5
    c3_ok = all(x['status'] == 'ARTEFACT' for x in pctl.values())
    checks_ok = bool(gate_ok and res['checks']['ok'] and c3_ok)
    res['folds'] = folds
    res['fold_status'] = {f: folds[f]['status'] for f in FOLDS}
    res['checks_ok'] = checks_ok
    res['verdict'] = overall_j(checks_ok, res['fold_status'])
    res['gpu'] = dict(seconds=inf['total_seconds'], gpu=inf['gpu'], gpu_h=round(inf['total_seconds'] / 3600, 3))
    RESULTS.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ('gates', 'fold_status', 'pooled_p_control', 'checks_ok', 'verdict')},
                     indent=1, default=float))


if __name__ == '__main__':
    main()
