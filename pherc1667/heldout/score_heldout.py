"""VES-2257 scorer: held-out confirmation of F-399 (VES-2241 attempt 3), README "Criteria" of heldout/. CPU only.
Reuses score.py's L / L0 / ink AUC (score_run), frame-shift control (frame_control) and verdict rules (verdict_of)
unchanged; only adds the fold split, the per-fold controls and the fold rule.

  python heldout/score_heldout.py --check   scorer + hecate PC (pooled and per fold), P-type known-artefact control
                                            per fold, synthetic fold-rule cases; no held-out output read
  python heldout/score_heldout.py           heldout/results.json
"""
import argparse
import json
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


class Loader:
    """Held-out outputs of fold f: site outputs (2, n, 16) and frame-shift outputs, by run name J0, J1, D0, D1."""

    def __init__(self, fold, n_sites):
        self.f, self.n = fold, n_sites
        self.z, self.s = {}, {}

    def __call__(self, name):
        if name not in self.z:
            self.z[name] = np.load(OUT / f'out_{name}_h{self.f}.npz')
        return [self.z[name][f'p_{i}'] for i in range(self.n)]

    def shift(self, name):
        if name not in self.s:
            self.s[name] = np.load(OUT / f'out_{name}_h{self.f}_shift.npz')
        z = self.s[name]
        return lambda k, i: z[f'p{k}_{i}']

    def tr_path(self, name):
        return OUT / f'train_{name}_h{self.f}.json'


def fold_controls(f, idx, pts, flags):
    """Per-fold controls that need no held-out output: hecate PC (L <= 0.5 L0 on the fold's label sites) and the
    P-type known-artefact control (attempt 2's P0, P1 frame-shift outputs on the fold's sites: both ARTEFACT)."""
    H = [pts[i]['p_hecate_fwd'].astype(np.float32) for i in idx]
    pc = S.summarise([S.leak_site(h, pts[i]) for h, i in zip(H, idx)], sub(flags, idx), H, sub(pts, idx))['label']
    pc_ok = bool(pc['L'] is not None and pc['L0'] and pc['L'] <= PC_FOLD_RATIO * pc['L0'])
    pfc = {}
    for n in S.FRAME_PC:
        z = np.load(C.DATA / f'out_{n}_shift.npz')
        pfc[n] = S.frame_control(lambda k, j: z[f'p{k}_{idx[j]}'], sub(pts, idx), sub(flags, idx))['status']
    return dict(pc_hecate=dict(L=pc['L'], L0=pc['L0'], ok=pc_ok), p_control=pfc,
                p_control_ok=all(v == 'ARTEFACT' for v in pfc.values()), ok=bool(pc_ok and all(
                    v == 'ARTEFACT' for v in pfc.values())))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    pts = S.f395_points()
    keys = S.site_list()
    flags = [k[1] == 'label' for k in keys]
    meta = json.loads((C.DATA / 'prep.json').read_text())
    assert meta['site_keys'] == keys
    res = dict(checks=S.checks(pts, flags))                       # scorer check + pooled hecate PC (4.2 / 18.4 pp)
    xdrop = S.xseg_drop(keys)
    fidx = {f: [i for i, k in enumerate(keys) if k[0] == f] for f in FOLDS}
    assert sorted(sum(fidx.values(), [])) == list(range(len(keys)))
    res['fold_controls'] = {f: fold_controls(f, fidx[f], pts, flags) for f in FOLDS}
    for f, c in res['fold_controls'].items():
        print(f'fold {f}: hecate L {c["pc_hecate"]["L"]:.4f} L0 {c["pc_hecate"]["L0"]:.4f} '
              f'{"PASS" if c["pc_hecate"]["ok"] else "FAIL"}; P control {c["p_control"]}')
    if a.check:
        res['checks']['synthetic_fold_rule'] = synthetic_overall()
        res['checks']['synthetic_attempt3'] = S.synthetic(pts, flags, xdrop, C.ATTEMPT_ARMS[3], 3)
        print(json.dumps(dict(ok=res['checks']['ok'] and res['checks']['synthetic_fold_rule']['ok'] and
                              res['checks']['synthetic_attempt3']['ok']), indent=1))
        return
    names = [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
    folds, ph_pool, sh_pool, tr_pool = {}, {}, {}, {}
    for f in FOLDS:
        ld = Loader(f, len(keys))
        if not all(ld.tr_path(n).exists() for n in names):
            folds[f] = dict(status='NOT-RUN')
            continue
        idx = fidx[f]
        trs = {n: json.loads(ld.tr_path(n).read_text()) for n in names}
        runs = {}
        n_lab = sum(flags[i] for i in idx)
        keep, S.MIN_MASS = S.MIN_MASS, MASS_PER_SITE * n_lab
        try:
            xd = {k for k, i in enumerate(idx) if i in xdrop}
            for n in names:
                runs[n] = S.score_run(sub(ld(n), idx), trs[n], sub(pts, idx), sub(flags, idx), xd)
        finally:
            S.MIN_MASS = keep
        frame, fc = {}, {}
        for n in names:
            g = ld.shift(n)
            if runs[n]['finite']:
                fc[n] = S.frame_control(lambda k, j: g(k, idx[j]), sub(pts, idx), sub(flags, idx))
                fc[n]['role'] = 'gate' if n[0] == 'J' else 'reading'
                frame[n] = fc[n]['status']
        st = fold_status(res['fold_controls'][f], runs, frame)
        folds[f] = dict(st, runs={n: {k: r.get(k) for k in ('fit_auc_shell', 'finite', 'fit_ok', 'measurable',
                                                             'ink_auc', 'label', 'N1', 'phase', 'label_xseg_dropped',
                                                             'share_def')} for n, r in runs.items()},
                        frame={n: {k: v for k, v in x.items() if k != 'peaks'} for n, x in fc.items()},
                        mass_floor=MASS_PER_SITE * n_lab, n_label_sites=n_lab,
                        train={n: {k: trs[n].get(k) for k in ('seconds', 'norm', 'kept_blocks', 'kept_segments')}
                               for n in names})
        print(f'fold {f}: {st}')
        if st.get('scored'):
            for n in names:
                ph = ld(n)
                for i in idx:
                    ph_pool.setdefault(n, {})[i] = ph[i]
                    sh_pool.setdefault(n, {})[i] = ld.shift(n)
                tr_pool.setdefault(n, []).append(trs[n])
    res['folds'] = folds
    pooled_v = None
    pidx = sorted(next(iter(ph_pool.values())).keys()) if ph_pool else []
    if pidx:
        n_lab = sum(flags[i] for i in pidx)
        keep, S.MIN_MASS = S.MIN_MASS, MASS_PER_SITE * n_lab
        try:
            xd = {k for k, i in enumerate(pidx) if i in xdrop}
            P, F = sub(pts, pidx), sub(flags, pidx)
            runs = {n: S.score_run([ph_pool[n][i] for i in pidx], dict(
                fit_auc_shell=min(t['fit_auc_shell'] for t in tr_pool[n]), steps=tr_pool[n][0]['steps'],
                seconds=sum(t['seconds'] for t in tr_pool[n]), gpu=tr_pool[n][0]['gpu'],
                n_params=tr_pool[n][0]['n_params']), P, F, xd) for n in names}
        finally:
            S.MIN_MASS = keep
        frame = {}
        pfc = {}
        for n in names:
            pfc[n] = S.frame_control(lambda k, j, n=n: sh_pool[n][pidx[j]](k, pidx[j]), P, F)
            frame[n] = pfc[n]['status']
        pctl = {}
        for n in S.FRAME_PC:
            z = np.load(C.DATA / f'out_{n}_shift.npz')
            pctl[n] = S.frame_control(lambda k, j, z=z: z[f'p{k}_{pidx[j]}'], P, F)['status']
        ck = dict(scorer=res['checks']['scorer']['ok'], pc_hecate=res['checks']['pc_hecate']['ok'],
                  frame_pc=all(v == 'ARTEFACT' for v in pctl.values()))
        arms, pooled_v = S.verdict_of(ck, runs, C.ATTEMPT_ARMS[3], {f'J{s}': frame[f'J{s}'] for s in C.SEEDS})
        res['pooled'] = dict(folds=[f for f in FOLDS if folds[f].get('scored')], n_sites=len(pidx),
                             mass_floor=MASS_PER_SITE * n_lab, runs={n: {k: r.get(k) for k in (
                                 'ink_auc', 'label', 'N1', 'label_xseg_dropped', 'measurable')} for n, r in runs.items()},
                             frame={n: x['status'] for n, x in pfc.items()}, frame_e={n: x['e'] for n, x in pfc.items()},
                             p_control=pctl, arms=arms, verdict=pooled_v)
    res['verdict'] = overall(res['checks']['ok'], folds, pooled_v)
    res['fold_status'] = {f: folds[f]['status'] for f in FOLDS}
    res['n_win'] = list(res['fold_status'].values()).count('WIN')
    (HERE / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ('fold_status', 'n_win', 'verdict')}, indent=1))
    if pooled_v:
        print('pooled:', {k: pooled_v.get(k) for k in ('verdict', 'lower_L_arm', 'dL', 'L_ratio', 'seed_noise')})


if __name__ == '__main__':
    main()
