"""VES-2264 scorer (VES-2241 attempt 4): the F-400 held-out rules (heldout/score_heldout.py, heldout/README "Criteria")
re-run on a larger site set per fold = the 80 F-400 sites of the fold + the new sites of attempt4/sites4.py. Every
rule function is imported unchanged: score.score_run / frame_control / verdict_of / summarise / leak_site / checks,
score_heldout.fold_status / overall / MASS_PER_SITE. New here: only the site union and its data access, the power
and replay gates (README "Criteria" 1-2), and readings (AUC CIs, old-only / new-only AUC).

  python attempt4/score4.py --check   gates + pooled controls + per-fold controls (hecate PC, P0/P1) on the union
                                      sites, F-400's synthetic cases; no fold-net output read
  python attempt4/score4.py           attempt4/results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'heldout'))
import common as C  # noqa: E402
import score as S  # noqa: E402
import score_heldout as SH  # noqa: E402

t4 = S.t4
OUT = C.DATA / 'attempt4'
HO = C.DATA / 'heldout'
FOLDS = SH.FOLDS
NAMES = [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
NBOOT = 2000


class Sites:
    """Union site list: entries ('old', i) index the F-395 / F-400 arrays, ('new', j) the attempt-4 arrays."""

    def __init__(self):
        self.old_pts, self.old_keys = S.f395_points(), S.site_list()
        meta = json.loads((C.DATA / 'prep.json').read_text())
        assert meta['site_keys'] == self.old_keys
        self.info = json.loads((OUT / 'sites4.json').read_text())
        self.new_keys = self.info['site_keys']
        z = np.load(OUT / 'points4.npz')
        self.new_pts = [{k: z[f'{k}_{j}'] for k in ('dfn', 'use', 'pm', 'pp', 'p_hecate_fwd')}
                        for j in range(len(self.new_keys))]
        assert [len(d['pm']) for d in self.new_pts] == self.info['n_points']
        self.ids = [('old', i) for i in range(len(self.old_keys))] + [('new', j) for j in range(len(self.new_keys))]
        self.keys = self.old_keys + self.new_keys
        self.pts = self.old_pts + self.new_pts
        self.flags = [k[1] == 'label' for k in self.keys]
        xo = S.xseg_drop(self.old_keys)
        xn = {tuple(k) for k in json.loads((OUT / 'xseg4.json').read_text())['drop_keys']}
        self.xdrop = set(xo) | {len(self.old_keys) + j for j, k in enumerate(self.new_keys) if tuple(k) in xn}
        self.fidx = {f: [u for u, k in enumerate(self.keys) if k[0] == f] for f in FOLDS}


class Out:
    """Site and shift outputs of run n of fold f on union index u."""

    def __init__(self, U, f):
        self.U, self.f, self.z, self.s = U, f, {}, {}

    def _ld(self, kind, n, shift):
        key = (kind, n, shift)
        if key not in self.z:
            d = HO if kind == 'old' else OUT
            self.z[key] = np.load(d / f'out_{n}_h{self.f}{"_shift" if shift else ""}.npz')
        return self.z[key]

    def site(self, n, u):
        kind, i = self.U.ids[u]
        return self._ld(kind, n, False)[f'p_{i}']

    def shift(self, n):
        def g(k, u):
            kind, i = self.U.ids[u]
            return self._ld(kind, n, True)[f'p{k}_{i}']
        return g


def p_shift(U):
    zo = {n: np.load(C.DATA / f'out_{n}_shift.npz') for n in S.FRAME_PC}
    zn = {n: np.load(OUT / f'out_{n}_shift.npz') for n in S.FRAME_PC}

    def get(n):
        def g(k, u):
            kind, i = U.ids[u]
            return (zo if kind == 'old' else zn)[n][f'p{k}_{i}']
        return g
    return get


def sub(xs, idx):
    return [xs[i] for i in idx]


def fold_controls(U, f, idx, pshift):
    """score_heldout.fold_controls on the union sites: hecate L <= 0.5 L0 and P0, P1 both ARTEFACT."""
    P, F = sub(U.pts, idx), sub(U.flags, idx)
    H = [d['p_hecate_fwd'].astype(np.float32) for d in P]
    sm = S.summarise([S.leak_site(h, d) for h, d in zip(H, P)], F, H, P)
    pc = sm['label']
    pc_ok = bool(pc['L'] is not None and pc['L0'] and pc['L'] <= SH.PC_FOLD_RATIO * pc['L0'])
    pfc = {n: S.frame_control(lambda k, j, n=n: pshift(n)(k, idx[j]), P, F)['status'] for n in S.FRAME_PC}
    p_ok = all(v == 'ARTEFACT' for v in pfc.values())
    return dict(pc_hecate=dict(L=pc['L'], L0=pc['L0'], ink_auc=sm['ink_auc'], ok=pc_ok), p_control=pfc,
                p_control_ok=p_ok, ok=bool(pc_ok and p_ok))


def auc_ci(probs, pts, flags, seed=2264):
    """Reading: ink AUC (score.summarise's column-max rule) with a 95 % site-bootstrap CI (label and N1 sites
    resampled separately)."""
    cols = [p[d['use']].max(1) for p, d in zip(probs, pts)]
    L = [c for c, f in zip(cols, flags) if f]
    N = [c for c, f in zip(cols, flags) if not f]
    a = t4.auc(np.concatenate(L), np.concatenate(N))
    rng = np.random.default_rng(seed)
    bs = [t4.auc(np.concatenate([L[i] for i in rng.integers(0, len(L), len(L))]),
                 np.concatenate([N[i] for i in rng.integers(0, len(N), len(N))])) for _ in range(NBOOT)]
    return dict(auc=a, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_label=len(L),
                n_N1=len(N))


def score_set(names, get_ph, tr, idx, U):
    """score.score_run per run on the sites idx with the mass floor scaled to the label-site count."""
    n_lab = sum(U.flags[u] for u in idx)
    keep, S.MIN_MASS = S.MIN_MASS, SH.MASS_PER_SITE * n_lab
    try:
        xd = {k for k, u in enumerate(idx) if u in U.xdrop}
        return {n: S.score_run([get_ph(n, u) for u in idx], tr[n], sub(U.pts, idx), sub(U.flags, idx), xd)
                for n in names}, SH.MASS_PER_SITE * n_lab
    finally:
        S.MIN_MASS = keep


def gates(U):
    pw, rp = U.info['power'], U.info['replay']
    g = dict(power=bool(pw['ok']), replay_sites=bool(rp['ok']), full_draw=U.info['limit'] is None)
    inf = OUT / 'infer4.json'
    if inf.exists():
        x = json.loads(inf.read_text())
        g['replay_nets'] = bool(all(np.isfinite(v) and v <= x['replay_tol'] for v in x['replay'].values()) and
                                len(x['replay']) == len(NAMES) * len(FOLDS) + 2 and sorted(x['folds']) == sorted(FOLDS))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    U = Sites()
    res = dict(gates=gates(U))
    if not (res['gates']['power'] and res['gates']['full_draw']):
        res['verdict'] = 'VOID'
        (HERE / 'results.json').write_text(json.dumps(res, indent=1))
        sys.exit('power rule / draw incomplete: attempt 4 VOID (no scoring)')
    res['checks'] = S.checks(U.old_pts, U.flags[:len(U.old_keys)])      # scorer + pooled hecate PC, F-395 points
    pshift = p_shift(U) if (OUT / 'out_P0_shift.npz').exists() else None
    res['power'] = U.info['power']['folds']
    res['n_sites'] = {f: dict(label=sum(U.flags[u] for u in U.fidx[f]),
                              N1=sum(not U.flags[u] for u in U.fidx[f])) for f in FOLDS}
    H = [d['p_hecate_fwd'].astype(np.float32) for d in U.new_pts]
    res['hecate_new_sites'] = S.summarise([S.leak_site(h, d) for h, d in zip(H, U.new_pts)],
                                          U.flags[len(U.old_keys):], H, U.new_pts)['label']   # reading
    if pshift is not None:
        res['fold_controls'] = {f: fold_controls(U, f, U.fidx[f], pshift) for f in FOLDS}
        for f, c in res['fold_controls'].items():
            print(f'fold {f}: hecate L {c["pc_hecate"]["L"]:.4f} L0 {c["pc_hecate"]["L0"]:.4f} '
                  f'{"PASS" if c["pc_hecate"]["ok"] else "FAIL"}; P control {c["p_control"]}')
    if a.check:
        res['checks']['synthetic_fold_rule'] = SH.synthetic_overall()
        print(json.dumps(dict(gates=res['gates'], checks_ok=res['checks']['ok'],
                              synthetic_ok=res['checks']['synthetic_fold_rule']['ok'], n_sites=res['n_sites']),
                         indent=1))
        return
    gate_ok = all(res['gates'].values()) and len(res['gates']) == 4
    folds, pool = {}, {}
    for f in FOLDS:
        idx, O = U.fidx[f], Out(U, f)
        trs = {n: json.loads((HO / f'train_{n}_h{f}.json').read_text()) for n in NAMES}
        runs, floor = score_set(NAMES, O.site, trs, idx, U)
        frame, fc = {}, {}
        for n in NAMES:
            if runs[n]['finite']:
                g = O.shift(n)
                fc[n] = S.frame_control(lambda k, j, g=g: g(k, idx[j]), sub(U.pts, idx), sub(U.flags, idx))
                fc[n]['role'] = 'gate' if n[0] == 'J' else 'reading'
                frame[n] = fc[n]['status']
        st = SH.fold_status(res['fold_controls'][f], runs, frame)
        rd = {}
        for n in NAMES:                                                  # readings: AUC CIs, old-only / new-only
            ph = [O.site(n, u).mean(0) for u in idx]
            P, F = sub(U.pts, idx), sub(U.flags, idx)
            old = [k for k, u in enumerate(idx) if U.ids[u][0] == 'old']
            new = [k for k, u in enumerate(idx) if U.ids[u][0] == 'new']
            rd[n] = dict(all=auc_ci(ph, P, F), f400_sites=auc_ci(sub(ph, old), sub(P, old), sub(F, old)),
                         new_sites=auc_ci(sub(ph, new), sub(P, new), sub(F, new)))
        folds[f] = dict(st, runs={n: {k: r.get(k) for k in ('fit_auc_shell', 'finite', 'fit_ok', 'measurable',
                                                             'ink_auc', 'label', 'N1', 'phase', 'label_xseg_dropped',
                                                             'share_def')} for n, r in runs.items()},
                        frame={n: {k: v for k, v in x.items() if k != 'peaks'} for n, x in fc.items()},
                        mass_floor=floor, n_label_sites=sum(U.flags[u] for u in idx), auc_readings=rd)
        print(f'fold {f}: {st}', flush=True)
        if st.get('scored'):
            for u in idx:
                pool[u] = O
    res['folds'] = folds
    pooled_v = None
    pidx = sorted(pool)
    if pidx:
        tr_pool = {n: [json.loads((HO / f'train_{n}_h{f}.json').read_text()) for f in FOLDS
                       if folds[f].get('scored')] for n in NAMES}
        tr = {n: dict(fit_auc_shell=min(t['fit_auc_shell'] for t in tr_pool[n]), steps=tr_pool[n][0]['steps'],
                      seconds=sum(t['seconds'] for t in tr_pool[n]), gpu=tr_pool[n][0]['gpu'],
                      n_params=tr_pool[n][0]['n_params']) for n in NAMES}
        runs, floor = score_set(NAMES, lambda n, u: pool[u].site(n, u), tr, pidx, U)
        P, F = sub(U.pts, pidx), sub(U.flags, pidx)
        pfc = {n: S.frame_control(lambda k, j, n=n: pool[pidx[j]].shift(n)(k, pidx[j]), P, F) for n in NAMES}
        pctl = {n: S.frame_control(lambda k, j, n=n: pshift(n)(k, pidx[j]), P, F)['status'] for n in S.FRAME_PC}
        ck = dict(scorer=res['checks']['scorer']['ok'], pc_hecate=res['checks']['pc_hecate']['ok'],
                  frame_pc=all(v == 'ARTEFACT' for v in pctl.values()))
        arms, pooled_v = S.verdict_of(ck, runs, C.ATTEMPT_ARMS[3], {f'J{s}': pfc[f'J{s}']['status'] for s in C.SEEDS})
        res['pooled'] = dict(folds=[f for f in FOLDS if folds[f].get('scored')], n_sites=len(pidx), mass_floor=floor,
                             runs={n: {k: r.get(k) for k in ('ink_auc', 'label', 'N1', 'label_xseg_dropped',
                                                             'measurable')} for n, r in runs.items()},
                             frame={n: x['status'] for n, x in pfc.items()}, frame_e={n: x['e'] for n, x in pfc.items()},
                             p_control=pctl, arms=arms, verdict=pooled_v)
    res['verdict'] = SH.overall(res['checks']['ok'] and gate_ok, folds, pooled_v)
    res['fold_status'] = {f: folds[f]['status'] for f in FOLDS}
    res['n_win'] = list(res['fold_status'].values()).count('WIN')
    res['site_keys_new'] = U.new_keys
    (HERE / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ('gates', 'fold_status', 'n_win', 'verdict')}, indent=1))
    if pooled_v:
        print('pooled:', {k: pooled_v.get(k) for k in ('verdict', 'lower_L_arm', 'dL', 'L_ratio', 'seed_noise')})


if __name__ == '__main__':
    main()
