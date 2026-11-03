"""VES-2241 scorer: leak L, depth-blind null L0 and ink AUC per (arm, seed) on the F-389 points, with F-395's
pitches (attempt-2 rule), and the verdict of README "Criteria". CPU only.

  python score.py --check      scorer checks + hecate positive control (F-395 points), no arm output read
  python score.py              results.json (attempt 1: arms S, D)
  python score.py --attempt 2  attempt2/results.json (arm P = S' against the attempt-1 D runs; same rules, P in S's role)
  python score.py --attempt 3  attempt3/results.json (arm J = P + frame jitter against the attempt-1 D runs; same rules,
                               plus the frame-shift control of attempt3/README.md, which gates J before the effect rules)
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import common as C

t4 = C.f389()
t23 = t4.t23
PC_L, PC_L0, PC_TOL = 0.042, 0.184, 0.003
MIN_AUC, MIN_MASS, MIN_FIT = 0.60, 1000.0, 0.80
EFF_RATIO, EFF_PP, NOISE_K = 1.5, 0.03, 2.0
NULL_PP, NULL_RATIO = 0.03, 1.25
MAX_SEED_PP = 0.05
AUC_TOL = 0.05
FOLLOW_TOL, ARTEFACT_AT = 1.0, 2.0          # attempt 3: |e_k| <= 1 vx follows; |e_k| >= k / 2 = 2 vx is an artefact
FRAME_PC = ('P0', 'P1')                    # attempt-2 nets, known frame artefact (F-397): control must say ARTEFACT
PEAK_MIN = 1.5                              # a profile has a peak if its max >= 1.5 x its median over C.SEARCH


def f395_points():
    sha = hashlib.sha256(C.F395_POINTS.read_bytes()).hexdigest()
    assert sha == C.F395_POINTS_SHA, f'F-395 points.npz changed: {sha}'
    z = np.load(C.F395_POINTS)
    n = len({k.rsplit('_', 1)[1] for k in z.files})
    return [{k: z[f'{k}_{i}'] for k in ('dfn', 'use', 'pm', 'pp', 'p_hecate_fwd')} for i in range(n)]


def site_list():
    """F-395 site order: per dev segment, 40 label then 40 N1 draws (the same draw code)."""
    out = []
    for wi, w in enumerate(t23.DEV):
        lab = np.load(t23.REPO_DATA / 'ves170/labels' / f'{w}.npz')
        g, v = t23.load_grid(t23.REPO_DATA / 'ves170/tifxyz' / w)
        valid = t23.pixel_valid(v, lab['pos'].shape)
        out += [s['key'] for s in C.sites(t23, w, wi, lab['pos'], lab['neg'], valid)]
    return out


def leak_site(prob, d):
    """run_t4.leak_sums on the defined points of one site, summed (F-389 / F-395 definition)."""
    k = d['dfn']
    s = t4.leak_sums(prob[k].astype(np.float64), d['pm'][k].astype(np.float64), d['pp'][k].astype(np.float64))
    return {kk: float(vv.sum()) for kk, vv in s.items()}


def summarise(per_site, label_flags, probs, pts):
    """L, L0 (CIs by bootstrap over sites, run_t4.boot_ratio), ink AUC (column max, label vs N1 use points)."""
    res = {}
    for grp, want in (('label', True), ('N1', False)):
        S = [s for s, f in zip(per_site, label_flags) if f == want]
        L, ci = t4.boot_ratio(S, 'num', 'den')
        L0, ci0 = t4.boot_ratio(S, 'num0', 'den')
        Ls, cis = t4.boot_ratio(S, 'nums', 'dens')
        res[grp] = dict(L=L, L_ci=ci, L0=L0, L0_ci=ci0, L_over_L0=(L / L0) if (L is not None and L0) else None,
                        L_soft=Ls, L_soft_ci=cis, ink_mass=float(sum(s['den'] for s in S)))
    cp = np.concatenate([p[d['use']].max(1) for p, d, f in zip(probs, pts, label_flags) if f])
    cn = np.concatenate([p[d['use']].max(1) for p, d, f in zip(probs, pts, label_flags) if not f])
    res['ink_auc'] = t4.auc(cp, cn)
    lab = [p[d['dfn']] for p, d, f in zip(probs, pts, label_flags) if f]
    res['mass_by_plane_label'] = np.sum([(p * (p >= t4.INK)).sum(0) for p in lab], 0).round(2).tolist()
    return res


def checks(pts, flags):
    chk = t4.scorer_check()
    pc = summarise([leak_site(d['p_hecate_fwd'], d) for d in pts], flags, [d['p_hecate_fwd'].astype(np.float32)
                                                                          for d in pts], pts)
    L, L0 = pc['label']['L'], pc['label']['L0']
    pc_ok = bool(L is not None and abs(L - PC_L) <= PC_TOL + 1e-12 and abs(L0 - PC_L0) <= PC_TOL + 1e-12)
    print(f'PC hecate (F-395 points, this scorer): L {L:.4f} L0 {L0:.4f} AUC {pc["ink_auc"]:.3f} '
          f'{"PASS" if pc_ok else "FAIL"}')
    return dict(scorer=chk, pc_hecate=dict(L=L, L0=L0, ink_auc=pc['ink_auc'], ok=pc_ok),
                ok=bool(chk['ok'] and pc_ok))


def sratio(hi, lo):
    return (hi / lo) if lo > 0 else (math.inf if hi > 0 else 1.0)


def mean_def(xs):
    """Mean if every value is defined, else None (a run with L undefined is caught by the gates first)."""
    return float(np.mean(xs)) if all(x is not None for x in xs) else None


def score_run(ph, tr, pts, flags, xdrop):
    """One (arm, seed) run. ph: per site (2 phases, n, 16). Non-finite output = diverged net -> not fit (VOID)."""
    r = dict(fit_auc_shell=tr['fit_auc_shell'], train={k: tr[k] for k in ('steps', 'seconds', 'gpu', 'n_params')})
    r['finite'] = bool(all(np.isfinite(p).all() for p in ph))
    r['fit_ok'] = bool(r['finite'] and tr['fit_auc_shell'] >= MIN_FIT)
    if not r['finite']:
        r['measurable'] = False
        return r
    prim = [p.mean(0) for p in ph]                                         # primary: mean of the 2 phases
    r.update(summarise([leak_site(p, d) for p, d in zip(prim, pts)], flags, prim, pts))
    r['phase'] = [summarise([leak_site(p[k], d) for p, d in zip(ph, pts)], flags, [p[k] for p in ph],
                            pts)['label']['L'] for k in (0, 1)]
    keep = [i for i in range(len(pts)) if i not in xdrop]                  # reading only: no verdict role
    r['label_xseg_dropped'] = summarise([leak_site(prim[i], pts[i]) for i in keep], [flags[i] for i in keep],
                                        [prim[i] for i in keep], [pts[i] for i in keep])['label']
    n = sum(int(d['use'].sum()) for d, f in zip(pts, flags) if f)
    r['share_def'] = sum(int(d['dfn'].sum()) for d, f in zip(pts, flags) if f) / n
    r['measurable'] = bool(r['share_def'] >= t4.MIN_DEF and r['label']['ink_mass'] >= MIN_MASS and
                           r['label']['L'] is not None and (r['ink_auc'] or 0) >= MIN_AUC)
    return r


def frame_peak(prof):
    """Peak of a 32-plane profile, in frame t (T32): argmax over the C.SEARCH planes, then the centroid of the
    contiguous run of SEARCH planes around it with p >= 0.5 x max. has_peak: max >= PEAK_MIN x median(SEARCH)."""
    idx = np.flatnonzero(C.SEARCH)
    q = np.asarray(prof, np.float64)[idx]
    a = int(np.argmax(q))
    lo = a
    while lo > 0 and q[lo - 1] >= 0.5 * q[a]:
        lo -= 1
    hi = a
    while hi < len(q) - 1 and q[hi + 1] >= 0.5 * q[a]:
        hi += 1
    w = q[lo:hi + 1]
    t = float((w * C.T32[idx[lo:hi + 1]]).sum() / w.sum()) if w.sum() > 0 else float('nan')
    med = float(np.median(q))
    return dict(t_frame=t, p_max=float(q[a]), median=med, has_peak=bool(q[a] > 0 and q[a] >= PEAK_MIN * med))


def frame_control(get, pts, flags):
    """Attempt 3 frame-shift control of one run. get(k, i) -> (n, 32) p at site i's points, window at mesh offset k.
    Profile = mean p over label-site use points by window plane. Mesh-frame peak m_k = t_frame_k + k; e_k = m_k - m_0.
    CT-driven: e_k = 0 (the frame peak moves by -k with the CT); frame artefact: e_k = k (the peak stays in the frame).
    FOLLOWS: every profile has a peak and |e_k| <= FOLLOW_TOL for both k; ARTEFACT: some |e_k| >= ARTEFACT_AT
    (peaks present); NO-PEAK: some profile has no peak; UNCLEAR otherwise."""
    pk = {}
    for k in (0,) + tuple(C.SHIFT_K):
        prof = np.mean(np.concatenate([get(k, i)[d['use']].astype(np.float64) for i, (d, f) in
                                       enumerate(zip(pts, flags)) if f]), 0)
        pk[k] = dict(frame_peak(prof), profile=prof.round(4).tolist())
        pk[k]['t_mesh'] = pk[k]['t_frame'] + k
    e = {k: pk[k]['t_mesh'] - pk[0]['t_mesh'] for k in C.SHIFT_K}
    peaks = all(v['has_peak'] for v in pk.values())
    if not peaks:
        st = 'NO-PEAK'
    elif any(abs(x) >= ARTEFACT_AT for x in e.values()):
        st = 'ARTEFACT'
    elif all(abs(x) <= FOLLOW_TOL for x in e.values()):
        st = 'FOLLOWS'
    else:
        st = 'UNCLEAR'
    return dict(status=st, e={str(k): float(x) for k, x in e.items()}, peaks={str(k): v for k, v in pk.items()})


def verdict_of(checks_ok, runs, arms=C.ATTEMPT_ARMS[1], frame=None):
    """README "Criteria": gates first (VOID, VOID, NO-VERDICT), arm statistics only for the effect rules.
    frame (attempt 3): {run name: frame_control status} of the informative arm's runs. Any ARTEFACT -> FRAME-ARTEFACT
    after the NO-VERDICT gate; LABEL-EFFECT needs every run FOLLOWS, else it is reported INCONCLUSIVE."""
    arm = {}
    for A in arms:
        rs = [runs[f'{A}{s}'] for s in C.SEEDS]
        if not all(r['finite'] for r in rs):
            arm[A] = None
            continue
        Ls = [r['label']['L'] for r in rs]
        arm[A] = dict(L=mean_def(Ls), L0=mean_def([r['label']['L0'] for r in rs]),
                      L_N1=mean_def([r['N1']['L'] for r in rs]), ink_auc=mean_def([r['ink_auc'] for r in rs]),
                      seed_spread=float(abs(Ls[0] - Ls[1])) if None not in Ls else None,
                      L_xseg_dropped=mean_def([r['label_xseg_dropped']['L'] for r in rs]))
    v = dict(scorer=checks_ok['scorer'], pc_hecate=checks_ok['pc_hecate'],
             fit_all=all(r['fit_ok'] for r in runs.values()),
             measurable_all=all(r['measurable'] for r in runs.values()))
    if 'frame_pc' in checks_ok:                                             # attempt 3: known-artefact control
        v['frame_pc'] = bool(checks_ok['frame_pc'])
    if not (v['scorer'] and v['pc_hecate']) or v.get('frame_pc') is False:
        v['verdict'] = 'VOID'
    elif not v['fit_all']:
        v['verdict'] = 'VOID'
    elif not v['measurable_all']:
        v['verdict'] = 'NO-VERDICT'
    if 'verdict' not in v and frame is not None:
        v['frame'] = dict(frame)
        v['frame_follows_all'] = all(x == 'FOLLOWS' for x in frame.values())
        if any(x == 'ARTEFACT' for x in frame.values()):
            v['verdict'] = 'FRAME-ARTEFACT'
    if 'verdict' in v:
        return arm, v
    a_s, a_d = arms
    S, Dd = arm[a_s], arm[a_d]
    lo_a, hi_a = (a_s, a_d) if S['L'] <= Dd['L'] else (a_d, a_s)
    Lo, Hi = arm[lo_a], arm[hi_a]
    v['lower_L_arm'] = lo_a
    v['dL'] = Hi['L'] - Lo['L']
    v['L_ratio'] = sratio(Hi['L'], Lo['L'])
    v['seed_noise'] = max(S['seed_spread'], Dd['seed_spread'])
    v['seed_stable'] = bool(v['seed_noise'] <= MAX_SEED_PP)
    v['e1_ratio'] = bool(v['L_ratio'] >= EFF_RATIO)
    v['e2_dL'] = bool(v['dL'] >= EFF_PP)
    v['e3_above_noise'] = bool(v['dL'] > NOISE_K * v['seed_noise'])
    lo_hi_ci = max(runs[f'{lo_a}{s}']['label']['L_ci'][1] for s in C.SEEDS)
    hi_lo_ci = min(runs[f'{hi_a}{s}']['label']['L_ci'][0] for s in C.SEEDS)
    v['e4_ci_apart'] = bool(lo_hi_ci < hi_lo_ci)
    v['e5_N1_direction'] = bool(Lo['L_N1'] is not None and Hi['L_N1'] is not None and Lo['L_N1'] < Hi['L_N1'])
    v['e6_equal_auc'] = bool(Lo['ink_auc'] >= Hi['ink_auc'] - AUC_TOL)
    v['n1_small'] = bool(v['dL'] < NULL_PP)
    v['n2_ratio'] = bool(v['L_ratio'] < NULL_RATIO)
    if not v['seed_stable']:
        v['verdict'] = 'INCONCLUSIVE'
    elif all(v[k] for k in ('e1_ratio', 'e2_dL', 'e3_above_noise', 'e4_ci_apart', 'e5_N1_direction',
                            'e6_equal_auc')):
        v['verdict'] = 'LABEL-EFFECT'
        if frame is not None and not v['frame_follows_all']:
            v['verdict'] = 'INCONCLUSIVE'                                   # effect rules met, frame control not
            v['frame_downgrade'] = True
    elif v['n1_small'] and v['n2_ratio']:
        v['verdict'] = 'NO-LABEL-EFFECT'
    else:
        v['verdict'] = 'INCONCLUSIVE'
    return arm, v


def fake_shift(pts, f):
    """Synthetic frame-shift outputs: a Gaussian depth peak (sigma 1.5 vx) whose frame position is -f x k, i.e. f = 1
    follows the CT (mesh peak fixed), f = 0 stays in the frame. Label sites peak 0.8, N1 sites 0.2."""
    def get(k, i):
        n = len(pts[i]['pm'])
        g = np.exp(-0.5 * ((C.T32 + f * k) / 1.5) ** 2)
        return np.tile(0.8 * g, (n, 1))
    return get


def synthetic_frame(pts, flags):
    """frame_control on synthetic outputs with a known answer (attempt 3)."""
    cases = {'follow': (1.0, 'FOLLOWS'), 'fixed': (0.0, 'ARTEFACT'), 'half': (0.5, 'ARTEFACT'),
             'most': (0.65, 'UNCLEAR'), 'near': (0.85, 'FOLLOWS')}
    out = {}
    for name, (f, want) in cases.items():
        got = frame_control(fake_shift(pts, f), pts, flags)
        out[name] = dict(want=want, got=got['status'], e=got['e'], ok=got['status'] == want)
        print(f'synthetic frame {name} (f {f}): {got["status"]} e {got["e"]} (want {want})')
    flat = frame_control(lambda k, i: np.full((len(pts[i]['pm']), C.NPL), 0.3), pts, flags)['status']
    out['flat'] = dict(want='NO-PEAK', got=flat, ok=flat == 'NO-PEAK')
    print(f'synthetic frame flat: {flat} (want NO-PEAK)')
    return out


def synthetic(pts, flags, xdrop, arms=C.ATTEMPT_ARMS[1], attempt=1):
    """Gate and verdict cases on fake runs built from hecate's F-395 outputs (R9 VES-2242 harness, in memory).
    Attempt 3 adds the frame cases: frame_control on known synthetic outputs, and the verdict of the 'effect' runs
    with the informative arm's frame status ARTEFACT / UNCLEAR / NO-PEAK, and of the 'same' runs with ARTEFACT."""
    H = [d['p_hecate_fwd'].astype(np.float32) for d in pts]
    same = lambda p: p
    blind = lambda p: np.repeat(p.max(1, keepdims=True), p.shape[1], 1)    # depth-blind: column mass on every plane
    cases = {'same': ({}, 'NO-LABEL-EFFECT', None), 'effect': ({'D0': blind, 'D1': blind}, 'LABEL-EFFECT', None),
             'nan_one': ({'D1': lambda p: p * np.nan}, 'VOID', None),
             'zeros_one': ({'D1': lambda p: p * 0}, 'NO-VERDICT', None),
             'lowp_one': ({'D1': lambda p: p * 0.4}, 'NO-VERDICT', None)}
    if attempt == 3:
        eff = {'D0': blind, 'D1': blind}
        cases.update({'effect_follows': (eff, 'LABEL-EFFECT', ('FOLLOWS', 'FOLLOWS')),
                      'effect_artefact': (eff, 'FRAME-ARTEFACT', ('FOLLOWS', 'ARTEFACT')),
                      'effect_unclear': (eff, 'INCONCLUSIVE', ('UNCLEAR', 'FOLLOWS')),
                      'effect_nopeak': (eff, 'INCONCLUSIVE', ('FOLLOWS', 'NO-PEAK')),
                      'same_artefact': ({}, 'FRAME-ARTEFACT', ('ARTEFACT', 'ARTEFACT')),
                      'same_nopeak': ({}, 'NO-LABEL-EFFECT', ('NO-PEAK', 'NO-PEAK')),
                      'lowp_artefact': ({'D1': lambda p: p * 0.4}, 'NO-VERDICT', ('ARTEFACT', 'ARTEFACT'))})
    tr = dict(fit_auc_shell=0.9, steps=1, seconds=1, gpu=None, n_params=1)
    out = {}
    for name, (fs, want, fr) in cases.items():
        runs = {f'{A}{s}': score_run([np.stack([fs.get(f'{B}{s}', same)(h)] * 2) for h in H], tr, pts, flags, xdrop)
                for A, B in zip(arms, ('S', 'D')) for s in C.SEEDS}          # case keys name the roles S, D
        frame = None if attempt != 3 else dict(zip([f'{arms[0]}{s}' for s in C.SEEDS], fr or ('FOLLOWS',) * 2))
        ck = dict(scorer=True, pc_hecate=True) if attempt != 3 else dict(scorer=True, pc_hecate=True, frame_pc=True)
        got = verdict_of(ck, runs, arms, frame)[1]['verdict']
        out[name] = dict(want=want, got=got, ok=got == want)
        print(f'synthetic {name}: {got} (want {want})')
    if attempt == 3:
        runs = {f'{A}{s}': score_run([np.stack([blind(h) if A == 'D' else h] * 2) for h in H], tr, pts, flags, xdrop)
                for A in arms for s in C.SEEDS}
        got = verdict_of(dict(scorer=True, pc_hecate=True, frame_pc=False), runs, arms,
                         {f'{arms[0]}{s}': 'FOLLOWS' for s in C.SEEDS})[1]['verdict']
        out['effect_frame_pc_fail'] = dict(want='VOID', got=got, ok=got == 'VOID')
        print(f'synthetic effect_frame_pc_fail: {got} (want VOID)')
        out['frame_control'] = synthetic_frame(pts, flags)
        return dict(cases=out, ok=all(c['ok'] for k, c in out.items() if k != 'frame_control') and
                    all(c['ok'] for c in out['frame_control'].values()))
    return dict(cases=out, ok=all(c['ok'] for c in out.values()))


def xseg_drop(keys):
    """Indices of sites with another dev segment's training label within 48 vx in 3D (xseg.py; reading only)."""
    f = C.DATA / 'xseg.json'
    if not f.exists():
        return set()
    drop = {tuple(k) for k in json.loads(f.read_text())['drop_keys']}
    return {i for i, k in enumerate(keys) if tuple(k) in drop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--attempt', type=int, choices=sorted(C.ATTEMPT_ARMS), default=1)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    arms = C.ATTEMPT_ARMS[a.attempt]
    out = Path(a.out) if a.out else C.HERE / ('results.json' if a.attempt == 1 else f'attempt{a.attempt}/results.json')
    pts = f395_points()
    keys = site_list()
    assert len(keys) == len(pts), (len(keys), len(pts))
    flags = [k[1] == 'label' for k in keys]
    if (C.DATA / 'prep.json').exists() or not a.check:
        meta = json.loads((C.DATA / 'prep.json').read_text())
        assert meta['site_keys'] == keys, 'prep site order != F-395 site order'
        assert meta['n_points'] == [len(d['pm']) for d in pts], 'prep points != F-395 points'
    res = dict(checks=checks(pts, flags))
    xdrop = xseg_drop(keys)
    res['n_xseg_dropped'] = len(xdrop)
    if a.check:
        res['checks']['synthetic'] = synthetic(pts, flags, xdrop, arms, a.attempt)
        print(json.dumps(res['checks'], indent=1))
        return
    runs = {}
    for arm in arms:
        for seed in C.SEEDS:
            name = f'{arm}{seed}'
            o = np.load(C.DATA / f'out_{name}.npz')
            tr = json.loads((C.DATA / f'train_{name}.json').read_text())
            runs[name] = score_run([o[f'p_{i}'] for i in range(len(pts))], tr, pts, flags, xdrop)   # (2, n, 16)
    res['runs'] = runs
    frame = None
    if a.attempt == 3:
        res['frame_control'] = {}
        for name in [f'{A}{s}' for A in arms for s in C.SEEDS] + list(FRAME_PC):
            f = C.DATA / f'out_{name}_shift.npz'
            if not f.exists():
                assert name[0] != arms[0] and name not in FRAME_PC, f'missing {f}'   # gate and control: required
                continue                                                      # comparison arm: reading only
            z = np.load(f)
            fc = frame_control(lambda k, i: z[f'p{k}_{i}'], pts, flags)
            const = z['const'].astype(np.float64)
            fc['const_input'] = dict(mean_p_by_plane=const.round(4).tolist(), **frame_peak(const),
                                     shell_share=float(const[np.abs(C.T32) <= 1.5].sum() / max(const.sum(), 1e-12)))
            fc['role'] = 'gate' if name[0] == arms[0] else 'known-artefact control' if name in FRAME_PC else 'reading'
            res['frame_control'][name] = fc
            print(f'frame control {name}: {fc["status"]} e {fc["e"]} ({fc["role"]})')
        frame = {n: res['frame_control'][n]['status'] for n in (f'{arms[0]}{s}' for s in C.SEEDS)}
    ck = dict(scorer=res['checks']['scorer']['ok'], pc_hecate=res['checks']['pc_hecate']['ok'])
    if a.attempt == 3:
        ck['frame_pc'] = all(res['frame_control'][n]['status'] == 'ARTEFACT' for n in FRAME_PC)
    res['arms'], res['verdict'] = verdict_of(ck, runs, arms, frame)
    res['attempt'], res['arm_roles'] = a.attempt, dict(zip(('S', 'D'), arms))
    res['site_keys'] = keys
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('checks', 'arms', 'verdict')}, indent=1))


if __name__ == '__main__':
    main()
