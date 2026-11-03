"""VES-2344 scorer: label-free held-out test of label J vs D on Will's fresh PHerc1667 surface (README "Step 0.3").

The leak and verdict rules are VES-2321's, imported unchanged from its code directory (sha256 per file pinned in
PINS): score.leak_site / summarise / score_run / frame_control / verdict_of, score_heldout.fold_status / overall /
MASS_PER_SITE, run_t4.scorer_check. New here: the site power rule, the gates and controls (R0, PC-S, G1 site sha,
G2 circularity, replay, orientation, PC-H, PC-P), the fold-ensemble pool over scored folds, the patch rule and the
held-out WIN bar.

  python -I score_lf.py --code DIR --dryrun        synthetic cases on PHerc0139 F-395 points (no 1667 output read)
  python -I score_lf.py --code DIR --check         gates (bars applied) + SC, PC-H, PC-P of a step-1 bundle, no J/D
                                                   output read; exit 1 on any fail
  python -I score_lf.py --code DIR                 step-1 scored run -> results.json (only after VES-2321 WIN + R9)

DIR = the pherc1667/ folder of this repository (pinned below). Run with
RUN_1667=RUN2321 (the run gate fails otherwise).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root; data defaults live under ROOT / 'data'
DATA = Path(os.environ.get('RUN_LABELFREE', ROOT / 'data/labelfree1667'))
# VES-2321 run whose nets and fit AUCs step 1 uses (R9 VES-2368 FIX 1): attempt 2. Re-pin to the run of VES-2321's WIN.
RUN2321 = Path(os.environ.get('RUN_1667', ROOT / 'data/run1667'))
INK2UM_SHA = '36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0'   # r152_3ddec_v2_l5_epoch13.ckpt
MAX_CAND, MAX_PER_CLASS = 1500, 400            # README "Candidates" / "Site classes" (attempt 2: 3000 candidates)
INK_FRAC, NULL_FRAC, NULL_MAX_S = 0.25, 0.02, 0.3
REPLAY_TOL, N_REPLAY = 5e-3, 4                 # README "Replay gate" (VES-2321 attempt4/infer4.py rule)
P0139 = Path(os.environ.get('POINTS_0139', ROOT / 'data/points0139')) / 'points.npz'      # F-395 points (PHerc0139), dry run only
P0139_SHA = '8987c20a9be633828d9de6bede487f6f2c1d7ac7341f953b0fd700e4afe88cb7'
# team/ves2321-label-j-1667. Step-1 re-pin (VES-2424): VES-2321's WIN (attempt 2, 1ac26432) ran the code of b1532b1d
# (R9 VES-2330); every VES-2321 file below is byte-identical at a3f082b6, b1532b1d and 1ac26432. run_t23.py stays at
# b1532b1d's version: master's later VES-2267 edit (in 1ac26432 via a merge) is transport only and its nested-tuple
# except clause raises TypeError on every exception (VES-2409). Step 1 adds prep, train, sites4 and infer4 (step1.py).
PINS_COMMIT = 'b1532b1d'
# Public release: re-pinned to this repository's pherc1667/ and shared/ files. They differ from b1532b1d only in
# the data-path lines (env vars with repo-relative defaults) and, for run_t23.py, the transport-only retry-tuple fix
# (PROVENANCE.md).
PINS = {
    'prep.py': '682b85265370631a03958a15db9753ecdbd0a19d5d04cbf8c198b8780e9922e0',
    'train.py': 'c7acbb2bf9bb2a8ee49578b022638a35db2c935b0eb7601fc9cbaa943376290a',
    'attempt4/sites4.py': 'c2067ce1842e93d8a6f1d7853c39a867495c3bc666203465cb1ffe5ee6298eef',
    'attempt4/infer4.py': 'dd53c0be89bb36133208121c8052b752ac0f966c45e8c662e7c1a87e3aeaf105',
    'common.py': '4945f0f343dfe50cbde317f49e33e0a2af818a8e395f1c5f4feaadda3a6999fe',
    'score.py': '334c9f3f938bc391b2d63a3d93ffa6826852ac031d39683d45894987f524b80c',
    'heldout/score_heldout.py': '3841a0e3a3359e4c1e228bd5e161566cdcec7b02e225fed7a8656a00f1daac61',
    'patch1667.py': '1a502b3fefd656c5be1684454a9d5a5f6b56756e9c0218e5a3f22f114b203853',
    'prep_data.py': '5c7119c456a7112d710c94cae5d18733daae69880218244b4db03d1c22fb3a70',
    '../shared/run_t4.py':
        'de198c1579d1a28fc154f098a34b3ea72a7269ce68b40c9fe92d9065e44c49d7',
    '../shared/run_t23.py':
        '1f5309e1ba1d75ecf370f5bc4d5c05c92d81f04fd9b66ff607b82f80e02239ef',
}
# README "Step 0.3" bars
MIN_SITES, MIN_PATCHES = 200, 3                # power: per class
R0_MIN, PCS_AUC = 0.90, 0.70
G2_K, G2_ADD = 1.5, 0.01                       # dL_src <= 1.5 dL_lab + 0.01
PCH_RATIO, PCH_AUC = 0.5, 0.60
WIN_RATIO = 0.5                                # pooled ensemble J L <= 0.5 x J L0
PATCH_MIN = 50                                 # patch rule: patches with >= 50 source-ink sites
SRC_INK, SRC_NULL = 'src_ink', 'src_null'
SCORED = ('WIN', 'LOSS', 'UNCLEAR', 'ARTEFACT')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rules(code):
    """Import VES-2321's modules after checking every pinned file."""
    code = Path(code).resolve()
    bad = {k: sha(code / k) for k, v in PINS.items() if sha(code / k) != v}
    assert not bad, f'VES-2321 code differs from the pins ({PINS_COMMIT}): {bad}'
    sys.path.insert(0, str(code))
    import common as C          # noqa: E402
    import score as S           # noqa: E402
    spec = importlib.util.spec_from_file_location('ves2321_score_heldout', code / 'heldout/score_heldout.py')
    SH = importlib.util.module_from_spec(spec)
    sys.modules['ves2321_score_heldout'] = SH
    spec.loader.exec_module(SH)
    return C, S, SH


# ---------------------------------------------------------------- pure rules (this task)
def power(classes, patches):
    """README power floor: >= 200 sites per class, each class from >= 3 patches."""
    out = {}
    for c in (SRC_INK, SRC_NULL):
        ps = [p for k, p in zip(classes, patches) if k == c]
        out[c] = dict(n=len(ps), n_patches=len(set(ps)), by_patch={str(p): ps.count(p) for p in sorted(set(ps))})
    out['ok'] = all(out[c]['n'] >= MIN_SITES and out[c]['n_patches'] >= MIN_PATCHES for c in (SRC_INK, SRC_NULL))
    return out


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and bool(np.isfinite(x))


def g2_rule(dl_src, dl_lab):
    ok = bool(_finite(dl_src) and _finite(dl_lab) and dl_src <= G2_K * dl_lab + G2_ADD)
    return dict(dL_src=dl_src, dL_lab=dl_lab, bar=G2_K * dl_lab + G2_ADD if _finite(dl_lab) else None, ok=ok)


def site_rules(info, pts):
    """README "Site source S / Site classes / Candidates" on the stored per-site numbers (R9 VES-2368 FIX 7).
    info['sites'][j]: n_box_px (box px sampled, every 2nd px), n_s05 (of them with S >= 0.5), max_S; a site's points
    are its box px with S >= 0.5 (source-ink) or all box px (source-null)."""
    bad = []
    keys, sites = info['site_keys'], info.get('sites')
    if info.get('model_sha') != INK2UM_SHA:
        bad.append('model sha')
    nc = info.get('n_candidates')
    if not (isinstance(nc, int) and 0 < nc <= MAX_CAND):
        bad.append(f'n_candidates {nc}')
    if not isinstance(sites, list) or len(sites) != len(keys) or len(pts) != len(keys):
        return dict(ok=False, bad=bad + ['site table length'])
    for c in (SRC_INK, SRC_NULL):
        if sum(k[1] == c for k in keys) > MAX_PER_CLASS:
            bad.append(f'{c} > {MAX_PER_CLASS}')
    for j, (k, s, d) in enumerate(zip(keys, sites, pts)):
        nb, n5, mx = s.get('n_box_px'), s.get('n_s05'), s.get('max_S')
        if not (isinstance(nb, int) and nb > 0 and isinstance(n5, int) and 0 <= n5 <= nb and _finite(mx)):
            bad.append(f'site {j}: numbers')
        elif k[1] == SRC_INK:
            if not (n5 / nb >= INK_FRAC and len(d['use']) == n5):
                bad.append(f'site {j}: ink rule')
        elif k[1] == SRC_NULL:
            if not (n5 / nb < NULL_FRAC and mx < NULL_MAX_S and len(d['use']) == nb):
                bad.append(f'site {j}: null rule')
        else:
            bad.append(f'site {j}: class {k[1]}')
    return dict(ok=not bad, bad=bad[:20], n_bad=len(bad))


def git_g1(sha_file, sites_sha, outputs):
    """G1 (R9 VES-2368 FIX 6): the site sha file is tracked, unchanged vs HEAD, holds sites_sha, and the commit that
    last wrote it is older than every Will output file (points_lf.npz holds hecate's output, so it counts)."""
    sha_file = Path(sha_file)
    cwd = sha_file.parent

    def git(*a):
        return subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True)
    out = dict(exists=sha_file.exists())
    out['sha_match'] = bool(out['exists'] and sha_file.read_text().split()[:1] == [sites_sha])
    out['tracked'] = git('ls-files', '--error-unmatch', sha_file.name).returncode == 0
    out['clean'] = out['tracked'] and git('diff', '--quiet', 'HEAD', '--', sha_file.name).returncode == 0
    r = git('log', '-1', '--format=%H %ct', '--', sha_file.name)
    parts = r.stdout.split() if r.returncode == 0 else []
    out['commit'], ct = (parts[0], int(parts[1])) if len(parts) == 2 else (None, None)
    mt = [Path(p).stat().st_mtime for p in outputs if Path(p).exists()]
    out['n_outputs'], out['first_output_mtime'] = len(mt), min(mt) if mt else None
    out['before_outputs'] = bool(ct is not None and (not mt or ct < min(mt)))
    out['ok'] = bool(out['sha_match'] and out['clean'] and out['commit'] and out['before_outputs'])
    return out


def g2_measure(S, pts, flags, outs, src, names=('J0', 'J1', 'D0', 'D1')):
    """G2 on VES-2321's label sites. outs[n][i]: (2, n_pts, 16) output of the net that held site i's segment out;
    src[i]: (n_pts,) site-source S at the points. dL = mean D L - mean J L (phase mean, score_run's primary)."""
    def L_of(n, keep):
        per = []
        for i, (d, f) in enumerate(zip(pts, flags)):
            if not f:
                continue
            dd = dict(d, dfn=d['dfn'] & keep[i])
            per.append(S.leak_site(outs[n][i].mean(0), dd))
        num, num0, den = (sum(s[k] for s in per) for k in ('num', 'num0', 'den'))
        return (num / den, num0 / den) if den > 0 else (None, None)
    allk = [np.ones(len(d['pm']), bool) for d in pts]
    srck = [s >= 0.5 for s in src]
    res = {}
    for tag, keep in (('lab', allk), ('src', srck)):
        LL = {n: L_of(n, keep) for n in names}
        L = {n: v[0] for n, v in LL.items()}
        res[f'L_{tag}'], res[f'L0_{tag}'] = L, {n: v[1] for n, v in LL.items()}
        res[tag] = None if None in L.values() else float(np.mean([L['D0'], L['D1']]) - np.mean([L['J0'], L['J1']]))
    return dict(g2_rule(res['src'], res['lab']), readings={k: v for k, v in res.items() if k.startswith('L')})


def pool_scored(folds, get_site, get_shift, names, n_sites):
    """Fold-ensemble pool (README "Pool"): per site and phase the mean p of the scored folds' nets of one run name;
    frame outputs averaged the same way. Returns (get_site_pool(n, i), get_shift_pool(n) -> fn(k, i)), fold list."""
    fs = [f for f, st in folds.items() if st['status'] in SCORED]

    def site(n, i):
        return np.mean([get_site(f, n, i) for f in fs], 0)

    def shift(n):
        gs = [get_shift(f, n) for f in fs]
        return lambda k, i: np.mean([g(k, i) for g in gs], 0)
    return site, shift, fs


def patch_rule(S, pts, flags, patches, runs_site):
    """In every patch with >= PATCH_MIN source-ink sites: mean J L < mean D L (pooled ensemble outputs)."""
    out = {}
    for p in sorted(set(patches)):
        idx = [i for i, q in enumerate(patches) if q == p]
        n_ink = sum(flags[i] for i in idx)
        if n_ink < PATCH_MIN:
            out[str(p)] = dict(n_ink=n_ink, applies=False)
            continue
        L, L0 = {}, {}
        for n, ph in runs_site.items():
            per = [S.leak_site(ph[i].mean(0), pts[i]) for i in idx if flags[i]]
            den = sum(s['den'] for s in per)
            L[n] = sum(s['num'] for s in per) / den if den > 0 else None
            L0[n] = sum(s['num0'] for s in per) / den if den > 0 else None
        ok = None not in L.values() and np.mean([L['J0'], L['J1']]) < np.mean([L['D0'], L['D1']])
        out[str(p)] = dict(n_ink=n_ink, applies=True, L=L, L0=L0, ok=bool(ok))
    applied = [v for v in out.values() if v['applies']]
    return dict(patches=out, ok=bool(applied and all(v['ok'] for v in applied)), n_applied=len(applied))


# ---------------------------------------------------------------- the scored decision
def gates_of(B):
    """Gate bars on B.ext (r0, pcs_auc, g2 dict recomputed by g2_rule, and the booleans set by Bundle / Synth).
    Every gate must be exactly True (a string or a missing value fails)."""
    pw = power(B.classes, B.patches)
    e = B.ext
    gates = dict(power=pw['ok'], R0=bool(_finite(e['r0']) and e['r0'] >= R0_MIN),
                 PC_S=bool(_finite(e['pcs_auc']) and e['pcs_auc'] >= PCS_AUC), G1_site_sha=e['site_sha_ok'] is True,
                 G2=g2_rule(e['g2'].get('dL_src'), e['g2'].get('dL_lab'))['ok'], replay=e['replay_ok'] is True,
                 orientation=e['orient_ok'] is True, full_draw=e['full_draw'] is True,
                 site_rules=e.get('site_rules_ok') is True, run_dir=e.get('run_ok') is True,
                 nets=e.get('nets_ok') is True)                  # missing -> fail (R9 VES-2384 note 1)
    return pw, gates


def controls_of(S, B):
    """SC, PC-H (hecate on Will's sites; non-finite output -> fail, R9 VES-2368 FIX 8), PC-P. No J/D output read."""
    sc = S.t4.scorer_check()
    H = [d['p_hecate_fwd'].astype(np.float32) for d in B.pts]
    if all(np.isfinite(h[d['dfn']]).all() for h, d in zip(H, B.pts)):
        sm = S.summarise([S.leak_site(h, d) for h, d in zip(H, B.pts)], B.flags, H, B.pts)
        pch = dict(L=sm['label']['L'], L0=sm['label']['L0'], L_ci=sm['label']['L_ci'], ink_auc=sm['ink_auc'])
        pch['ok'] = bool(pch['L'] is not None and pch['L0'] and pch['L'] <= PCH_RATIO * pch['L0'] and
                         (pch['ink_auc'] or 0) >= PCH_AUC)
    else:
        pch = dict(L=None, L0=None, ink_auc=None, ok=False, why='non-finite hecate output on defined points')
    pcp = {n: S.frame_control(B.pshift(n), B.pts, B.flags)['status'] for n in S.FRAME_PC}
    ctrl = dict(scorer=bool(sc['ok']), PC_H=pch, PC_P=pcp, PC_P_ok=all(v == 'ARTEFACT' for v in pcp.values()))
    ctrl['ok'] = bool(ctrl['scorer'] and pch['ok'] and ctrl['PC_P_ok'])
    return ctrl


def decide(C, S, SH, B):
    """B: bundle with pts, flags (True = source-ink), classes, patches, folds, names, get_site(f, n, i) ->
    (2, n_pts, 16), get_shift(f, n) -> fn(k, i) -> (n_pts, 32), pshift(n) -> fn(k, i), tr[f][n], ext (gate
    inputs: r0, pcs_auc, g2 dict, replay_ok, site_sha_ok, orient_ok, full_draw [, site_rules_ok, run_ok, nets_ok])."""
    res = {}
    e = B.ext
    pw, gates = gates_of(B)
    res.update(power=pw, gates=gates, g2=dict(e['g2'], recomputed=g2_rule(e['g2'].get('dL_src'), e['g2'].get('dL_lab'))),
               r0=e['r0'], pcs_auc=e['pcs_auc'], gate_detail=e.get('detail'))
    if not all(gates.values()):
        res['verdict'] = 'VOID'
        res['heldout_win'] = dict(ok=False, why='gate: ' + ', '.join(k for k, v in gates.items() if not v))
        return res
    # pooled controls on Will's sites
    ctrl = controls_of(S, B)
    pch = ctrl['PC_H']
    res['controls'] = ctrl
    # per fold: VES-2321 fold_status with the pooled controls (the same sites for every fold)
    n_ink = sum(B.flags)
    keep, S.MIN_MASS = S.MIN_MASS, SH.MASS_PER_SITE * n_ink
    try:
        folds = {}
        for f in B.folds:
            runs = {n: S.score_run([B.get_site(f, n, i) for i in range(len(B.pts))], B.tr[f][n], B.pts, B.flags,
                                   set()) for n in B.names}
            frame = {n: S.frame_control(B.get_shift(f, n), B.pts, B.flags)['status'] for n in B.names
                     if runs[n]['finite']}
            st = SH.fold_status(dict(ok=ctrl['ok']), runs, frame)
            st['runs'] = {n: dict(L=(r.get('label') or {}).get('L'), L0=(r.get('label') or {}).get('L0'),
                                  L_ci=(r.get('label') or {}).get('L_ci'), ink_auc=r.get('ink_auc'),
                                  measurable=r.get('measurable')) for n, r in runs.items()}
            st['frame'] = frame
            folds[f] = st
        res['folds'] = folds
        res['fold_status'] = {f: v['status'] for f, v in folds.items()}
        # pool over the scored folds
        pooled_v, arms, pruns = None, None, None
        site, shift, fs = pool_scored(folds, B.get_site, B.get_shift, B.names, len(B.pts))
        if fs:
            ph = {n: [site(n, i) for i in range(len(B.pts))] for n in B.names}
            tr = {n: dict(fit_auc_shell=min(B.tr[f][n]['fit_auc_shell'] for f in fs), steps=1, seconds=0, gpu=None,
                          n_params=0) for n in B.names}
            pruns = {n: S.score_run(ph[n], tr[n], B.pts, B.flags, set()) for n in B.names}
            pfr = {f'J{s}': S.frame_control(shift(f'J{s}'), B.pts, B.flags)['status'] for s in C.SEEDS}
            ck = dict(scorer=ctrl['scorer'], pc_hecate=pch['ok'], frame_pc=ctrl['PC_P_ok'])
            arms, pooled_v = S.verdict_of(ck, pruns, C.ATTEMPT_ARMS[3], pfr)
            res['pooled'] = dict(folds=fs, frame=pfr, arms=arms, verdict=pooled_v,
                                 runs={n: dict(L=(r.get('label') or {}).get('L'), L0=(r.get('label') or {}).get('L0'),
                                               L_ci=(r.get('label') or {}).get('L_ci'),
                                               L_N1=(r.get('N1') or {}).get('L'), L0_N1=(r.get('N1') or {}).get('L0'),
                                               ink_auc=r.get('ink_auc'))
                                       for n, r in pruns.items()})
            res['patch_rule'] = patch_rule(S, B.pts, B.flags, B.patches, ph)
    finally:
        S.MIN_MASS = keep
    sh_folds = {f: folds.get(f, dict(status='NOT-RUN')) for f in SH.FOLDS}
    res['verdict'] = SH.overall(ctrl['ok'], sh_folds, pooled_v)
    hw = dict(confirmed=res['verdict'] == 'CONFIRMED', controls=ctrl['ok'])
    if pruns:
        LJ = [pruns[f'J{s}']['label']['L'] for s in C.SEEDS]
        L0J = [pruns[f'J{s}']['label']['L0'] for s in C.SEEDS]
        ok = all(x is not None for x in LJ + L0J)
        hw.update(J_L=float(np.mean(LJ)) if ok else None, J_L0=float(np.mean(L0J)) if ok else None)
        hw['J_below_half_L0'] = bool(ok and np.mean(LJ) <= WIN_RATIO * np.mean(L0J))
        hw['patch_rule'] = res['patch_rule']['ok']
    else:
        hw.update(J_below_half_L0=False, patch_rule=False)
    hw['ok'] = bool(hw['confirmed'] and hw['controls'] and hw['J_below_half_L0'] and hw['patch_rule'])
    res['heldout_win'] = hw
    return res


def reading_s08(S, B, res, thr=0.8):
    """Reading (no bar): pooled-ensemble L and L0 of each run on the source-ink points with S >= thr only."""
    fs = (res.get('pooled') or {}).get('folds')
    sp = getattr(B, 's_pts', None)
    if not fs or not sp or len(sp) != len(B.pts) or any(len(s) != len(d['pm']) for s, d in zip(sp, B.pts)):
        return None
    out = {}
    for n in B.names:
        per = []
        for i, (d, f) in enumerate(zip(B.pts, B.flags)):
            if f:
                p = np.mean([B.get_site(g, n, i) for g in fs], 0).mean(0)
                per.append(S.leak_site(p, dict(d, dfn=d['dfn'] & (sp[i] >= thr))))
        den = sum(x['den'] for x in per)
        out[n] = dict(L=sum(x['num'] for x in per) / den if den > 0 else None,
                      L0=sum(x['num0'] for x in per) / den if den > 0 else None, mass=den)
    return dict(thr=thr, runs=out)


# ---------------------------------------------------------------- step-1 bundle (files written by the step-1 pipeline)
class Bundle:
    """Step-1 inputs under DATA (written by the step-1 pipeline, which gets its own delta R9):
      sites_lf.json   site_keys [[patch, class, k, r0, c0]], sites [{n_box_px, n_s05, max_S}], model_sha,
                      n_candidates, limit (None = full draw); its sha256 in sites_lf.sha256 next to this file (G1)
      points_lf.npz   dfn/use/pm/pp/p_hecate_fwd_<i> per site, plus site_sha (= G1 sha) and n_sites
      gates_lf.json   r0, pcs_auc, g2 {dL_src, dL_lab} (from g2_measure), orientation_applied, flips {patch: bool}
      infer_lf.json   run_dir, site_sha, nets {name: sha256}, replay {name: {d, n_sites}} for <n>_h<f> and P0, P1
      out_<n>_h<f>_lf.npz (p_i), out_<n>_h<f>_lf_shift.npz (p<k>_i), out_P<s>_lf_shift.npz
    Fit AUCs and nets come from RUN2321 (heldout/train_<n>_h<f>.json, heldout/net_<n>_h<f>.pt, net_P<s>.pt)."""

    def __init__(self, C, SH, read_outputs=True, data=None, run=None, sha_file=None, ct_json=None):
        self.data = data = Path(data or DATA)
        run = Path(run or RUN2321)
        sha_file = Path(sha_file or HERE / 'sites_lf.sha256')
        info = json.loads((data / 'sites_lf.json').read_text())
        site_sha = sha(data / 'sites_lf.json')
        z = np.load(data / 'points_lf.npz')
        self.keys = info['site_keys']                    # [patch, class, k, r0, c0]
        self.pts = [{k: z[f'{k}_{i}'] for k in ('dfn', 'use', 'pm', 'pp', 'p_hecate_fwd')} for i in range(len(self.keys))]
        # S at each site's points (its stored use px, in box order = the points order); reading "L on S >= 0.8" only
        self.s_pts = [np.asarray(sv, float)[np.asarray(u, bool)] for sv, u in zip(info.get('S', []), info.get('use', []))]
        self.classes = [k[1] for k in self.keys]
        self.patches = [k[0] for k in self.keys]
        self.flags = [c == SRC_INK for c in self.classes]
        self.folds, self.names = tuple(SH.FOLDS), [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
        g = json.loads((data / 'gates_lf.json').read_text())
        inf = json.loads((data / 'infer_lf.json').read_text()) if (data / 'infer_lf.json').exists() else {}
        det = {}
        # FIX 1: one pinned VES-2321 run for fit AUCs and nets; net sha256 recorded at inference = files in that run
        det['run'] = dict(pinned=str(run), C_DATA=str(C.DATA), infer=inf.get('run_dir'))
        run_ok = (Path(C.DATA).resolve() == run.resolve() and inf.get('run_dir') is not None and
                  Path(inf['run_dir']).resolve() == run.resolve())
        net_files = {f'{n}_h{f}': run / 'heldout' / f'net_{n}_h{f}.pt' for f in self.folds for n in self.names}
        net_files.update({p: run / f'net_{p}.pt' for p in ('P0', 'P1')})
        rec = inf.get('nets') or {}
        miss = [k for k, p in net_files.items() if not p.exists() or rec.get(k) != sha(p)]
        miss += [f'train_{n}_h{f}' for f in self.folds for n in self.names
                 if not (run / 'heldout' / f'train_{n}_h{f}.json').exists()]
        nets_ok = set(rec) == set(net_files) and not miss
        det['nets'] = dict(n_expected=len(net_files), mismatch=miss[:10])
        # FIX 2: tolerance and net set fixed here, >= N_REPLAY replayed sites per net
        rp = inf.get('replay') or {}
        replay_ok = set(rp) == set(net_files) and all(
            isinstance(v, dict) and _finite(v.get('d')) and v['d'] <= REPLAY_TOL and isinstance(v.get('n_sites'), int)
            and v['n_sites'] >= N_REPLAY for v in rp.values())
        # FIX 6: G1 site sha tracked, clean, committed before every Will output; points and inference bound to it
        outs = [data / 'points_lf.npz'] + sorted(data.glob('out_*_lf*.npz'))
        g1 = git_g1(sha_file, site_sha, outs)
        zsha = str(z['site_sha']) if 'site_sha' in z.files else None
        zn = int(z['n_sites']) if 'n_sites' in z.files else None
        g1.update(points_sha=zsha == site_sha, points_n=zn == len(self.keys), infer_sha=inf.get('site_sha') == site_sha)
        site_sha_ok = bool(g1['ok'] and g1['points_sha'] and g1['points_n'] and g1['infer_sha'])
        det['G1'] = g1
        # FIX 7: site classes, caps and the site-source model sha
        sr = site_rules(info, self.pts)
        det['site_rules'] = sr
        # FIX 4: orientation = the step-0.2 flips exactly
        ct = json.loads(Path(ct_json or HERE / 'results_ct.json').read_text())
        want = {p: bool(v['flip']) for p, v in ct['orientation'].items()}
        det['orientation'] = dict(want=want, got=g.get('flips'))
        orient_ok = g.get('orientation_applied') is True and g.get('flips') == want
        self.ext = dict(r0=g.get('r0'), pcs_auc=g.get('pcs_auc'), g2=g.get('g2') or {}, replay_ok=replay_ok,
                        site_sha_ok=site_sha_ok, orient_ok=orient_ok,
                        full_draw='limit' in info and info['limit'] is None,          # FIX 5: missing key fails
                        site_rules_ok=sr['ok'], run_ok=run_ok, nets_ok=nets_ok, detail=det)
        self.tr = {f: {n: json.loads(p.read_text()) if (p := run / 'heldout' / f'train_{n}_h{f}.json').exists()
                       else dict(fit_auc_shell=None) for n in self.names} for f in self.folds}
        self._z = {}
        self.read_outputs = read_outputs

    def _ld(self, name):
        assert self.read_outputs, 'output read in --check mode'
        if name not in self._z:
            self._z[name] = np.load(self.data / name)
        return self._z[name]

    def get_site(self, f, n, i):
        return self._ld(f'out_{n}_h{f}_lf.npz')[f'p_{i}']

    def get_shift(self, f, n):
        z = self._ld(f'out_{n}_h{f}_lf_shift.npz')
        return lambda k, i: z[f'p{k}_{i}']

    def pshift(self, n):
        z = np.load(self.data / f'out_{n}_lf_shift.npz')   # P0, P1: the known-artefact control, read in --check too
        return lambda k, i: z[f'p{k}_{i}']


# ---------------------------------------------------------------- dry run on PHerc0139 (synthetic outputs)
class Synth:
    """F-395 0139 points (400 sites: per dev segment 40 label then 40 N1) as a stand-in for Will's sites. flags =
    the 0139 label/N1 role (stands for source-ink / source-null); 'patches' = segment // 2 (80, 80, 40 ink sites),
    so the patch rule applies to two patches. Outputs are built from hecate's 0139 outputs: J-like = hecate,
    D-like = depth-blind (column max on every plane). Frame outputs: score.fake_shift (f = 1 follows, f = 0 stays)."""

    def __init__(self, C, S, SH, case):
        assert sha(P0139) == P0139_SHA, 'F-395 points changed'
        z = np.load(P0139)
        n = len({k.rsplit('_', 1)[1] for k in z.files})
        self.pts = [{k: z[f'{k}_{i}'] for k in ('dfn', 'use', 'pm', 'pp', 'p_hecate_fwd')} for i in range(n)]
        assert n == 400
        seg = [i // 80 for i in range(n)]
        self.flags = [(i % 80) < 40 for i in range(n)]
        self.classes = [SRC_INK if f else SRC_NULL for f in self.flags]
        self.patches = [s // 2 for s in seg]
        self.folds, self.names = tuple(SH.FOLDS), [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
        self.C, self.S, self.case = C, S, case
        H = [d['p_hecate_fwd'].astype(np.float32) for d in self.pts]
        blind = [np.repeat(h.max(1, keepdims=True), h.shape[1], 1) for h in H]
        lk = [S.t4.leak_mask(d['pm'], d['pp']) for d in self.pts]
        leak_only = [b * m for b, m in zip(blind, lk)]               # all column mass on the leak planes
        J, D = H, blind
        if case == 'same':
            D = H
        elif case == 'reversed':
            J, D = blind, H
        elif case == 'partial':
            J, D = blind, leak_only
        elif case == 'low_auc':
            D = [b * 0.4 for b in blind]
        self.J, self.D = J, D
        if case == 'one_patch_reversed':
            self.J = [blind[i] if self.patches[i] == 0 else H[i] for i in range(n)]
            self.D = [H[i] if self.patches[i] == 0 else blind[i] for i in range(n)]
        if case == 'pch_fail':
            for d, b in zip(self.pts, blind):
                d['p_hecate_fwd'] = b
        if case == 'pch_nan':
            for d in self.pts:
                d['p_hecate_fwd'] = np.full(d['p_hecate_fwd'].shape, np.nan, np.float32)
        if case == 'power_short':
            keep = [i for i in range(n) if not self.flags[i] or (i % 80) < 25]      # 125 ink sites
            for a in ('pts', 'flags', 'classes', 'patches'):
                setattr(self, a, [getattr(self, a)[i] for i in keep])
            self.J, self.D = [self.J[i] for i in keep], [self.D[i] for i in keep]
        self.tr = {f: {nm: dict(fit_auc_shell=0.9, steps=1, seconds=1, gpu=None, n_params=1) for nm in self.names}
                   for f in self.folds}
        g2 = g2_rule(0.05, 0.10)
        if case == 'g2_fail':
            g2 = g2_rule(0.30, 0.10)
        self.ext = dict(r0=0.95 if case != 'r0_fail' else 0.80, pcs_auc=0.80 if case != 'pcs_fail' else 0.65, g2=g2,
                        replay_ok=True, site_sha_ok=case != 'site_sha_fail', orient_ok=True, full_draw=True,
                        site_rules_ok=True, run_ok=True, nets_ok=True)

    def get_site(self, f, n, i):
        k = self.folds.index(f)
        if self.case == 'fold_nan' and f == self.folds[2] and n == 'D1':
            return np.full((2,) + self.D[i].shape, np.nan, np.float32)
        base = (self.J if n[0] == 'J' else self.D)[i]
        g = 1.0 - 0.01 * k - 0.005 * int(n[1])                       # small fold / seed differences
        return np.stack([base * g, base * g])

    def get_shift(self, f, n):
        fol = 0.0 if (self.case == 'j_artefact' and n[0] == 'J') else 1.0
        return self.S.fake_shift(self.pts, fol)

    def pshift(self, n):
        return self.S.fake_shift(self.pts, 1.0 if self.case == 'pcp_fail' else 0.0)


DRY_CASES = {      # case -> expected (verdict, held-out WIN); None = not asserted
    'effect': ('CONFIRMED', True),
    'same': ('NOT-CONFIRMED', False),
    'reversed': ('NOT-CONFIRMED', False),
    'partial': ('CONFIRMED', False),
    'one_patch_reversed': (None, False),
    'j_artefact': ('NOT-CONFIRMED', False),
    'fold_nan': ('CONFIRMED', True),
    'low_auc': ('NO-VERDICT', False),
    'power_short': ('VOID', False),
    'r0_fail': ('VOID', False),
    'pcs_fail': ('VOID', False),
    'g2_fail': ('VOID', False),
    'site_sha_fail': ('VOID', False),
    'pch_fail': ('VOID', False),
    'pcp_fail': ('VOID', False),
    'pch_nan': ('VOID', False),
}


def bundle_cases(C, SH):
    """R9 VES-2368 regression: a stub step-1 bundle (F-395 0139 points as sites, fake nets in a temp run dir, the
    site sha file in a temp git repo) must pass every Bundle gate; each tamper must fail its gate (and only that one
    where the tamper is local). No 1667 output exists or is read."""
    import shutil
    import tempfile
    import time
    z = np.load(P0139)
    n = 400
    tmp = Path(tempfile.mkdtemp(prefix='ves2344-bundle-'))
    data, run, repo = tmp / 'data', tmp / 'run', tmp / 'repo'
    for d in (data, run / 'heldout', repo):
        d.mkdir(parents=True)
    names = [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
    nets = {}
    for f in SH.FOLDS:
        for nm in names:
            (run / 'heldout' / f'net_{nm}_h{f}.pt').write_bytes(f'{nm}{f}'.encode())
            (run / 'heldout' / f'train_{nm}_h{f}.json').write_text(json.dumps(dict(fit_auc_shell=0.9)))
            nets[f'{nm}_h{f}'] = sha(run / 'heldout' / f'net_{nm}_h{f}.pt')
    for p in ('P0', 'P1'):
        (run / f'net_{p}.pt').write_bytes(p.encode())
        nets[p] = sha(run / f'net_{p}.pt')
    keys = [[f'patch_{(i // 80) % 3}', SRC_INK if (i % 80) < 40 else SRC_NULL, i, 0, 0] for i in range(n)]
    nuse = [len(z[f'use_{i}']) for i in range(n)]
    sites = [dict(n_box_px=2 * u, n_s05=u, max_S=0.9) if k[1] == SRC_INK else dict(n_box_px=u, n_s05=0, max_S=0.1)
             for k, u in zip(keys, nuse)]
    info = dict(site_keys=keys, sites=sites, model_sha=INK2UM_SHA, n_candidates=1500, limit=None)
    ct = json.loads((HERE / 'results_ct.json').read_text())
    flips = {p: bool(v['flip']) for p, v in ct['orientation'].items()}

    def git(*a):
        subprocess.run(['git', *a], cwd=repo, check=True, capture_output=True)

    def build(tamper=None):
        for p in data.iterdir():
            p.unlink()
        inf_ = json.loads(json.dumps(info))
        if tamper == 'no_limit':
            del inf_['limit']
        if tamper == 'model_sha':
            inf_['model_sha'] = '0' * 64
        if tamper == 'null_rule':
            inf_['sites'][45]['max_S'] = 0.35
        if tamper == 'ink_rule':
            inf_['sites'][0]['n_box_px'] = 5 * inf_['sites'][0]['n_s05']
        (data / 'sites_lf.json').write_text(json.dumps(inf_))
        ssha = sha(data / 'sites_lf.json')
        sf = repo / 'sites_lf.sha256'
        if sf.exists():
            sf.unlink()
            subprocess.run(['git', 'rm', '-q', '--cached', sf.name], cwd=repo, capture_output=True)
        sf.write_text(ssha + '\n')
        if tamper != 'sha_untracked':
            git('add', sf.name)
            git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 's', '--allow-empty')
        if tamper == 'sha_dirty':
            sf.write_text(ssha + '\nx\n')
        time.sleep(1.1)                                   # commit time (s) strictly before the outputs
        arr = {f'{k}_{i}': z[f'{k}_{i}'] for i in range(n) for k in ('dfn', 'use', 'pm', 'pp', 'p_hecate_fwd')}
        arr.update(site_sha=np.array('0' * 64 if tamper == 'points_sha' else ssha), n_sites=np.array(n))
        np.savez(data / 'points_lf.npz', **arr)
        if tamper == 'sha_after_outputs':                 # re-commit the sha file after the outputs exist
            time.sleep(1.1)
            git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'late', '--allow-empty')
            sf.write_text(ssha + '\n ')
            git('add', sf.name)
            git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'late2')
        g2 = dict(dL_src=0.05, dL_lab=0.10, ok=True)
        if tamper == 'g2_flag':
            g2 = dict(dL_src=0.50, dL_lab=0.0, ok=True)
        gl = dict(r0=0.95, pcs_auc=0.8, g2=g2, orientation_applied=True, flips=dict(flips))
        if tamper == 'orient_str':
            gl['orientation_applied'] = 'no'
        if tamper == 'orient_flips':
            gl['flips']['patch_5'] = False
        (data / 'gates_lf.json').write_text(json.dumps(gl))
        rp = {k: dict(d=1e-4, n_sites=4) for k in nets}
        inf = dict(run_dir=str(run), site_sha=ssha, nets=dict(nets), replay=rp)
        if tamper == 'replay_tol':
            inf['replay'] = {k: dict(d=0.9, n_sites=4) for k in nets}
            inf['replay_tol'] = 1.0
        if tamper == 'replay_keys':
            inf['replay'] = {str(j): dict(d=1e-4, n_sites=4) for j in range(len(nets))}
        if tamper == 'replay_n':
            inf['replay']['P0'] = dict(d=1e-4, n_sites=3)
        if tamper == 'net_sha':
            inf['nets']['J0_h' + SH.FOLDS[0]] = '0' * 64
        if tamper == 'infer_run':
            inf['run_dir'] = str(tmp / 'other')
        if tamper == 'infer_site_sha':
            inf['site_sha'] = '0' * 64
        (data / 'infer_lf.json').write_text(json.dumps(inf))

    want = {None: [], 'no_limit': ['full_draw'], 'model_sha': ['site_rules'], 'null_rule': ['site_rules'],
            'ink_rule': ['site_rules'], 'sha_untracked': ['G1_site_sha'], 'sha_dirty': ['G1_site_sha'],
            'sha_after_outputs': ['G1_site_sha'], 'points_sha': ['G1_site_sha'], 'infer_site_sha': ['G1_site_sha'],
            'g2_flag': ['G2'], 'orient_str': ['orientation'], 'orient_flips': ['orientation'],
            'replay_tol': ['replay'], 'replay_keys': ['replay'], 'replay_n': ['replay'], 'net_sha': ['nets'],
            'infer_run': ['run_dir'], 'C_DATA_wrong': ['run_dir']}
    keep = C.DATA
    out = {}
    git('init', '-q')
    try:
        for t, w in want.items():
            build(None if t == 'C_DATA_wrong' else t)
            C.DATA = tmp / 'other' if t == 'C_DATA_wrong' else run
            B = Bundle(C, SH, read_outputs=False, data=data, run=run, sha_file=repo / 'sites_lf.sha256')
            _, gates = gates_of(B)
            failed = sorted(k for k, v in gates.items() if not v)
            out[str(t)] = dict(failed=failed, want=sorted(w), ok=failed == sorted(w))
            print(f'bundle case {t}: failed gates {failed} (want {sorted(w)}) -> '
                  f'{"ok" if failed == sorted(w) else "WRONG"}', flush=True)
    finally:
        C.DATA = keep
        shutil.rmtree(tmp, ignore_errors=True)
    return dict(cases=out, ok=all(v['ok'] for v in out.values()))


def g2_synthetic(S, pts, flags):
    """g2_measure on known cases. J leaks like D on 60 % of the sites (depth-blind there) and is hecate on the rest,
    so its advantage over D on all label points is small. A neutral source (keeps every point) must pass; a source
    that keeps only the points where J puts no p >= 0.5 mass on a leak plane (it favours J) must fail."""
    H = [d['p_hecate_fwd'].astype(np.float32) for d in pts]
    blind = [np.repeat(h.max(1, keepdims=True), h.shape[1], 1) for h in H]
    Jm = [b if (i % 5) < 3 else h for i, (h, b) in enumerate(zip(H, blind))]
    outs = {'J0': [np.stack([j, j]) for j in Jm], 'J1': [np.stack([j, j]) for j in Jm],
            'D0': [np.stack([b, b]) for b in blind], 'D1': [np.stack([b, b]) for b in blind]}
    allsrc = [np.ones(len(d['pm'])) for d in pts]
    a = g2_measure(S, pts, flags, outs, allsrc)
    fav = []
    for j, d in zip(Jm, pts):
        lk = S.t4.leak_mask(d['pm'], d['pp'])
        fav.append((((j * (j >= 0.5)) * lk).sum(1) == 0).astype(float))
    b = g2_measure(S, pts, flags, outs, fav)
    out = dict(neutral=dict(a, want=True), favours_J=dict(b, want=False))
    out['ok'] = bool(a['ok'] and not b['ok'])
    print(f'G2 synthetic: neutral dL_src {a["dL_src"]:.4f} dL_lab {a["dL_lab"]:.4f} ok {a["ok"]} (want True); '
          f'favours-J dL_src {b["dL_src"]:.4f} bar {b["bar"]:.4f} ok {b["ok"]} (want False)')
    return out


def dryrun(C, S, SH):
    out = {}
    for case, (want_v, want_w) in DRY_CASES.items():
        r = decide(C, S, SH, Synth(C, S, SH, case))
        ok = (want_v is None or r['verdict'] == want_v) and r['heldout_win']['ok'] == want_w
        out[case] = dict(want_verdict=want_v, got_verdict=r['verdict'], want_win=want_w,
                         got_win=r['heldout_win']['ok'], ok=bool(ok), fold_status=r.get('fold_status'),
                         pooled=None if 'pooled' not in r else {k: r['pooled']['verdict'].get(k) for k in
                                                                ('verdict', 'lower_L_arm', 'dL', 'L_ratio')},
                         heldout_win=r['heldout_win'], patch_rule=r.get('patch_rule', {}).get('ok'))
        print(f'dry run {case}: verdict {r["verdict"]} (want {want_v}), held-out WIN {r["heldout_win"]["ok"]} '
              f'(want {want_w}) -> {"ok" if ok else "WRONG"}', flush=True)
    s0 = Synth(C, S, SH, 'effect')
    g2s = g2_synthetic(S, s0.pts, s0.flags)
    pw = dict(at_bar=power([SRC_INK] * 200 + [SRC_NULL] * 200, [0, 1, 2] * 133 + [0]),
              two_patches=power([SRC_INK] * 200 + [SRC_NULL] * 200, [0, 1] * 200),
              short=power([SRC_INK] * 199 + [SRC_NULL] * 300, [0, 1, 2] * 166 + [0]))
    pw_ok = pw['at_bar']['ok'] and not pw['two_patches']['ok'] and not pw['short']['ok']
    print(f'power rule cases: at bar {pw["at_bar"]["ok"]} (want True), 2 patches {pw["two_patches"]["ok"]} '
          f'(want False), 199 ink {pw["short"]["ok"]} (want False)')
    bc = bundle_cases(C, SH)
    res = dict(pins_commit=PINS_COMMIT, points='PHerc0139 F-395 points.npz ' + P0139_SHA[:8], cases=out, g2=g2s,
               power=pw, bundle=bc, ok=bool(all(c['ok'] for c in out.values()) and g2s['ok'] and pw_ok and bc['ok']))
    (HERE / 'results_dryrun.json').write_text(json.dumps(res, indent=1, default=float))
    print('DRY RUN', 'PASS' if res['ok'] else 'FAIL')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', required=True, help="VES-2321's experiment directory at the pinned commit")
    ap.add_argument('--dryrun', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    C, S, SH = load_rules(a.code)
    if a.dryrun:
        sys.exit(0 if dryrun(C, S, SH)['ok'] else 1)
    B = Bundle(C, SH, read_outputs=not a.check)
    if a.check:
        pw, gates = gates_of(B)
        ctrl = controls_of(S, B)
        ok = all(gates.values()) and ctrl['ok']
        print(json.dumps(dict(power=pw, gates=gates, controls=ctrl, ext=B.ext, ok=ok), indent=1, default=float))
        print('CHECK', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    res = decide(C, S, SH, B)
    res['readings_S08'] = reading_s08(S, B, res)
    res['site_keys'] = B.keys
    (HERE / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res.get(k) for k in ('gates', 'fold_status', 'verdict', 'heldout_win')}, indent=1,
                     default=float))


if __name__ == '__main__':
    main()
