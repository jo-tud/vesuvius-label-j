"""VES-2264 (VES-2241 attempt 4) site prep, CPU, no fold-net output read: per site the F-395 per-point arrays (use,
pitches pm / pp by the attempt-2 rule, dfn) and hecate's 3D forward output at the points (the positive control), and
the power table (site counts, Hanley-McNeil AUC standard error).
VES-2578 step 1 (level 0), changes against F-418's sites4.py:
- sites: the step-0 draws (common.sites0: old 40 + spacing <= 120 per class, box +-64, every 8th px; README
  "Step 1 criteria"), not drawn here; the union order is per segment (run_t23.DEV order), step-0 order within;
- profiles on the level-0 volume along the level-0 normal, t = -96 .. 96 L0 vx, dt 1.0, band options x 4
  (patchl0.apply); profiles with > 10 % zero samples are not used (run_t23.ZERO_MAX, unchanged);
- hecate (9.6 um) runs on the LEVEL-2 render of the same site (F-418's level-2 copy mesh and run_t4.render_tile,
  common.f418_l2): L2 tile centre (r0 // 4, c0 // 4), L0 point (r, c) -> L2 px (r // 4, c // 4) (L0 px 4p + 1.5 is
  the centre of L2 px p); each of hecate's 16 L2 planes t2 = -7.5 .. 7.5 is given to its 4 L0 planes
  t0 = 4 t2 - 1.5 .. 4 t2 + 1.5 (np.repeat), so p_hecate_fwd is (n, 64) on the 64 L0 slab planes;
- tiles are not rendered here (attempt4/infer4.py renders them when it runs the nets).
Writes DATA/points.npz (keys <k>_<u>: rr, cc, use, pm, pp, dfn, p_hecate_fwd) + points.sha256 next to common.py
(committed) and DATA/attempt4/sites4.json (site_keys, kinds, n_points, power, limit).

  python attempt4/sites4.py --count-only        power table only
  python attempt4/sites4.py [--workers 10] [--threads 10]     (VES2578_SMOKE=N: smoke subset, common.sites0)
"""
import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402

OUT = C.DATA / 'attempt4'
MIN_FOLD = 80                # power rule: every fold needs >= 80 label and >= 80 N1 sites (all six folds)
_G = {}


def draw_all():
    """Every step-0 site of every segment, union order (DEV order, step-0 order within a segment)."""
    return [s for w in C.f389().t23.DEV for s in C.sites0(w)]


def hm_se(a, n1, n2):
    """Hanley-McNeil AUC standard error with sites as the unit (n1 label, n2 N1 sites)."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


def power(sites):
    t23 = C.f389().t23
    tab = {}
    for w in t23.DEV:
        n = {c: sum(s['key'][0] == w and s['key'][1] == c for s in sites) for c in ('label', 'N1_nonlabel')}
        n_old = {c: sum(s['key'][0] == w and s['key'][1] == c and s['kind'] == 'old' for s in sites)
                 for c in ('label', 'N1_nonlabel')}
        tab[w] = dict(n_label=n['label'], n_N1=n['N1_nonlabel'], n_label_old=n_old['label'],
                      n_N1_old=n_old['N1_nonlabel'],
                      se_auc={str(a): round(hm_se(a, n['label'], n['N1_nonlabel']), 4) for a in (0.6, 0.65, 0.7)})
    ok = all(tab[w]['n_label'] >= MIN_FOLD and tab[w]['n_N1'] >= MIN_FOLD for w in t23.DEV)
    return dict(folds=tab, min_fold=MIN_FOLD, power_folds=list(t23.DEV), ok=bool(ok))


def seg(w):
    """Level-0 (grid, normals) and F-418's level-2 copy (grid, normals, canvas validity); one segment per process."""
    if w not in _G:
        _G.clear()
        t2 = C.f418_l2().t23
        g0, v0, n0 = C.grid0(w)
        g2, v2 = t2.load_grid(C.L2COPY / 'tifxyz' / w)
        lab2 = np.load(C.L2COPY / 'labels' / f'{w}.npz')
        _G[w] = (g0, n0, g2, t2.grid_normals(g2, v2), t2.pixel_valid(v2, lab2['pos'].shape))
    return _G[w]


def job(j):
    """One site: per-point level-0 profiles / pitches (F-395 run.py lines), hecate's level-2 16-plane tile."""
    t4 = C.f389()
    t23 = t4.t23
    g0, n0, g2, n2, valid2 = seg(j['w'])
    r0, c0 = j['r0'], j['c0']
    rr, cc = j['rr'], j['cc']
    P, Nn = t23.at_pixels(g0, n0, rr, cc)
    okp = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
    rr, cc, P, Nn = rr[okp], cc[okp], P[okp], Nn[okp]
    if len(rr):
        t, pr = t23.profiles(t23.VOL9, P, Nn, t4.H_A2)
        use = (pr <= 0).mean(1) <= t23.ZERO_MAX
        pmpp = np.array([t4.pitches(t, p, 'a2') for p in pr]).reshape(-1, 2)
    else:
        use, pmpp = np.zeros(0, bool), np.zeros((0, 2))
    pm, pp = pmpp[:, 0], pmpp[:, 1]
    dfn = use & ~np.isnan(pm) & ~np.isnan(pp)
    volu8, (ha, hc), okt = C.f418_l2().render_tile(g2, n2, valid2, r0 // C.F, c0 // C.F)
    return dict(id=j['id'], rr=rr, cc=cc, use=use, pm=pm, pp=pp, dfn=dfn, h_vol=volu8, h_ra=ha, h_ca=hc)


def hecate_points(hec, model, r):
    """hecate's 3D forward output at the level-0 points: L2 px (r // 4, c // 4), each L2 plane on its 4 L0 planes."""
    o3 = np.zeros(r['h_vol'].shape, np.uint8)
    hec.predict(model, r['h_vol'], o3, reverse=False, batch_size=4, three_d=True)
    ir, ic = r['rr'] // C.F - r['h_ra'], r['cc'] // C.F - r['h_ca']
    assert (not len(ir)) or (ir.min() >= 0 and ic.min() >= 0 and ir.max() < o3.shape[1] and ic.max() < o3.shape[2]), \
        f"site {r['id']}: L0 points outside hecate's L2 tile (rows {ir.min()}..{ir.max()}, cols {ic.min()}..{ic.max()})"
    p2 = o3[:, ir, ic].T.astype(np.float64) / 255                                                     # (n, 16)
    return np.repeat(p2, C.F, axis=1).astype(np.float16)                                             # (n, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count-only', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--threads', type=int, default=10)
    a = ap.parse_args()
    t4 = C.f389()
    t23 = t4.t23
    assert C.f418_l2().NPL * C.F == t4.NPL == len(C.SCORE_K)
    T0 = time.time()
    sites = draw_all()
    pw = power(sites)
    print(json.dumps(pw, indent=1), flush=True)
    if a.count_only:
        return
    if not pw['ok'] and not C.SMOKE:
        sys.exit('power rule fails: step 1 VOID, no scoring')
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [dict(id=u, w=s['key'][0], r0=s['key'][3], c0=s['key'][4], rr=s['rr'], cc=s['cc'])
            for u, s in enumerate(sites)]
    import torch
    torch.set_num_threads(a.threads)
    t4l2 = C.f418_l2()
    hec = t4l2.load('hecate_ref', t4l2.HEC_DIR / 'hecate.py')
    assert hashlib.sha256(t4l2.CKPT.read_bytes()).hexdigest().startswith('809f4f10'), 'hecate checkpoint changed'
    model = hec.load_model(str(t4l2.CKPT), 'cpu')
    pts = [None] * len(sites)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(job, jobs, chunksize=2)):
            pts[r['id']] = dict(rr=r['rr'], cc=r['cc'], use=r['use'], pm=r['pm'], pp=r['pp'], dfn=r['dfn'],
                                p_hecate_fwd=hecate_points(hec, model, r))
            if n % 50 == 0:
                print(f'  {n + 1}/{len(jobs)} {time.time() - T0:.0f} s', flush=True)
    assert all(p is not None for p in pts)
    np.savez_compressed(C.F395_POINTS, **{f'{k}_{u}': p[k] for u, p in enumerate(pts) for k in p})
    sha = hashlib.sha256(C.F395_POINTS.read_bytes()).hexdigest()
    if not C.SMOKE:
        C.F395_SHA_FILE.write_text(f'{sha}  points.npz\n')
    info = dict(n_sites=len(sites), limit=C.SMOKE or None, site_keys=[s['key'] for s in sites],
                kinds=[s['kind'] for s in sites], n_points=[int(len(p['rr'])) for p in pts], power=pw,
                points_sha256=sha, hecate=dict(ckpt=str(t4l2.CKPT), level=2, planes_l2=t4l2.NPL, repeat=C.F),
                seconds=round(time.time() - T0, 1))
    (OUT / 'sites4.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ('site_keys', 'kinds', 'n_points')}, indent=1,
                     default=str))


if __name__ == '__main__':
    main()
