"""VES-2264 (VES-2241 attempt 4) site prep, CPU, no fold-net output read: draw new label / N1 sites on the dev five by
the committed sampler rule (README "Sites"), render them as attempt 3 did (44 planes T44, 100 x 100 px tile = both
grid phases), and compute the F-395 per-point arrays (use, pitches pm / pp by the attempt-2 rule, dfn) and hecate's
3D forward output at the points (the per-fold positive control). Also: the replay check (the same code path on 2 old
sites per segment must reproduce attempt 3's render byte for byte and F-395's points), the xseg reading for the new
sites, and the power table (site counts, Hanley-McNeil AUC standard error).

  python attempt4/sites4.py --count-only        draw + power table only (no render), prints counts
  python attempt4/sites4.py [--workers 10] [--threads 10] [--limit N]
Writes DATA/attempt4/{sites4.npz, points4.npz, sites4.json, xseg4.json}.

VES-2321 (PHerc1667): there are no 0139 artefacts to replay, so the replay check is replaced by the build of the
F-395-role file: the same job + hecate_points run on EVERY old (F-389-rule) site and write DATA/points_old.npz
(C.F395_POINTS, keys dfn/use/pm/pp/p_hecate_fwd_<i>, F-395's format) and its sha256 to points_old.sha256 next to
common.py (committed). The power rule covers every 1667 fold.
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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
from prep import seg  # noqa: E402

OUT = C.DATA / 'attempt4'
SEED4 = 2264                 # sampler rng [2264, segment index, class index]
N_NEW = 120                  # new sites per class and segment (fewer if the spacing rule runs out of room)
SP = 2 * 16 + 1              # min centre spacing (Chebyshev, px): 33 x 33 site boxes never overlap
MIN_FOLD = 80                # power rule: every fold needs >= 80 label and >= 80 N1 sites (VES-2321: all six folds)


def draw_new(m, valid, occ, rng, n):
    """Sampler rule: random order of the class mask px (mask & valid), a centre is accepted if no accepted or old
    site centre lies within SP - 1 px (Chebyshev); points = mask & valid px of the 33 x 33 box, every 2nd px
    (run_t23.draw_sites' box)."""
    t23 = C.f389().t23
    H, W = m.shape
    out = []
    for f in rng.permutation(np.flatnonzero(m & valid)):
        if len(out) == n:
            break
        r0, c0 = divmod(int(f), W)
        if occ[r0, c0]:
            continue
        occ[max(r0 - SP + 1, 0):r0 + SP, max(c0 - SP + 1, 0):c0 + SP] = True
        rr, cc = np.mgrid[max(r0 - t23.BOX, 0):min(r0 + t23.BOX + 1, H):t23.STRIDE,
                          max(c0 - t23.BOX, 0):min(c0 + t23.BOX + 1, W):t23.STRIDE]
        rr, cc = rr.ravel(), cc.ravel()
        k = m[rr, cc] & valid[rr, cc]
        out.append((r0, c0, rr[k], cc[k]))
    return out


def draw_all():
    """Old F-389 sites (as C.sites) and the new draws, per segment: label first, then N1; one occupancy map per
    segment holds the old sites of both classes and every accepted new site."""
    t23 = C.f389().t23
    old, new = [], []
    for wi, w in enumerate(t23.DEV):
        _, _, valid = seg(w)
        lab = np.load(t23.REPO_DATA / 'ves170/labels' / f'{w}.npz')
        pos, neg = lab['pos'], lab['neg']
        S = C.sites(t23, w, wi, pos, neg, valid)
        occ = np.zeros(pos.shape, bool)
        for s in S:
            r0, c0 = s['key'][3], s['key'][4]
            occ[max(r0 - SP + 1, 0):r0 + SP, max(c0 - SP + 1, 0):c0 + SP] = True
        old += S
        for si, (sname, m) in enumerate((('label', pos), ('N1_nonlabel', neg))):
            rng = np.random.default_rng([SEED4, wi, si])
            for k, (r0, c0, rr, cc) in enumerate(draw_new(m, valid, occ, rng, N_NEW)):
                new.append(dict(key=[w, sname, 1000 + k, r0, c0], rr=rr, cc=cc))
    return old, new


def hm_se(a, n1, n2):
    """Hanley-McNeil AUC standard error with sites as the unit (n1 label, n2 N1 sites)."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


def power(old, new):
    t23 = C.f389().t23
    tab = {}
    for w in t23.DEV:
        n = {c: sum(s['key'][0] == w and s['key'][1] == c for s in old + new) for c in ('label', 'N1_nonlabel')}
        n_old = {c: sum(s['key'][0] == w and s['key'][1] == c for s in old) for c in ('label', 'N1_nonlabel')}
        tab[w] = dict(n_label=n['label'], n_N1=n['N1_nonlabel'], n_label_f400=n_old['label'],
                      n_N1_f400=n_old['N1_nonlabel'],
                      se_auc={str(a): round(hm_se(a, n['label'], n['N1_nonlabel']), 4) for a in (0.6, 0.65, 0.7)},
                      se_auc_f400={str(a): round(hm_se(a, n_old['label'], n_old['N1_nonlabel']), 4)
                                   for a in (0.6, 0.65, 0.7)})
    ok = all(tab[w]['n_label'] >= MIN_FOLD and tab[w]['n_N1'] >= MIN_FOLD for w in t23.DEV)
    return dict(folds=tab, min_fold=MIN_FOLD, power_folds=list(t23.DEV), ok=bool(ok))


def job(j):
    """One site: 44-plane tile render, per-point profiles / pitches (F-395 run.py lines), hecate's 16-plane tile."""
    t4 = C.f389()
    t23 = t4.t23
    g, nrm, valid = seg(j['w'])
    r0, c0 = j['r0'], j['c0']
    ra, ca = r0 - C.TILE, c0 - C.TILE
    n2 = 2 * C.TILE + C.PHASE
    vol, ok = C.render(t23, g, nrm, valid, ra, ra + n2, ca, ca + n2, T=C.T44)
    rr, cc = j['rr'], j['cc']
    P, Nn = t23.at_pixels(g, nrm, rr, cc)
    okp = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
    rr, cc, P, Nn = rr[okp], cc[okp], P[okp], Nn[okp]
    t, pr = t23.profiles(t23.VOL9, P, Nn, t4.H_A2)
    use = (pr <= 0).mean(1) <= t23.ZERO_MAX
    pmpp = np.array([t4.pitches(t, p, 'a2') for p in pr]).reshape(-1, 2)
    pm, pp = pmpp[:, 0], pmpp[:, 1]
    dfn = use & ~np.isnan(pm) & ~np.isnan(pp)
    volu8, (ha, hc), okt = t4.render_tile(g, nrm, valid, r0, c0)
    return dict(id=j['id'], vol=vol, ok=ok, ra=ra, ca=ca, rr=rr, cc=cc, use=use, pm=pm, pp=pp, dfn=dfn,
                h_vol=volu8, h_ra=ha, h_ca=hc)


def hecate_points(hec, model, r):
    o3 = np.zeros(r['h_vol'].shape, np.uint8)
    hec.predict(model, r['h_vol'], o3, reverse=False, batch_size=4, three_d=True)
    return (o3[:, r['rr'] - r['h_ra'], r['cc'] - r['h_ca']].T.astype(np.float64) / 255).astype(np.float16)


def xseg4(new):
    """xseg.py's reading for the new sites: min 3D distance to ANOTHER dev segment's supervised training px."""
    t23 = C.f389().t23
    meta = json.loads((C.DATA / 'prep.json').read_text())
    sup = np.load(C.DATA / 'train.npz')['sup']
    G = {w: seg(w)[:2] for w in t23.DEV}
    TP = {w: [] for w in t23.DEV}
    for b, bl in enumerate(meta['blocks']):
        rr, cc = np.nonzero(sup[b][::4, ::4])
        P, _ = t23.at_pixels(*G[bl['w']], rr * 4 + bl['r'], cc * 4 + bl['c'])
        TP[bl['w']].append(P[np.all(np.isfinite(P), 1)])
    tree = {w: cKDTree(np.concatenate(v)) for w, v in TP.items() if v}
    drop = []
    for s in new:
        w, _, _, r0, c0 = s['key']
        P, _ = t23.at_pixels(*G[w], np.array([r0]), np.array([c0]))
        if np.all(np.isfinite(P)) and min(tree[x].query(P[0])[0] for x in tree if x != w) < 48.0:
            drop.append(s['key'])
    return dict(r_vx=48.0, n_sites=len(new), n_drop=len(drop), drop_keys=drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count-only', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--limit', type=int, default=None, help='smoke: first N new sites per segment only')
    a = ap.parse_args()
    t4 = C.f389()
    t23 = t4.t23
    T0 = time.time()
    old, new = draw_all()
    pw = power(old, new)
    print(json.dumps(pw, indent=1), flush=True)
    if a.count_only:
        return
    if not pw['ok']:
        sys.exit('power rule fails: attempt 4 VOID, no render, no scoring')
    if a.limit:
        new = [s for w in t23.DEV for s in [x for x in new if x['key'][0] == w][:a.limit]]
    OUT.mkdir(parents=True, exist_ok=True)
    rep = list(range(len(old)))                                       # VES-2321: every old site -> points_old.npz
    jobs = [dict(id=('new', i), w=s['key'][0], r0=s['key'][3], c0=s['key'][4], rr=s['rr'], cc=s['cc'])
            for i, s in enumerate(new)]
    jobs += [dict(id=('old', i), w=old[i]['key'][0], r0=old[i]['key'][3], c0=old[i]['key'][4], rr=old[i]['rr'],
                  cc=old[i]['cc']) for i in rep]
    import torch
    torch.set_num_threads(a.threads)
    hec = t4.load('hecate_ref', t4.HEC_DIR / 'hecate.py')
    model = hec.load_model(str(t4.CKPT), 'cpu')
    n2 = 2 * C.TILE + C.PHASE
    vol = np.zeros((len(new), C.NPL_W, n2, n2), np.uint8)
    okm = np.zeros((len(new), n2, n2), bool)
    meta_new, pts, opts = [None] * len(new), [None] * len(new), [None] * len(old)
    st = np.load(C.DATA / 'sites.npz')
    assert json.loads((C.DATA / 'prep.json').read_text())['site_keys'] == [s['key'] for s in old]
    with ProcessPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(job, jobs, chunksize=2)):
            kind, i = r['id']
            ph = hecate_points(hec, model, r)
            if kind == 'new':
                vol[i], okm[i] = r['vol'], r['ok']
                meta_new[i] = dict(key=new[i]['key'], ra=r['ra'], ca=r['ca'], n=int(len(r['rr'])))
                pts[i] = dict(rr=r['rr'], cc=r['cc'], use=r['use'], pm=r['pm'], pp=r['pp'], dfn=r['dfn'],
                              p_hecate_fwd=ph)
            else:                                                     # old site i (VES-2321): F-395-role points
                assert np.array_equal(r['rr'], st[f'rr_{i}']) and np.array_equal(r['ok'], st['ok'][i]), ('old', i)
                opts[i] = dict(dfn=r['dfn'], use=r['use'], pm=r['pm'], pp=r['pp'], p_hecate_fwd=ph)
            if n % 50 == 0:
                print(f'  {n + 1}/{len(jobs)} {time.time() - T0:.0f} s', flush=True)
    assert all(o is not None for o in opts)
    np.savez_compressed(C.F395_POINTS, **{f'{k}_{i}': o[k] for i, o in enumerate(opts) for k in
                                          ('dfn', 'use', 'pm', 'pp', 'p_hecate_fwd')})
    sha = hashlib.sha256(C.F395_POINTS.read_bytes()).hexdigest()
    C.F395_SHA_FILE.write_text(f'{sha}  points_old.npz\n')
    rep_ok = True                                                     # no 0139 replay on 1667 (see docstring)
    replay = dict(none='VES-2321: no 0139 artefacts on 1667', points_old_sha256=sha, n_old=len(old))
    np.savez(OUT / 'sites4.npz', vol=vol, ok=okm, ra=np.array([m['ra'] for m in meta_new]),
             ca=np.array([m['ca'] for m in meta_new]))
    np.savez_compressed(OUT / 'points4.npz', **{f'{k}_{i}': p[k] for i, p in enumerate(pts) for k in p})
    xs = xseg4(new)
    (OUT / 'xseg4.json').write_text(json.dumps(xs, indent=1))
    info = dict(seed=SEED4, n_new_per_class=N_NEW, spacing_px=SP, limit=a.limit, n_new=len(new),
                site_keys=[m['key'] for m in meta_new], n_points=[m['n'] for m in meta_new], power=pw,
                replay=dict(sites=replay, ok=bool(rep_ok)), xseg_n_drop=xs['n_drop'],
                seconds=round(time.time() - T0, 1))
    (OUT / 'sites4.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ('site_keys', 'n_points')}, indent=1, default=str))


if __name__ == '__main__':
    main()
