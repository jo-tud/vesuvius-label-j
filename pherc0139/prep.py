"""VES-2241 data prep (CPU, no model, no score): render the 32-plane CT along the 9.362 um grid normal for
(a) the 400 F-389 site tiles (100 x 100 px: both grid phases) and (b) training blocks of the dev five, whose
labels exclude every px within +-48 px of a site centre. Writes DATA/sites.npz, DATA/train.npz, DATA/prep.json.

  python prep.py [--workers 10] [--limit-blocks N]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C

_G = {}


def seg(w):
    if w not in _G:
        t23 = C.f389().t23
        g, v = t23.load_grid(t23.REPO_DATA / 'ves170/tifxyz' / w)
        nrm = t23.grid_normals(g, v)
        lab = np.load(t23.REPO_DATA / 'ves170/labels' / f'{w}.npz')
        _G[w] = (g, nrm, t23.pixel_valid(v, lab['pos'].shape))
    return _G[w]


def job(j):
    t23 = C.f389().t23
    g, nrm, valid = seg(j['w'])
    vol, ok = C.render(t23, g, nrm, valid, j['ra'], j['rb'], j['ca'], j['cb'])
    return j['id'], vol, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit-blocks', type=int, default=None)
    a = ap.parse_args()
    t4 = C.f389()
    t23 = t4.t23
    assert np.array_equal(C.T32[C.SCORE_K], t4.T_PL), 'score planes != F-389 planes'
    C.DATA.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    site_meta, jobs, blocks = [], [], []
    for wi, w in enumerate(t23.DEV):
        g, nrm, valid = seg(w)
        lab = np.load(t23.REPO_DATA / 'ves170/labels' / f'{w}.npz')
        pos, neg = lab['pos'], lab['neg']
        S = C.sites(t23, w, wi, pos, neg, valid)
        excl = np.zeros(pos.shape, bool)
        for s in S:
            _, _, _, r0, c0 = s['key']
            excl[max(r0 - C.EXCL, 0):r0 + C.EXCL + 1, max(c0 - C.EXCL, 0):c0 + C.EXCL + 1] = True
            P, Nn = t23.at_pixels(g, nrm, s['rr'], s['cc'])
            okp = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)        # as F-395 run.py
            ra, ca = r0 - C.TILE, c0 - C.TILE
            jobs.append(dict(id=('site', len(site_meta)), w=w, ra=ra, rb=ra + 2 * C.TILE + C.PHASE,
                             ca=ca, cb=ca + 2 * C.TILE + C.PHASE))
            site_meta.append(dict(key=s['key'], ra=ra, ca=ca, rr=s['rr'][okp], cc=s['cc'][okp]))
        trainable = (pos | neg) & valid & ~excl
        H, W = pos.shape
        cand = []
        for r in range(0, H - C.BLOCK + 1, C.BLOCK):
            for c in range(0, W - C.BLOCK + 1, C.BLOCK):
                if trainable[r:r + C.BLOCK, c:c + C.BLOCK].mean() >= C.MIN_TRAIN_FRAC:
                    cand.append((r, c))
        rng = np.random.default_rng([2241, wi])
        pick = [cand[i] for i in sorted(rng.choice(len(cand), min(C.N_BLOCKS, len(cand)), replace=False))]
        pick = pick[:a.limit_blocks]
        for r, c in pick:
            jobs.append(dict(id=('block', len(blocks)), w=w, ra=r, rb=r + C.BLOCK, ca=c, cb=c + C.BLOCK))
            sl = (slice(r, r + C.BLOCK), slice(c, c + C.BLOCK))
            blocks.append(dict(w=w, r=r, c=c, pos=pos[sl].copy(), sup=trainable[sl].copy(),
                               excl_px=int(excl[sl].sum())))
        print(f'{w}: {len(S)} sites, {len(cand)} candidate blocks, {len(pick)} picked, '
              f'trainable px {int(trainable.sum())} (excluded {int((excl & (pos | neg)).sum())})', flush=True)
    svol = np.zeros((len(site_meta), C.NPL, 2 * C.TILE + C.PHASE, 2 * C.TILE + C.PHASE), np.uint8)
    sok = np.zeros((len(site_meta), 2 * C.TILE + C.PHASE, 2 * C.TILE + C.PHASE), bool)
    bvol = np.zeros((len(blocks), C.NPL, C.BLOCK, C.BLOCK), np.uint8)
    bok = np.zeros((len(blocks), C.BLOCK, C.BLOCK), bool)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (jid, vol, ok) in enumerate(ex.map(job, jobs, chunksize=4)):
            kind, i = jid
            if kind == 'site':
                svol[i], sok[i] = vol, ok
            else:
                bvol[i], bok[i] = vol, ok
            if n % 50 == 0:
                print(f'  rendered {n + 1}/{len(jobs)} {time.time() - T0:.0f} s', flush=True)
    sup = np.stack([b['sup'] for b in blocks]) & bok
    vals = bvol.transpose(1, 0, 2, 3)[:, bok].astype(np.float64)
    norm = dict(mean=float(vals.mean()), std=float(vals.std()))
    np.savez(C.DATA / 'sites.npz', vol=svol, ok=sok, ra=np.array([s['ra'] for s in site_meta]),
             ca=np.array([s['ca'] for s in site_meta]),
             **{f'rr_{i}': s['rr'] for i, s in enumerate(site_meta)},
             **{f'cc_{i}': s['cc'] for i, s in enumerate(site_meta)})
    np.savez(C.DATA / 'train.npz', vol=bvol, ok=bok, pos=np.stack([b['pos'] for b in blocks]), sup=sup)
    meta = dict(n_sites=len(site_meta), site_keys=[s['key'] for s in site_meta],
                n_points=[int(len(s['rr'])) for s in site_meta],
                blocks=[dict(w=b['w'], r=b['r'], c=b['c'], excl_px=b['excl_px']) for b in blocks],
                sup_px=int(sup.sum()), pos_px=int((np.stack([b['pos'] for b in blocks]) & sup).sum()),
                norm=norm, seconds=round(time.time() - T0, 1))
    (C.DATA / 'prep.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ('site_keys', 'n_points', 'blocks')}, indent=1))


if __name__ == '__main__':
    main()
