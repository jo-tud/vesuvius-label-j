"""VES-2241 data prep (CPU, no model, no score): render the CT along the grid normal for the training blocks, whose
labels exclude every px near a site centre. VES-2578 step 1 (level 0), changes against F-418's prep.py:
- blocks are rendered once on the 176 planes TWIDE (F-418: 32 planes here, then 44 in attempt3/prep3.py with the
  offset-0 window asserted equal; here the offset-0 window TWIDE[JIT:JIT + 128] == TWIN by construction, common.py);
- sites are the step-0 draws (common.sites0); site tiles are NOT rendered here (1920 x 28 MB of disk):
  attempt4/infer4.py renders each tile when it runs the nets. Only the replay inputs are rendered here: the first
  N_REPLAY sites of each segment (DATA/replay.npz), on which the trained nets store outputs for the net replay gate;
- labels within +-192 px of an old-draw step-0 site are excluded (F-418's rule x 4: its prep.py excluded around its
  old F-389-rule sites only; the spacing-draw sites came later and are only scored by held-out nets);
- blocks 768 x 768, rng [2578, wi]; per block the int64 sums of the offset-0 window over ok px (s1, s2, n), so the
  normalisation of any block set is prep.py's mean / std formula without a float copy of the volume.
Writes DATA/train176.npy (n, 176, 768, 768) u8, DATA/train_{ok,pos,sup}.npy (n, 768, 768) bool, DATA/replay.npz,
DATA/prep.json.

  python prep.py [--workers 10] [--limit-blocks N]
"""
import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

import common as C

N_REPLAY = 4
_G = {}


def seg(w):
    """(grid, normals, node validity) of one segment; one segment cached per process (level-0 grids are ~0.5 GB)."""
    if w not in _G:
        _G.clear()
        g, v, nrm = C.grid0(w)
        _G[w] = (g, nrm, v)
    return _G[w]


def bounded(ex, fn, jobs, k):
    """Results of fn over jobs (completion order) with at most k jobs in flight (a block result is 104 MB)."""
    it, fut = iter(jobs), set()
    for j in it:
        fut.add(ex.submit(fn, j))
        if len(fut) >= k:
            done, fut = wait(fut, return_when=FIRST_COMPLETED)
            yield from (f.result() for f in done)
    for f in fut:
        yield f.result()


def job(j):
    t23 = C.f389().t23
    g, nrm, v = seg(j['w'])
    vol, ok = C.render(t23, g, nrm, v, j['ra'], j['rb'], j['ca'], j['cb'], T=C.TWIDE)
    w0 = vol[C.JIT:C.JIT + C.NPL][:, ok].astype(np.int64)                 # offset-0 window, ok px
    return j['id'], vol, ok, (int(w0.sum()), int((w0 * w0).sum()), int(w0.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit-blocks', type=int, default=None)
    a = ap.parse_args()
    t4 = C.f389()
    t23 = t4.t23
    assert np.array_equal(C.TWIN[C.SCORE_K], t4.T_PL), 'score planes != scorer planes'
    C.DATA.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    site_keys, rep, jobs, blocks = [], [], [], []
    for wi, w in enumerate(t23.DEV):
        g, nrm, v = seg(w)
        pos, neg = C.labels0(w)
        valid = t23.pixel_valid(v, pos.shape)
        S = C.sites0(w)
        site_keys += [s['key'] for s in S]
        keep = np.ones(pos.shape, bool)                                       # = ~excl (RAM: no temporaries)
        for s in S:
            if s['kind'] != 'old':
                continue
            _, _, _, r0, c0 = s['key']
            keep[max(r0 - C.EXCL, 0):r0 + C.EXCL + 1, max(c0 - C.EXCL, 0):c0 + C.EXCL + 1] = False
        for s in S[:N_REPLAY]:
            _, _, _, r0, c0 = s['key']
            P, Nn = t23.at_pixels(g, nrm, s['rr'], s['cc'])
            okp = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)        # as F-395 run.py
            ra, ca = r0 - C.TILE, c0 - C.TILE
            jobs.append(dict(id=('site', len(rep)), w=w, ra=ra, rb=ra + 2 * C.TILE + C.PHASE,
                             ca=ca, cb=ca + 2 * C.TILE + C.PHASE))
            rep.append(dict(key=s['key'], u=len(site_keys) - len(S) + S.index(s), ra=ra, ca=ca, rr=s['rr'][okp],
                            cc=s['cc'][okp]))
        trainable = pos | neg                                                 # = (pos | neg) & valid & ~excl
        n_lab = int(np.count_nonzero(trainable))
        np.logical_and(trainable, keep, out=trainable)
        n_excl = n_lab - int(np.count_nonzero(trainable))                    # labelled px within the exclusion
        np.logical_and(trainable, valid, out=trainable)
        del neg, valid
        H, W = pos.shape
        cand = []
        for r in range(0, H - C.BLOCK + 1, C.BLOCK):
            for c in range(0, W - C.BLOCK + 1, C.BLOCK):
                if trainable[r:r + C.BLOCK, c:c + C.BLOCK].mean() >= C.MIN_TRAIN_FRAC:
                    cand.append((r, c))
        rng = np.random.default_rng([C.SEED_BLOCKS, wi])
        pick = [cand[i] for i in sorted(rng.choice(len(cand), min(C.N_BLOCKS, len(cand)), replace=False))]
        pick = pick[:a.limit_blocks]
        for r, c in pick:
            jobs.append(dict(id=('block', len(blocks)), w=w, ra=r, rb=r + C.BLOCK, ca=c, cb=c + C.BLOCK))
            sl = (slice(r, r + C.BLOCK), slice(c, c + C.BLOCK))
            blocks.append(dict(w=w, r=r, c=c, pos=pos[sl].copy(), sup=trainable[sl].copy(),
                               excl_px=int((~keep[sl]).sum())))
        print(f'{w}: {len(S)} sites, {len(cand)} candidate blocks, {len(pick)} picked, '
              f'trainable px {int(trainable.sum())} (excluded {n_excl}), '
              f'{time.time() - T0:.0f} s', flush=True)
        del pos, trainable, keep
    n2 = 2 * C.TILE + C.PHASE
    rvol = np.zeros((len(rep), C.NPL_W, n2, n2), np.uint8)
    rok = np.zeros((len(rep), n2, n2), bool)
    nb = len(blocks)
    bvol = np.lib.format.open_memmap(C.DATA / 'train176.npy', mode='w+', dtype=np.uint8,
                                     shape=(nb, C.NPL_W, C.BLOCK, C.BLOCK))
    bok = np.zeros((nb, C.BLOCK, C.BLOCK), bool)
    sums = [None] * nb
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (jid, vol, ok, sm) in enumerate(bounded(ex, job, jobs, 2 * a.workers)):
            kind, i = jid
            if kind == 'site':
                rvol[i], rok[i] = vol, ok
            else:
                bvol[i], bok[i], sums[i] = vol, ok, sm
            if n % 20 == 0:
                print(f'  rendered {n + 1}/{len(jobs)} {time.time() - T0:.0f} s', flush=True)
    bvol.flush()
    del bvol
    sup = np.stack([b['sup'] for b in blocks]) & bok
    pos = np.stack([b['pos'] for b in blocks])
    for name, arr in (('ok', bok), ('pos', pos), ('sup', sup)):
        np.save(C.DATA / f'train_{name}.npy', arr)
    norm = dict(zip(('mean', 'std'), norm_of([dict(s1=s[0], s2=s[1], n=s[2]) for s in sums])))
    np.savez(C.DATA / 'replay.npz', vol=rvol, ok=rok, ra=np.array([s['ra'] for s in rep]),
             ca=np.array([s['ca'] for s in rep]), u=np.array([s['u'] for s in rep]),
             **{f'rr_{i}': s['rr'] for i, s in enumerate(rep)}, **{f'cc_{i}': s['cc'] for i, s in enumerate(rep)})
    meta = dict(n_sites=len(site_keys), site_keys=site_keys, replay=[dict(key=s['key'], u=s['u']) for s in rep],
                blocks=[dict(w=b['w'], r=b['r'], c=b['c'], excl_px=b['excl_px'], s1=sums[i][0], s2=sums[i][1],
                             n=sums[i][2]) for i, b in enumerate(blocks)],
                n_blocks={w: sum(b['w'] == w for b in blocks) for w in t23.DEV}, smoke=C.SMOKE,
                sup_px=int(sup.sum()), pos_px=int((pos & sup).sum()), norm=norm, limit_blocks=a.limit_blocks,
                seconds=round(time.time() - T0, 1))
    (C.DATA / 'prep.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ('site_keys', 'blocks', 'replay')}, indent=1))


def norm_of(blocks):
    """prep.py's mean / std (offset-0 window, ok px, population std) of a block subset, from the stored sums."""
    s1, s2, nn = (sum(b[k] for b in blocks) for k in ('s1', 's2', 'n'))
    mean = s1 / nn
    return float(mean), float(np.sqrt(s2 / nn - mean * mean))


if __name__ == '__main__':
    main()
