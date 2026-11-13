"""VES-2578 step 0, G3 + S1 (CPU, no model): level-0 (2.399 um) 1667 ink labels, their agreement with F-418's level-2
labels, and the F-401/F-418 site rules with every in-plane length x 4 (README "Step-0 checks").

Per segment: fetch inklabels.zarr/0 and supervision.zarr/0 (release 20260918) in 512-row bands, pos = ink & sup,
neg = sup & ~ink, cache them as packbits (DATA/labels0/<w>.npz); G3 = Dice of the 4 x 4 pooled level-0 class masks
(mean >= 0.5) against F-418's level-2 pos / neg; S1 = old draw (run_t23.draw_sites rule, 40 per class) + spacing
draw (attempt-4 sites4.draw_new rule, up to 120 per class, Chebyshev spacing 129 px), box +-64 px, every 8th px.
The spacing rule keeps a list of accepted centres instead of an occupancy canvas (same rule, less RAM).

  python step0_labels.py [--segs w028 ...]      -> step0_labels.json (+ DATA/labels0/*.npz, DATA/sites0.npz)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
L2DIR = HERE.parents[1] / 'pherc1667'
sys.path.insert(0, str(L2DIR))
import prep_data as PD  # noqa: E402  (SEGS, S3, VOL_ID; data-access names only)

DATA = Path(os.environ.get('VES2578_DATA', _VD + '/ves2578'))
L2_LABELS = Path(os.environ.get('DATA_1667', str(_ROOT / 'data/pherc1667'))) / 'ves170' / 'labels'
MESH0 = Path(os.environ.get('REPO_DATA', str(_ROOT / 'data'))) / 'pherc1667_meshes'
F = 4                       # level 2 -> level 0
STEP = 20.0                 # canvas px per grid node (level-0 mesh, scale 0.05): canvas px = L0 vx
BOX, STRIDE = 16 * F, 2 * F                 # site box +-64 px, every 8th px (<= 17 x 17 points, as at L2)
N_OLD, N_NEW = 40, 120                      # per class and segment (run_t23.N_SITES, sites4.N_NEW)
SP = 2 * BOX + 1                            # 129: min Chebyshev centre spacing, site boxes never overlap (L2: 33)
SEED_OLD, SEED_NEW = 2578, 2579
MIN_FOLD = 300                              # S1: held-out sites (label + N1) per fold
BAND = 512


def fetch(seg):
    """(pos, neg) level-0 bool arrays of one segment, read in 512-row bands (8 bands in flight)."""
    lab = f'{PD.S3}/segments/{seg}/ink-labels/2.399um-volume-{PD.VOL_ID}/20260918'
    ink = zarr.open(f'{lab}/inklabels.zarr', mode='r')['0']
    sup = zarr.open(f'{lab}/supervision.zarr', mode='r')['0']
    H, W = ink.shape
    pos, neg = np.zeros((H, W), bool), np.zeros((H, W), bool)

    def band(r0):
        for k in range(6):
            try:
                a = np.asarray(ink[r0:r0 + BAND]) > 0
                s = np.asarray(sup[r0:r0 + BAND]) > 0
                break
            except Exception as e:                                   # transport only; same data on retry
                if k == 5:
                    raise
                print(f'  retry band {r0}: {type(e).__name__}', flush=True)
                time.sleep(2.0 * 2 ** k)
        pos[r0:r0 + BAND] = a & s
        neg[r0:r0 + BAND] = s & ~a
        return int((a & ~s).sum())

    with ThreadPoolExecutor(8) as ex:
        ink_out = sum(ex.map(band, range(0, H, BAND)))
    return pos, neg, ink_out


def cached(w, seg):
    p = DATA / 'labels0' / f'{w}.npz'
    if p.exists():
        z = np.load(p)
        sh = tuple(z['shape'])
        unpack = lambda k: np.unpackbits(z[k], count=sh[0] * sh[1]).reshape(sh).astype(bool)  # noqa: E731
        return unpack('pos'), unpack('neg'), int(z['ink_outside_sup'])
    pos, neg, io = fetch(seg)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, pos=np.packbits(pos), neg=np.packbits(neg), shape=np.array(pos.shape),
                        ink_outside_sup=np.array(io))
    return pos, neg, io


def pool4(m, shape2):
    """4 x 4 mean >= 0.5 of a level-0 mask on the level-2 canvas shape (rows / cols past the L0 edge are False)."""
    h, w = min(shape2[0], m.shape[0] // F), min(shape2[1], m.shape[1] // F)
    out = np.zeros(shape2, bool)
    for r0 in range(0, h, 1024):
        r1 = min(r0 + 1024, h)
        blk = m[r0 * F:r1 * F, :w * F].reshape(r1 - r0, F, w, F).sum((1, 3), dtype=np.uint8)
        out[r0:r1, :w] = blk >= (F * F) // 2
    return out


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else float('nan')


def load_grid(d):
    import tifffile
    g = np.stack([tifffile.imread(Path(d) / f'{c}.tif').astype(np.float64) for c in 'xyz'])
    return g, np.all(np.isfinite(g), 0) & ~np.all(g == -1, 0) & (g[2] >= 0)


def pixel_valid(v, shape):
    """run_t23.pixel_valid: canvas px whose 4 surrounding nodes are valid (px p <-> node (p + 0.5) / STEP)."""
    vv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    r = np.floor((np.arange(shape[0]) + 0.5) / STEP).astype(int)
    c = np.floor((np.arange(shape[1]) + 0.5) / STEP).astype(int)
    ok_r, ok_c = r < vv.shape[0], c < vv.shape[1]
    out = np.zeros(shape, bool)
    out[np.ix_(ok_r, ok_c)] = vv[np.ix_(r[ok_r], c[ok_c])]
    return out


def box_points(m, valid, r0, c0):
    H, W = m.shape
    rr, cc = np.mgrid[max(r0 - BOX, 0):min(r0 + BOX + 1, H):STRIDE, max(c0 - BOX, 0):min(c0 + BOX + 1, W):STRIDE]
    rr, cc = rr.ravel(), cc.ravel()
    k = m[rr, cc] & valid[rr, cc]
    return rr[k], cc[k]


def draw_old(m, valid, rng, n):
    """run_t23.draw_sites with BOX / STRIDE x 4."""
    cand = np.flatnonzero(m & valid)
    if not len(cand):
        return []
    W = m.shape[1]
    out = []
    for f in rng.choice(cand, size=min(n, len(cand)), replace=False):
        r0, c0 = divmod(int(f), W)
        out.append((r0, c0) + box_points(m, valid, r0, c0))
    return out


def draw_new(m, valid, cen, rng, n):
    """sites4.draw_new with SP x 4: random order of the class px; a centre is accepted if no accepted or old centre
    lies within SP - 1 px (Chebyshev). cen: list of accepted (r, c), shared by both classes, updated in place."""
    W = m.shape[1]
    out = []
    C = np.array(cen, np.int64).reshape(-1, 2)
    for f in rng.permutation(np.flatnonzero(m & valid)):
        if len(out) == n:
            break
        r0, c0 = divmod(int(f), W)
        if len(C) and (np.maximum(np.abs(C[:, 0] - r0), np.abs(C[:, 1] - c0)) < SP).any():
            continue
        cen.append((r0, c0))
        C = np.array(cen, np.int64)
        out.append((r0, c0) + box_points(m, valid, r0, c0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', nargs='*', default=list(PD.SEGS))
    a = ap.parse_args()
    zarr.config.set({'async.concurrency': 16})
    out_json = HERE / 'step0_labels.json'
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    res.setdefault('segs', {})
    allsites = {}
    for w in a.segs:
        wi = list(PD.SEGS).index(w)
        T0 = time.time()
        pos, neg, io = cached(w, PD.SEGS[w])
        t_fetch = time.time() - T0
        g, v = load_grid(MESH0 / w)
        assert pos.shape == (g.shape[1] * int(STEP), g.shape[2] * int(STEP)), (w, pos.shape, g.shape)
        l2 = np.load(L2_LABELS / f'{w}.npz')
        g3 = dict(dice_pos=dice(pool4(pos, l2['pos'].shape), l2['pos']),
                  dice_neg=dice(pool4(neg, l2['neg'].shape), l2['neg']))
        g3['pass'] = bool(g3['dice_pos'] >= 0.90 and g3['dice_neg'] >= 0.90)
        valid = pixel_valid(v, pos.shape)
        cen, sites = [], []
        for si, (sname, m) in enumerate((('label', pos), ('N1_nonlabel', neg))):
            rng = np.random.default_rng([SEED_OLD, wi, si])
            for k, s in enumerate(draw_old(m, valid, rng, N_OLD)):
                sites.append(dict(cls=sname, kind='old', k=k, r0=s[0], c0=s[1], rr=s[2], cc=s[3]))
                cen.append((s[0], s[1]))
        for si, (sname, m) in enumerate((('label', pos), ('N1_nonlabel', neg))):
            rng = np.random.default_rng([SEED_NEW, wi, si])
            for k, s in enumerate(draw_new(m, valid, cen, rng, N_NEW)):
                sites.append(dict(cls=sname, kind='new', k=1000 + k, r0=s[0], c0=s[1], rr=s[2], cc=s[3]))
        npts = {c: [len(s['rr']) for s in sites if s['cls'] == c] for c in ('label', 'N1_nonlabel')}
        n_ok = {c: sum(x >= 1 for x in npts[c]) for c in npts}
        res['segs'][w] = dict(
            shape0=list(pos.shape), grid0=list(g.shape[1:]), pos_px=int(pos.sum()), neg_px=int(neg.sum()),
            ink_outside_sup_px=io, valid_px=int(valid.sum()), fetch_s=round(t_fetch, 1), g3=g3,
            n_label=n_ok['label'], n_N1=n_ok['N1_nonlabel'], n_heldout=n_ok['label'] + n_ok['N1_nonlabel'],
            n_old={c: sum(s['cls'] == c and s['kind'] == 'old' for s in sites) for c in npts},
            n_new={c: sum(s['cls'] == c and s['kind'] == 'new' for s in sites) for c in npts},
            points_per_site_median={c: float(np.median(x)) if x else None for c, x in npts.items()},
            s1_fold_pass=bool(n_ok['label'] + n_ok['N1_nonlabel'] >= MIN_FOLD))
        allsites[w] = sites
        print(w, json.dumps(res['segs'][w]), f'{time.time() - T0:.0f} s', flush=True)
        del pos, neg, valid
        out_json.write_text(json.dumps(res, indent=1))
        sp = DATA / 'sites0' / f'{w}.npz'
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sp, **{f'{k}_{i}': s[k] for i, s in enumerate(sites) for k in ('rr', 'cc')},
                            meta=np.array(json.dumps([{k: s[k] for k in ('cls', 'kind', 'k', 'r0', 'c0')}
                                                      for s in sites])))
    segs = res['segs']
    if len(segs) == len(PD.SEGS):
        n_pass = sum(segs[w]['s1_fold_pass'] for w in PD.SEGS)
        res['S1'] = dict(rule='>= 300 held-out sites on >= 4 of 6 folds', n_folds_pass=n_pass, pass_=n_pass >= 4)
        res['G3'] = dict(rule='Dice >= 0.90 per class per segment', pass_=all(segs[w]['g3']['pass'] for w in PD.SEGS))
        res['rules'] = dict(BOX=BOX, STRIDE=STRIDE, SP=SP, N_OLD=N_OLD, N_NEW=N_NEW, seeds=[SEED_OLD, SEED_NEW])
        out_json.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ('S1', 'G3')}), flush=True)


if __name__ == '__main__':
    main()
