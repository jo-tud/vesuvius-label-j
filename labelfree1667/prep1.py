"""VES-2344 step 1, label-free prep (VES-2369): Will's level-2 grids, candidate sites, R0, 2.399 um renders,
ink_canonical_2um site source S, site classes, power floor and PC-S. No J, D, P or hecate output is made or read.

Rules: README "Step 0.3" (frozen 6eb3709e) and "Step 1 prep" (implementation, written before any S output).

  python -I prep1.py grid      Will's patches -> level-2 tifxyz grids (STEP 20 canvas px per node), flips applied
  python -I prep1.py cands     candidate centres (fresh, whole tile valid, spacing, CT sheet peak) -> cands_lf.json
  python -I prep1.py r0        R0: w018 L2-copy render vs the published surface volume, 2 labelled boxes -> r0.json
  python -I prep1.py render    level-0 tiles (388 x 388 x 62, R0 convention) of Will's candidates + the 480 old sites
  python -I prep1.py infer     GPU: S = ink_canonical_2um on every tile, pooled 4 x 4 to the canvas
  python -I prep1.py sites     site classes, power floor, PC-S -> DATA/sites_lf.json, DATA/gates_lf.json, prep1.json
  (any step: --limit N for a smoke run; its outputs go to DATA/smoke and are never used)
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root; data defaults live under ROOT / 'data'
REPO = Path(os.environ.get('INKCHECK_REPO', ROOT))   # checkout holding tools/inkcheck/src (ink_canonical_2um loader)
DATA0 = Path(os.environ.get('RUN_LABELFREE', ROOT / 'data/labelfree1667'))
CODE = Path(os.environ.get('CODE_1667', ROOT / 'pherc1667'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'tools/inkcheck/src'))
import ctcheck as CT  # noqa: E402  (step-0.2 rules: to_l2, node_measure, OV, PITCH791, SCALE, flips)
import score_lf as SL  # noqa: E402  (sha pins of VES-2321's code)

C, _S, _SH = SL.load_rules(CODE)                   # asserts the pinned sha256 of every VES-2321 file
T4 = C.f389()
T23 = T4.t23                                       # patched to PHerc1667 level 2 (patch1667.apply)

S3 = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667'
VOL0 = f'{S3}/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr'
SURF018 = (f'{S3}/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/'
           '2.399um-0.22m-78keV-volume-20251217075048.zarr/0')
CKPT = Path(os.environ.get('INK2UM_CKPT', ROOT / 'data/models/ink_canonical_2um/r152_3ddec_v2_l5_epoch13.ckpt'))
CKPT_SHA = '36dd0de8'                              # README prefix; full sha recorded in infer json
FLIP = {0: False, 1: True, 2: True, 3: False, 4: False, 5: True}   # results_ct.json orientation (step 0.2 rule)
SEED = 2344
N_CAND = 1500                                      # README: up to 1500 candidate centres
PRE_FACTOR = 2.5                                   # pre-candidates drawn per kept candidate (peak check after)
SPACE = 33                                         # Chebyshev spacing, canvas px
TILE_H = C.TILE                                    # 48: VES-2321 site tile half size (96 x 96)
CTX = 32                                           # canvas px of context each side of the 33 x 33 box (128 l0 px)
REN = 2 * (T23.BOX + CTX) + 1                      # 97 canvas px -> 388 level-0 px
WIN = (23, 85)                                     # ink_canonical_2um layer window of the 109-layer volume
INK_S, INK_FRAC, NULL_FRAC, NULL_MAX = 0.5, 0.25, 0.02, 0.3
CAP = 400
TORN = 2 * T23.STEP                                # level-0 cell edge bar, 2 x the 20 vx node spacing (VES-2375)
THREADS, WORKERS = 4, 2                            # 8 https connections


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def data(limit):
    d = DATA0 / 'smoke' if limit else DATA0
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- grid
def resample(xyz, valid, k, nr=None, nc=None, rows=512):
    """Bilinear resample of a (H, W, 3) node grid at native index j * k (j < nr, nc; default: up to H - 1, W - 1);
    node valid iff its 4 native neighbours are. Row blocks keep memory flat."""
    H, W = valid.shape
    nr = len(np.arange(0, H - 1, k)) if nr is None else nr
    nc = len(np.arange(0, W - 1, k)) if nc is None else nc
    u, w = np.arange(nr) * k, np.arange(nc) * k
    g = np.zeros((nr, nc, 3), np.float64)
    v = np.zeros((nr, nc), bool)
    b = np.floor(w).astype(int)
    b1 = np.minimum(b + 1, W - 1)
    for i0 in range(0, nr, rows):
        uu = u[i0:i0 + rows]
        lo, hi = int(np.floor(uu[0])), min(int(np.floor(uu[-1])) + 2, H)
        sub, sv = xyz[lo:hi], valid[lo:hi]
        U, Wm = np.meshgrid(uu - lo, w, indexing='ij')
        for c in range(3):
            g[i0:i0 + rows, :, c] = ndimage.map_coordinates(np.where(sv, sub[..., c], 0.0), [U, Wm], order=1)
        a = np.floor(uu).astype(int) - lo
        a1 = np.minimum(a + 1 + lo, H - 1) - lo
        v[i0:i0 + rows] = (sv[a][:, b] & sv[a1][:, b] & sv[a][:, b1] & sv[a1][:, b1])
    return g, v


def grid(limit=None):
    A = np.asarray(json.load(open(CT.TF))['transformation_matrix'], np.float64)
    L, tt = A[:, :3], A[:, 3]
    Ainv = np.linalg.inv(L)
    out = {}
    for p in range(6):
        d = CT.WILL / f'patch_{p}'
        sc = json.loads((d / 'meta.json').read_text())['scale']
        xyz, valid = CT.OV.load_xyz(d)
        k = T23.STEP / ((1.0 / sc[0]) * CT.SCALE)          # native nodes per 20 level-2 vx
        g791, v = resample(xyz, valid, k)
        # attempt 2: level-0-density grid (every k / 4 native nodes) whose every 4th node is the level-2 grid
        nr0, nc0 = 4 * (v.shape[0] - 1) + 1, 4 * (v.shape[1] - 1) + 1
        g0, v0 = resample(xyz, valid, k / 4, nr0, nc0)
        del xyz
        g2 = np.full(g791.shape, -1.0)
        g2[v] = CT.to_l2(g791[v], Ainv, tt)
        del g791
        for i0 in range(0, nr0, 512):                       # 7.91 um -> 2.399 um level 0, in place
            blk = g0[i0:i0 + 512]
            blk[:] = (blk - tt) @ Ainv.T
        g0[~v0] = -1.0
        sub = g0[::4, ::4]
        m = v & v0[::4, ::4]
        assert (v == v0[::4, ::4]).all(), 'level-0 grid validity differs from the level-2 grid'
        dev = float(np.abs((sub[m] - 1.5) / 4 - g2[m]).max())
        assert dev < 1e-3, dev
        if FLIP[p]:                                         # mirror the canvas: the grid normal changes sign
            g2, v = g2[:, ::-1], v[:, ::-1]
            g0, v0 = g0[:, ::-1], v0[:, ::-1]
        (DATA0 / 'grid0').mkdir(parents=True, exist_ok=True)
        np.save(DATA0 / 'grid0' / f'will{p}.npy', np.ascontiguousarray(np.moveaxis(g0, -1, 0)).astype(np.float32))
        np.save(DATA0 / 'grid0' / f'will{p}_valid.npy', np.ascontiguousarray(v0))
        del g0
        td = DATA0 / 'tifxyz' / f'will{p}'
        td.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate('xyz'):
            tifffile.imwrite(td / f'{c}.tif', np.ascontiguousarray(g2[..., i]).astype(np.float32))
        gg, vv = T23.load_grid(td)
        dr = np.linalg.norm(np.diff(gg, axis=1), axis=0)[:, :-1]
        dc = np.linalg.norm(np.diff(gg, axis=2), axis=0)[:-1]
        both = vv[:-1, :-1] & vv[1:, :-1] & vv[:-1, 1:]
        out[f'will{p}'] = dict(native_scale=sc, k=k, nodes=list(vv.shape), valid_nodes=int(vv.sum()),
                               nodes_l0=[nr0, nc0], l0_vs_l2_maxdev=dev,
                               flip=FLIP[p], spacing_r_med=float(np.median(dr[both])),
                               spacing_c_med=float(np.median(dc[both])),
                               sha={c: sha(td / f'{c}.tif') for c in 'xyz'})
        log(p, out[f'will{p}'])
    (DATA0 / 'grids.json').write_text(json.dumps(out, indent=1))
    (HERE / 'grids_lf.json').write_text(json.dumps(out, indent=1))


def load_seg(name):
    """(g, nrm, v) of a Will grid ('will<p>') or a VES-2321 L2 copy ('w013' ...)."""
    d = DATA0 / 'tifxyz' / name if name.startswith('will') else T23.REPO_DATA / 'ves170/tifxyz' / name
    g, v = T23.load_grid(d)
    return g, T23.grid_normals(g, v), v


def at(g, nrm, r, c):
    P, N = T23.at_pixels(g, nrm, np.atleast_1d(np.asarray(r, float)), np.atleast_1d(np.asarray(c, float)))
    return P, N


# ---------------------------------------------------------------- candidates
def cands(limit=None):
    A = np.asarray(json.load(open(CT.TF))['transformation_matrix'], np.float64)
    from scipy.spatial import cKDTree
    lab_pts = []
    for w in CT.SEGS:                                         # ctcheck's fresh-area rule, unchanged
        xyz, valid = CT.OV.load_xyz(CT.MESH0 / w)
        lab = np.load(CT.L2 / 'labels' / f'{w}.npz')
        sup = (lab['pos'] | lab['neg'])
        H, W = valid.shape
        sup = sup.reshape(H, 5, W, 5).any((1, 3))
        lab_pts.append(CT.OV.apply_affine(A, xyz[valid & sup].astype(np.float64)))
        del xyz
    t_lab = cKDTree(np.concatenate(lab_pts))
    res = json.loads((HERE / 'results_ct.json').read_text())
    area = {p: res['per_patch'][f'patch_{p}']['area_fresh_cm2'] for p in range(6)}
    tot = sum(area.values())
    quota = {p: int(round(N_CAND * area[p] / tot)) for p in range(6)}
    if limit:
        quota = {p: min(limit, q) for p, q in quota.items()}
    pre = []
    for p in range(6):
        g, nrm, v = load_seg(f'will{p}')
        Hn, Wn = v.shape
        # pixel_valid needs the 4 nodes around a pixel; a tile [r-48, r+48] x [c-48, c+48] is valid iff every node
        # in [floor((r-48.5)/20), floor((r+48.5)/20)+1] (rows, same for cols) is valid
        vv = np.pad(v.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

        def tile_ok(r, c):
            a, b = int((r - TILE_H + 0.5) // T23.STEP), int((r + TILE_H + 0.5) // T23.STEP) + 1
            e, f = int((c - TILE_H + 0.5) // T23.STEP), int((c + TILE_H + 0.5) // T23.STEP) + 1
            if a < 0 or e < 0 or b >= Hn or f >= Wn:
                return False
            s = vv[b + 1, f + 1] - vv[a, f + 1] - vv[b + 1, e] + vv[a, e]
            return s == (b - a + 1) * (f - e + 1)
        gg = np.moveaxis(g, 0, -1)
        cr = np.cross(gg[1:, :-1] - gg[:-1, :-1], gg[:-1, 1:] - gg[:-1, :-1])
        aw = np.linalg.norm(cr, axis=-1) / T23.STEP ** 2      # area per canvas px at a node cell (level-2 vx^2)
        cell_ok = v[1:, 1:] & v[:-1, 1:] & v[1:, :-1] & v[:-1, :-1]
        aw = np.where(cell_ok, aw, 0.0)
        amax = float(np.percentile(aw[cell_ok], 99.5))
        rng = np.random.default_rng([SEED, p])
        acc, tries, n_want = [], 0, int(math.ceil(PRE_FACTOR * quota[p]))
        ar, ac = np.zeros(n_want, np.int64), np.zeros(n_want, np.int64)
        while len(acc) < n_want and tries < 400 * n_want:
            tries += 1
            r, c = int(rng.integers(0, Hn * T23.STEP)), int(rng.integers(0, Wn * T23.STEP))
            u = rng.random()
            if not tile_ok(r, c):
                continue
            i, j = min(int((r + 0.5) // T23.STEP), Hn - 2), min(int((c + 0.5) // T23.STEP), Wn - 2)
            if u > min(aw[i, j] / amax, 1.0):
                continue
            if acc and np.any((np.abs(ar[:len(acc)] - r) < SPACE) & (np.abs(ac[:len(acc)] - c) < SPACE)):
                continue
            P, N = at(g, nrm, r, c)
            if not (np.isfinite(P).all() and np.isfinite(N).all()):
                continue
            p0 = 4.0 * P[0] + 1.5
            p791 = A[:, :3] @ p0 + A[:, 3]
            dl, _ = t_lab.query(p791)
            if dl <= 2 * CT.PITCH791:
                continue
            ar[len(acc)], ac[len(acc)] = r, c
            acc.append((r, c, P[0], N[0], float(dl)))
        log(f'will{p}: quota {quota[p]}, pre {len(acc)} ({tries} tries)')
        pre += [dict(patch=p, r0=a, c0=b, P=Pp, N=Nn, dl=dl, order=k) for k, (a, b, Pp, Nn, dl) in enumerate(acc)]

    def run(j):
        return CT.node_measure(j['P'], j['N'], 0.0)
    with ThreadPoolExecutor(8) as ex:
        ms = list(ex.map(run, pre))
    out = []
    for p in range(6):
        k = 0
        for j, m in zip(pre, ms):
            if j['patch'] != p or k >= quota[p]:
                continue
            j['d0'], j['off_scan'] = m['d0'], m['off_scan']
            if (not m['off_scan']) and m['d0'] <= CT.NEAR_PEAK:
                out.append(dict(patch=p, k=k, r0=j['r0'], c0=j['c0'], d0=m['d0'], dl791=j['dl'],
                                P2=[float(x) for x in j['P']], N2=[float(x) for x in j['N']]))
                k += 1
    peak = {p: dict(pre=sum(j['patch'] == p for j in pre), kept=sum(o['patch'] == p for o in out), quota=quota[p],
                    pre_on_peak=float(np.mean([(not m['off_scan']) and m['d0'] <= CT.NEAR_PEAK
                                               for j, m in zip(pre, ms) if j['patch'] == p])))
            for p in range(6)}
    log(json.dumps(peak))
    res = dict(limit=limit, n=len(out), per_patch=peak, cands=out)
    (data(limit) / 'cands_lf.json').write_text(json.dumps(res))
    if not limit:
        (HERE / 'cands_lf.json').write_text(json.dumps(res))


# ---------------------------------------------------------------- render (level 0, from a level-2 grid)
_V0 = {}


def vol0():
    if 'v' not in _V0:
        from inkcheck.volume import Volume, ZarrLevel
        v = Volume(VOL0, 0)
        v.array = ZarrLevel(VOL0, 0, workers=WORKERS)
        _V0['v'] = v
    return _V0['v']


def load_l0(name):
    """Attempt 2: the level-0-density grid (3, H, W) in level-0 coordinates, 20 level-0 px per node, and its valid
    mask: Will's from `grid` (mirrored like the level-2 grid), VES-2321's segments = their level-0 mesh."""
    if name.startswith('will'):
        return (np.load(DATA0 / 'grid0' / f'{name}.npy', mmap_mode='r'),
                np.load(DATA0 / 'grid0' / f'{name}_valid.npy'))
    return T23.load_grid(CT.MESH0 / name)


def render_tile(g0, v0, v, r0, c0, T):
    """(388, 388, len(T)) u8 render of canvas rows/cols [r0-48, r0+48] at level 0 (2.399 um px). Attempt 2: level-0
    px R sits at node (R + 0.5) / 20 of the level-0-density grid g0 (the published convention; canvas px r = R // 4,
    and at_pixels' node (r + 0.5) / 20 of the level-2 grid is the same point for R = 4 r + 1.5); positions are
    level-0 coordinates; normals = run_t23.grid_normals of g0 on a sub-grid around the tile (2-node margin), bilinear
    as at_pixels; plane t samples p + t n, trilinear, cval 0. valid (388, 388): its canvas px is pixel_valid on the
    level-2 grid v, its 4 level-0-grid nodes are valid in v0 (VES-2374) and every edge of its level-0 cell is
    <= TORN (VES-2375)."""
    from inkcheck.volume import linear_stencil_chunks
    n0 = 4 * REN
    R = 4 * (r0 - T23.BOX - CTX) + np.arange(n0)
    Cc = 4 * (c0 - T23.BOX - CTX) + np.arange(n0)
    rq, cq = R // 4, Cc // 4                                # canvas px of each level-0 px
    vv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]  # T23.pixel_valid, evaluated at these px only
    ri, ci = np.floor((rq + 0.5) / T23.STEP).astype(int), np.floor((cq + 0.5) / T23.STEP).astype(int)
    okr, okc = (rq >= 0) & (ri < vv.shape[0]), (cq >= 0) & (ci < vv.shape[1])
    ok = (vv[np.clip(ri, 0, vv.shape[0] - 1)][:, np.clip(ci, 0, vv.shape[1] - 1)] & okr[:, None] & okc[None])
    u, w = (R + 0.5) / T23.STEP, (Cc + 0.5) / T23.STEP
    # VES-2374: the level-0-density grid has holes inside level-2-valid cells (Will's resample: 31 of 1499 tiles,
    # VES-2321's meshes: none); bilinear across a hole mixes the -1 fill, so its 4 level-0 nodes must be valid too
    v0a = np.asarray(v0)
    q0 = v0a[:-1, :-1] & v0a[1:, :-1] & v0a[:-1, 1:] & v0a[1:, 1:]
    iu = np.clip(np.floor(u).astype(int), 0, q0.shape[0] - 1)
    iw = np.clip(np.floor(w).astype(int), 0, q0.shape[1] - 1)
    ok &= q0[iu][:, iw]
    a = max(int(np.floor(u[0])) - 2, 0)
    b = min(int(np.ceil(u[-1])) + 3, g0.shape[1])
    e = max(int(np.floor(w[0])) - 2, 0)
    f = min(int(np.ceil(w[-1])) + 3, g0.shape[2])
    sub = np.asarray(g0[:, a:b, e:f], np.float64)
    # VES-2375: Will's traces have torn / stretched cells (edges up to 627 level-0 vx; VES-2321's meshes <= 23):
    # bilinear points inside them leave the sheet, so every edge of a level-0 px's cell must be <= TORN
    du = np.linalg.norm(np.diff(sub, axis=1), axis=0)
    dw = np.linalg.norm(np.diff(sub, axis=2), axis=0)
    ok &= (np.maximum.reduce([du[:, :-1], du[:, 1:], dw[:-1], dw[1:]]) <= TORN)[iu - a][:, iw - e]
    out = np.zeros((n0, n0, len(T)), np.uint8)
    if not ok.any():
        return out, ok
    nrm = T23.grid_normals(sub, np.asarray(v0[a:b, e:f]))
    uu, ww = np.meshgrid(u - a, w - e, indexing='ij')
    uu, ww = uu[ok], ww[ok]
    P = np.stack([ndimage.map_coordinates(sub[k], [uu, ww], order=1) for k in range(3)], 1)
    N = np.stack([ndimage.map_coordinates(np.nan_to_num(nrm[..., k]), [uu, ww], order=1) for k in range(3)], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    fin = np.isfinite(P).all(1) & np.isfinite(N).all(1)
    okf = ok.copy()
    okf[ok] = fin
    P, N = P[fin], N[fin]
    if not len(P):
        return out, okf
    q = P[None] + np.asarray(T, np.float64)[:, None, None] * N[None]                # (nT, n, xyz)
    zyx = q[..., ::-1].reshape(-1, 3)
    V = vol0()
    only = linear_stencil_chunks(zyx, V.array.meta.chunks)
    lo = np.maximum(np.floor(zyx.min(0)).astype(int) - 1, 0)
    hi = np.ceil(zyx.max(0)).astype(int) + 2
    # a tile may hold two sheet pieces across a tear (on-mesh, far apart): bound the read box by memory (VES-2375)
    assert np.prod((hi - lo).astype(float)) <= 2 ** 30, f'read box {hi - lo} > 1 GiB: off-mesh points'
    blk = V.read_box(lo, hi, only=only)
    val = ndimage.map_coordinates(blk, (zyx - lo).T, order=1, mode='constant', cval=0.0, output=np.float32)
    out[okf] = np.clip(np.rint(val), 0, 255).astype(np.uint8).reshape(len(T), -1).T
    return out, okf


def old_sites():
    """VES-2321's 480 F-389-rule sites (C.sites: per segment 40 label then 40 N1), as sites4.draw_all builds them."""
    S = []
    for wi, w in enumerate(T23.DEV):
        g, v = T23.load_grid(T23.REPO_DATA / 'ves170/tifxyz' / w)
        lab = np.load(T23.REPO_DATA / 'ves170/labels' / f'{w}.npz')
        valid = T23.pixel_valid(v, lab['pos'].shape)
        S += C.sites(T23, w, wi, lab['pos'], lab['neg'], valid)
    return S


def r0(limit=None):
    """R0: two w018 label sites (the first two of C.sites), L2-copy path, every plane t = -32 .. 32 step 0.5;
    convention (sign s, centre c): published layer k <-> plane t = s (k - c), k in [23, 85). Best mean r wins."""
    import zarr
    zarr.config.set({'async.concurrency': 8})
    S = [s for s in old_sites() if s['key'][0] == 'w018' and s['key'][1] == 'label'][:2]
    _, _, v = load_seg('w018')
    g0, v0 = load_l0('w018')
    T = np.arange(-32.0, 32.01, 0.5)
    pub = zarr.open(SURF018, mode='r')
    ks = np.arange(*WIN)
    rows = []
    boxes = []
    for s in S:
        r0_, c0_ = s['key'][3], s['key'][4]
        ren, ok = render_tile(g0, v0, v, r0_, c0_, T)
        R0, C0 = 4 * (r0_ - T23.BOX - CTX), 4 * (c0_ - T23.BOX - CTX)
        ref = np.transpose(np.asarray(pub[WIN[0]:WIN[1], R0:R0 + 4 * REN, C0:C0 + 4 * REN]), (1, 2, 0))
        boxes.append(dict(key=s['key'], l0_box=[R0, R0 + 4 * REN, C0, C0 + 4 * REN], ren=ren, ok=ok, ref=ref))
    for sign in (1.0, -1.0):
        for cen in (53.5, 54.0, 54.5):
            idx = [int(round((sign * (k - cen) + 32.0) / 0.5)) for k in ks]
            rs = []
            for b in boxes:
                m = b['ok'] & (b['ref'].max(-1) > 0)
                x = b['ren'][..., idx][m].astype(np.float64).ravel()
                y = b['ref'][m].astype(np.float64).ravel()
                rs.append(dict(r=float(np.corrcoef(x, y)[0, 1]), mad=float(np.median(np.abs(x - y))),
                               n_px=int(m.sum())))
            rows.append(dict(sign=sign, centre=cen, boxes=rs, r_mean=float(np.mean([q['r'] for q in rs])),
                             r_min=float(np.min([q['r'] for q in rs]))))
            log(f'sign {sign:+.0f} centre {cen}: r {[round(q["r"], 4) for q in rs]}')
    best = max(rows, key=lambda q: q['r_mean'])
    res = dict(boxes=[dict(key=b['key'], l0_box=b['l0_box']) for b in boxes], rows=rows,
               chosen=dict(sign=best['sign'], centre=best['centre']), r0=best['r_min'],
               bar=SL.R0_MIN, ok=bool(best['r_min'] >= SL.R0_MIN), attempt=2,
               path='level-0 mesh data/pherc1667_meshes/w018, node (R + 0.5) / 20')
    (HERE / 'r0.json').write_text(json.dumps(res, indent=1))
    log('R0', res['chosen'], res['r0'], 'PASS' if res['ok'] else 'FAIL')


def planes():
    r = json.loads((HERE / 'r0.json').read_text())
    assert r['ok'], 'R0 failed: no renders'
    s, cen = r['chosen']['sign'], r['chosen']['centre']
    return s * (np.arange(*WIN) - cen)


def tile_jobs(limit=None):
    cj = json.loads((data(limit) / 'cands_lf.json').read_text())['cands']
    jobs = [dict(name=f"will{c['patch']}_{c['k']:04d}", seg=f"will{c['patch']}", r0=c['r0'], c0=c['c0']) for c in cj]
    for s in old_sites():
        w, cl, k, r0_, c0_ = s['key']
        jobs.append(dict(name=f'{w}_{cl}_{k:03d}', seg=w, r0=r0_, c0=c0_))
    if limit:
        jobs = jobs[:limit] + [j for j in jobs if not j['seg'].startswith('will')][:limit]
    return jobs


def render(limit=None):
    T = planes()
    out = data(limit) / 'tiles'
    out.mkdir(exist_ok=True)
    jobs = [j for j in tile_jobs(limit) if not (out / f"{j['name']}.npy").exists()]
    log(f'{len(jobs)} tiles to render')
    segs = {}
    t0 = time.time()

    def one(j):
        g0, v0, v = segs[j['seg']]
        ren, ok = render_tile(g0, v0, v, j['r0'], j['c0'], T)
        np.save(out / f"{j['name']}.tmp.npy", ren)
        np.save(out / f"{j['name']}_ok.npy", ok)
        os.replace(out / f"{j['name']}.tmp.npy", out / f"{j['name']}.npy")
        return j['name']
    for s in sorted({j['seg'] for j in jobs}):             # one grid at a time keeps memory flat
        segs.clear()
        segs[s] = (*load_l0(s), load_seg(s)[2])
        js = [j for j in jobs if j['seg'] == s]
        with ThreadPoolExecutor(THREADS) as ex:
            for i, _ in enumerate(ex.map(one, js)):
                if i % 50 == 0:
                    log(f'{s}: {i + 1}/{len(js)} {time.time() - t0:.0f}s {vol0().array.stats()}')
    miss = [j['name'] for j in tile_jobs(limit) if not (out / f"{j['name']}.npy").exists()]
    assert not miss, f'{len(miss)} tiles missing'
    (data(limit) / 'render_done.json').write_text(json.dumps(dict(n=len(tile_jobs(limit)), seconds=time.time() - t0,
                                                                  stats=vol0().array.stats() if jobs else None)))
    log(f'render done {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- infer (GPU)
def infer(limit=None):
    import torch
    from inkcheck import ink2um
    d = data(limit)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    m = ink2um.load(CKPT, device=dev, repo_root=REPO)
    out = d / 'S'
    out.mkdir(exist_ok=True)
    t0 = time.time()
    while not (d / 'tiles').exists():                      # the render step creates tiles/ after cands are final
        assert time.time() - t0 < 3 * 3600, 'no renders within 3 h of the GPU start'
        time.sleep(30)
    jobs = tile_jobs(limit)
    ck = sha(CKPT)
    assert ck.startswith(CKPT_SHA), ck
    t_wait = 0.0
    while True:                                            # tiles may still be rendering: run what exists, wait
        done = (d / 'render_done.json').exists()
        todo = [j for j in jobs if not (out / f"{j['name']}.npy").exists()
                and (d / 'tiles' / f"{j['name']}.npy").exists()]
        if not todo:
            if done:
                break
            assert t_wait < 3 * 3600, 'render did not finish within 3 h of the GPU start'
            time.sleep(30)
            t_wait += 30
            continue
        infer_list(m, d, out, todo, t0, len(jobs))
    secs = time.time() - t0 - t_wait
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else None
    (d / 'infer_S.json').write_text(json.dumps(dict(ckpt=str(CKPT), ckpt_sha256=ck, device=dev, gpu=gpu,
                                                    n_tiles=len(jobs), seconds=secs, wait_s=t_wait,
                                                    gpu_hours=(secs + t_wait) / 3600 if dev == 'cuda' else 0.0)))
    log(f'infer done {secs:.0f}s on {dev} (+{t_wait:.0f}s waiting for renders)')


def infer_list(m, d, out, jobs, t0, n_all):
    from inkcheck import ink2um
    for i, j in enumerate(jobs):
        f = out / f"{j['name']}.npy"
        x = np.load(d / 'tiles' / f"{j['name']}.npy")
        s = ink2um.predict(m, x, batch=4)                  # (388, 388), NaN where uncovered / all-zero window
        np.save(out / f"{j['name']}_full.npy", s.astype(np.float16))
        with np.errstate(all='ignore'):
            pooled = np.nanmean(s.reshape(REN, 4, REN, 4), axis=(1, 3))
        np.save(f, pooled.astype(np.float32))
        if i % 100 == 0:
            log(f'{i + 1}/{len(jobs)} of {n_all} {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- sites
def auc(pos, neg):
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def box_S(Sp, r0, c0, rr, cc):
    """S at canvas px (rr, cc) from a pooled tile centred on (r0, c0)."""
    a, b = rr - (r0 - T23.BOX - CTX), cc - (c0 - T23.BOX - CTX)
    return Sp[a, b]


def sites(limit=None):
    d = data(limit)
    rng = np.random.default_rng(SEED)
    cj = json.loads((d / 'cands_lf.json').read_text())['cands']
    if limit:
        cj = cj[:limit]
    recs, other = [], dict(nan=0, between=0)
    for c in cj:
        name = f"will{c['patch']}_{c['k']:04d}"
        Sp = np.load(d / 'S' / f'{name}.npy')
        rr, cc = np.mgrid[c['r0'] - T23.BOX:c['r0'] + T23.BOX + 1:T23.STRIDE,
                          c['c0'] - T23.BOX:c['c0'] + T23.BOX + 1:T23.STRIDE]
        rr, cc = rr.ravel(), cc.ravel()
        s = box_S(Sp, c['r0'], c['c0'], rr, cc)
        if not np.isfinite(s).all():
            other['nan'] += 1
            continue
        frac = float((s >= INK_S).mean())
        if frac >= INK_FRAC:
            cls, use = SL.SRC_INK, s >= INK_S
        elif frac < NULL_FRAC and s.max() < NULL_MAX:
            cls, use = SL.SRC_NULL, np.ones(len(s), bool)
        else:
            other['between'] += 1
            continue
        recs.append(dict(key=[c['patch'], cls, c['k'], c['r0'], c['c0']], rr=rr.tolist(), cc=cc.tolist(),
                         use=use.tolist(), S=np.round(s, 4).tolist(), frac=frac,
                         nums=dict(n_box_px=int(len(s)), n_s05=int((s >= INK_S).sum()), max_S=float(s.max()))))
    n_class = {k: sum(r['key'][1] == k for r in recs) for k in (SL.SRC_INK, SL.SRC_NULL)}
    keep = []
    for k in (SL.SRC_INK, SL.SRC_NULL):
        idx = [i for i, r in enumerate(recs) if r['key'][1] == k]
        if len(idx) > CAP:
            idx = sorted(rng.choice(idx, CAP, replace=False).tolist())
        keep += idx
    recs = [recs[i] for i in sorted(keep)]
    pw = SL.power([r['key'][1] for r in recs], [r['key'][0] for r in recs])
    # PC-S: S at the old (F-389-rule) label vs N1 points, 6 segments pooled
    pos, neg, per = [], [], {}
    s_old = {}
    for s in old_sites():
        w, cl, k, r0_, c0_ = s['key']
        f = d / 'S' / f'{w}_{cl}_{k:03d}.npy'
        assert f.exists(), f'PC-S: S missing for old site {f.name} (R9 VES-2384 note 2: all 480 must have S)'
        v = box_S(np.load(f), r0_, c0_, s['rr'], s['cc'])
        s_old[f'{w}_{cl}_{k:03d}'] = v
        (pos if cl == 'label' else neg).append(v)
        per.setdefault(w, dict(label=[], N1=[]))['label' if cl == 'label' else 'N1'].append(v)
    assert len(s_old) == 480, f'PC-S on {len(s_old)} old sites, want 480'
    fin = np.isfinite
    P_ = np.concatenate(pos) if pos else np.zeros(0)
    N_ = np.concatenate(neg) if neg else np.zeros(0)
    pcs = auc(P_[fin(P_)], N_[fin(N_)]) if fin(P_).any() and fin(N_).any() else None
    pcs_seg = {}
    for w, v in per.items():
        a_, b_ = (np.concatenate(v[c]) if v[c] else np.zeros(0) for c in ('label', 'N1'))
        pcs_seg[w] = auc(a_[fin(a_)], b_[fin(b_)]) if fin(a_).any() and fin(b_).any() else None
    np.savez_compressed(d / 's_old.npz', **s_old)
    r0j = json.loads((HERE / 'r0.json').read_text())
    ck = sha(CKPT)                                         # score_lf Bundle schema (site_rules, FIX 7)
    assert ck == SL.INK2UM_SHA, ck
    info = dict(limit=limit, site_keys=[r['key'] for r in recs], sites=[r['nums'] for r in recs], model_sha=ck,
                n_candidates=len(cj), rr=[r['rr'] for r in recs],
                cc=[r['cc'] for r in recs], use=[r['use'] for r in recs], S=[r['S'] for r in recs],
                rule=dict(ink=f'>= {INK_FRAC} of box px S >= {INK_S}', null=f'< {NULL_FRAC} S >= {INK_S}, max S < '
                          f'{NULL_MAX}', cap=CAP, box='33 x 33 every 2nd px', nan='box with a NaN S: not used'),
                grids=json.loads((DATA0 / 'grids.json').read_text()), r0_convention=r0j['chosen'],
                code_pins=SL.PINS_COMMIT)
    (d / 'sites_lf.json').write_text(json.dumps(info))
    gates = dict(r0=r0j['r0'], pcs_auc=pcs, g2=dict(ok=False, pending='G2 needs VES-2321 J/D outputs: after its '
                 'verdict'), orientation_applied=True, flips={f'patch_{p}': f for p, f in FLIP.items()})
    (d / 'gates_lf.json').write_text(json.dumps(gates, indent=1))
    res = dict(limit=limit, n_cands=len(cj), n_class_before_cap=n_class, other=other, power=pw,
               PC_S=dict(auc=pcs, bar=SL.PCS_AUC, ok=bool(pcs is not None and pcs >= SL.PCS_AUC), per_segment=pcs_seg,
                         n_label_pts=int(fin(P_).sum()), n_N1_pts=int(fin(N_).sum())),
               R0=dict(r=r0j['r0'], bar=SL.R0_MIN, ok=r0j['ok'], convention=r0j['chosen']),
               sites_lf_sha256=sha(d / 'sites_lf.json'),
               src_ink_share=n_class[SL.SRC_INK] / max(len(cj), 1),
               src_null_share=n_class[SL.SRC_NULL] / max(len(cj), 1))
    (HERE / ('prep1.json' if not limit else 'prep1_smoke.json')).write_text(json.dumps(res, indent=1))
    log(json.dumps({k: res[k] for k in ('n_cands', 'n_class_before_cap', 'other', 'power', 'PC_S', 'R0')},
                   default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('grid', 'cands', 'r0', 'render', 'infer', 'sites'))
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    dict(grid=grid, cands=cands, r0=r0, render=render, infer=infer, sites=sites)[a.cmd](a.limit)
