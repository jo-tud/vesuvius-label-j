"""VES-2199 tests 2 and 3 on PHerc0139: second sheet faces near label points at 9.362 um vs 2.399 um, and the
share of label area where sheets are < 2 vx apart.

Criteria: README "Tests 2 and 3" (committed ae8f4663 before any number). The band finder is villa PR #2368
band.sheet_faces at head 87223503, vendored byte-identical as pr2368_band.py. Writes results_t23.json.

  python run_t23.py --check        synthetic gap-ladder detector check only
  python run_t23.py [--workers 8]  full run (reads both scans over https)
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiohttp
import numpy as np
import tifffile
import zarr
from scipy import ndimage

# zarr 2 names the 'store key missing' error PathNotFoundError; zarr 3 has NodeNotFoundError and friends instead
# (VES-2267: under zarr 3.2.1 the bare name raised AttributeError inside the retry handler and killed the run).
# The tuple is concatenated, not nested: a tuple inside an except tuple raises TypeError at match time (VES-2407).
_ZARR_MISSING = tuple(getattr(zarr.errors, n) for n in ('PathNotFoundError', 'NodeNotFoundError', 'ArrayNotFoundError',
                                                        'GroupNotFoundError') if hasattr(zarr.errors, n))
# One flat tuple: an except clause refuses a tuple nested in a tuple with "catching classes that do not inherit from
# BaseException" (VES-2409: the first S3 'Server disconnected' crashed the run instead of retrying).
_RETRY_ERRORS = (OSError, aiohttp.ClientError, asyncio.TimeoutError) + _ZARR_MISSING

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root; data defaults live under ROOT / 'data'
REPO_DATA = Path(os.environ.get('REPO_DATA', ROOT / 'data'))   # input layout root: ves170/{tifxyz,labels}
FROZEN = Path(os.environ.get('FROZEN_TEST_DIR', ROOT / 'data/frozen_test'))   # main() only (source study tests 2-3)
BAND_SHA = 'f4b3636dfa56d4e0b82fb0ea480cfb40ba9a7206eab8d6da094d8ee31269ffa1'

S3 = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes'
VOL9 = f'{S3}/20250728140407-9.362um-1.2m-113keV-masked.zarr'
VOL24 = f'{S3}/20260102150214-2.399um-0.2m-78keV-masked.zarr'
UM9, UM24 = 9.362, 2.399

# F-386 CT refit, 2.399 -> 9.362 um, xyz, level-0 voxels (refit_transform.json of the F-386 study)
A_REFIT = np.array([[-0.2568903058062711, -0.0098212221083555, 0.0028388962294269063, 6678.150391423756],
                    [0.009836046534659217, -0.2570071140336112, -0.0013371654726888667, 6690.784673221844],
                    [0.002805719812015442, -0.0012533976346535765, 0.2570714839189746, 623.4539730107373]])
# catalogue affine (data/catalog/metadata.json, PHerc0139 20260102150214 -> 20250728140407), reading (a)
A_CAT = np.array([[-0.2569036650583705, -0.009715391643719536, 0.002790733227688403, 6677.364709056732],
                  [0.009850885045974334, -0.25680016838938274, -0.0014215065457306453, 6689.120376657908],
                  [0.0029113360690184954, -0.0014844301309079428, 0.257134581099402, 624.1652625002474]])

DEV = ('w035', 'w039', 'w040', 'w041', 'w044')     # data/ves170/{labels,tifxyz}
STEP = 20.0                 # canvas px per grid node (tifxyz scale 0.05)
N_SITES, N_SITES_SURF = 40, 20
BOX, STRIDE = 16, 2         # 33 x 33 px box, every 2nd px
SHIFT_X = 300               # N2 shifted-label null
SEED = 2199
H9, H24, DT = 12.0, 47.0, 0.25
NEAR = 2.0                  # own voxels
DROP_T3 = 0.05
ZERO_MAX = 0.10             # a profile with > 10 % zero samples (outside the masked scan) is dropped


def load_band():
    p = HERE / 'pr2368_band.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == BAND_SHA, 'vendored band.py changed'
    spec = importlib.util.spec_from_file_location('pr2368_band', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


band = load_band()
SCALED = {k: band.FACE_DEFAULTS[k] * UM9 / UM24 for k in ('sigma', 'fill', 'min_band', 'ambiguous')}


# ---------------------------------------------------------------- geometry
def load_grid(d):
    g = np.stack([tifffile.imread(Path(d) / f'{c}.tif').astype(np.float64) for c in 'xyz'])
    v = np.all(np.isfinite(g), 0) & ~np.all(g == -1, 0) & (g[2] >= 0)
    return g, v


def geo_dir(seg_id):
    """9.362 um tifxyz of a VES-1018 segment in data/ves1018/geo, as a dict of x/y/z paths."""
    hits = sorted((REPO_DATA / 'ves1018/geo').glob(f'PHerc0139__segments__{seg_id}-*-9.362um.tifxyz__x.tif'))
    assert len(hits) == 1, (seg_id, hits)
    return str(hits[0])[:-len('x.tif')]


def load_grid_flat(prefix):
    g = np.stack([tifffile.imread(prefix + f'{c}.tif').astype(np.float64) for c in 'xyz'])
    v = np.all(np.isfinite(g), 0) & ~np.all(g == -1, 0) & (g[2] >= 0)
    return g, v


def grid_normals(g, v):
    """Unit normals at nodes from central differences (one-sided at holes); NaN where undefined."""
    gg = np.where(v[None], g, np.nan)
    dr = np.gradient(gg, axis=1)
    dc = np.gradient(gg, axis=2)
    n = np.cross(np.moveaxis(dr, 0, -1), np.moveaxis(dc, 0, -1))
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.where(ln > 0, ln, np.nan)


def pixel_valid(v, shape):
    """Canvas pixels whose 4 surrounding nodes are all valid (pixel p <-> node (p + 0.5) / STEP)."""
    vv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    r = np.floor((np.arange(shape[0]) + 0.5) / STEP).astype(int)
    c = np.floor((np.arange(shape[1]) + 0.5) / STEP).astype(int)
    ok_r, ok_c = r < vv.shape[0], c < vv.shape[1]
    out = np.zeros(shape, bool)
    out[np.ix_(ok_r, ok_c)] = vv[np.ix_(r[ok_r], c[ok_c])]
    return out


def at_pixels(g, nrm, rr, cc):
    u, w = (rr + 0.5) / STEP, (cc + 0.5) / STEP
    P = np.stack([ndimage.map_coordinates(g[k], [u, w], order=1) for k in range(3)], 1)
    Nn = np.stack([ndimage.map_coordinates(np.nan_to_num(nrm[..., k]), [u, w], order=1) for k in range(3)], 1)
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    return P, Nn


def to24(A, P9, N9):
    L, t = A[:, :3], A[:, 3]
    P24 = np.linalg.solve(L, (P9 - t).T).T
    N24 = N9 @ L          # L^T n (normals of the inverse map)
    return P24, N24 / np.linalg.norm(N24, axis=1, keepdims=True)


# ---------------------------------------------------------------- sampling
_VOLS = {}


def vol(url, tries=6):
    """Open (and cache) the level-0 array. A dropped connection while fetching the store metadata surfaces as
    zarr PathNotFoundError (VES-2208), so the open is retried with backoff like the box reads."""
    for k in range(tries):
        if url in _VOLS:
            return _VOLS[url]
        try:
            _VOLS[url] = zarr.open(url, mode='r')['0']
        except _RETRY_ERRORS as e:
            if k == tries - 1:
                raise
            print(f'  open retry {k + 1}/{tries - 1} after {type(e).__name__}: {e}', flush=True)
            time.sleep(2.0 * 2 ** k)
    return _VOLS[url]


def read_box(url, lo, hi, tries=6):
    """Read one box over https; on a dropped connection (S3 'Server disconnected', VES-2200) reopen the
    store and retry with backoff. The data read is the same; only the transport is retried."""
    for k in range(tries):
        try:
            V = vol(url)
            return np.asarray(V[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]).astype(np.float32)
        except _RETRY_ERRORS as e:
            if k == tries - 1:
                raise
            print(f'  read retry {k + 1}/{tries - 1} after {type(e).__name__}: {e}', flush=True)
            _VOLS.pop(url, None)
            time.sleep(2.0 * 2 ** k)


def profiles(url, P, Nn, half):
    t = np.arange(-half, half + 0.5 * DT, DT)
    q = P[:, None, :] + t[None, :, None] * Nn[:, None, :]          # (n, T, xyz)
    lo = np.floor(q.reshape(-1, 3).min(0)).astype(int) - 2
    hi = np.ceil(q.reshape(-1, 3).max(0)).astype(int) + 3
    V = vol(url)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(V.shape[::-1]))
    sub = read_box(url, lo, hi)
    coords = np.stack([q[..., 2] - lo[2], q[..., 1] - lo[1], q[..., 0] - lo[0]]).reshape(3, -1)
    pr = ndimage.map_coordinates(sub, coords, order=1, mode='constant', cval=0.0).reshape(len(P), -1)
    return t, pr


def analyse(t, pr, opts=None):
    keys = ('second', 'dist', 'gap_recto', 'gap_verso', 'n_bands')
    out = {k: np.full(len(pr), np.nan) for k in keys}
    out['split'] = np.zeros(len(pr), bool)
    out['ok'] = np.zeros(len(pr), bool)
    out['zero'] = (pr <= 0).mean(1)
    for i, p in enumerate(pr):
        r = band.sheet_faces(t, p, 1.0, at=0.0, opts=opts)
        for k in keys:
            out[k][i] = r[k]
        out['split'][i], out['ok'][i] = r['split'], r['ok']
    return out


def site_job(job):
    """One site: profiles at 9.362 um (mesh), 2.399 um (refit), 2.399 um (catalogue); bands with defaults and,
    at 2.4 um, with physically scaled options."""
    P9, N9 = job['P'], job['N']
    t9, pr9 = profiles(VOL9, P9, N9, H9)
    P24, N24 = to24(A_REFIT, P9, N9)
    t24, pr24 = profiles(VOL24, P24, N24, H24)
    res = dict(key=job['key'], n=len(P9), b9=analyse(t9, pr9), b24=analyse(t24, pr24),
               b24s=analyse(t24, pr24, SCALED))
    if job.get('cat'):
        Pc, Nc = to24(A_CAT, P9, N9)
        tc, prc = profiles(VOL24, Pc, Nc, H24)
        res['b24c'] = analyse(tc, prc)
    return res


# ---------------------------------------------------------------- site draws
def draw_sites(mask, valid, rng, n_sites):
    cand = np.flatnonzero(mask & valid)
    if not len(cand):
        return []
    picks = rng.choice(cand, size=min(n_sites, len(cand)), replace=False)
    H, W = mask.shape
    out = []
    for f in picks:
        r0, c0 = divmod(int(f), W)
        rr, cc = np.mgrid[max(r0 - BOX, 0):min(r0 + BOX + 1, H):STRIDE, max(c0 - BOX, 0):min(c0 + BOX + 1, W):STRIDE]
        rr, cc = rr.ravel(), cc.ravel()
        k = mask[rr, cc] & valid[rr, cc]
        out.append((r0, c0, rr[k], cc[k]))
    return out


def build_jobs():
    jobs = []
    for wi, w in enumerate(DEV):
        g, v = load_grid(REPO_DATA / 'ves170/tifxyz' / w)
        nrm = grid_normals(g, v)
        lab = np.load(REPO_DATA / 'ves170/labels' / f'{w}.npz')
        pos, neg = lab['pos'], lab['neg']
        valid = pixel_valid(v, pos.shape)
        sh = np.zeros_like(pos)
        sh[:, SHIFT_X:] = pos[:, :-SHIFT_X]
        for si, (sname, m) in enumerate((('label', pos), ('N1_nonlabel', neg), ('N2_shifted', sh))):
            rng = np.random.default_rng([SEED, wi, si])
            for k, (r0, c0, rr, cc) in enumerate(draw_sites(m, valid, rng, N_SITES)):
                if not len(rr):
                    continue
                P, Nn = at_pixels(g, nrm, rr, cc)
                ok = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
                jobs.append(dict(key=(w, sname, k, r0, c0), P=P[ok], N=Nn[ok], cat=True))
    # R-surf: frozen windows of the 8 Bt segments and of w030/w045 (geometry only, no label read)
    bt = json.loads((FROZEN / 'bt_windows.json').read_text())['chosen']
    ft = json.loads((FROZEN / 'frozen_test.json').read_text())['test']
    wins = [(b['short'], b['segment_id'], b['window_rc']) for b in bt] + \
           [(k, v_['segment_id'], v_['window_rc']) for k, v_ in sorted(ft.items())]
    for wi, (w, sid, (r0w, c0w)) in enumerate(wins):
        g, v = load_grid_flat(geo_dir(sid))
        nrm = grid_normals(g, v)
        shape = (v.shape[0] * int(STEP), v.shape[1] * int(STEP))
        valid = pixel_valid(v, shape)
        m = np.zeros(shape, bool)
        m[r0w:r0w + 2048, c0w:c0w + 2048] = True
        rng = np.random.default_rng([SEED, 100 + wi])
        for k, (r0, c0, rr, cc) in enumerate(draw_sites(m, valid, rng, N_SITES_SURF)):
            if not len(rr):
                continue
            P, Nn = at_pixels(g, nrm, rr, cc)
            ok = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
            jobs.append(dict(key=(w, 'R_surf', k, r0, c0), P=P[ok], N=Nn[ok], cat=False))
    return jobs


# ---------------------------------------------------------------- metrics
def site_metrics(r):
    """Per-site counts on points with usable profiles at 9.362 and 2.399 um (refit)."""
    b9, b24, b24s = r['b9'], r['b24'], r['b24s']
    use = (b9['zero'] <= ZERO_MAX) & (b24['zero'] <= ZERO_MAX)
    gap = lambda b: np.fmin(b['gap_recto'], b['gap_verso']) < NEAR     # fmin: NaN only if both NaN
    amb = lambda b: (b['second'] - b['dist']) < band.FACE_DEFAULTS['ambiguous']
    m = dict(n=int(use.sum()), n_all=r['n'],
             s9=int((use & (b9['second'] <= NEAR)).sum()), s24=int((use & (b24['second'] <= NEAR)).sum()),
             s24s=int((use & (b24s['second'] <= NEAR)).sum()),
             g9=int((use & gap(b9)).sum()), g24=int((use & gap(b24)).sum()), g24s=int((use & gap(b24s)).sum()),
             amb9=int((use & amb(b9)).sum()), amb24=int((use & amb(b24)).sum()),
             s24p=int((use & (b24['second'] <= NEAR * UM9 / UM24)).sum()),
             g9na=int((use & gap(b9) & ~amb(b9)).sum()), g9amb=int((use & gap(b9) & amb(b9)).sum()),
             split9=int((use & b9['split']).sum()), split24=int((use & b24['split']).sum()),
             noband9=int((use & (b9['n_bands'] == 0)).sum()), noband24=int((use & (b24['n_bands'] == 0)).sum()),
             nb9=float(np.nanmedian(b9['n_bands'][use])) if use.any() else None,
             nb24=float(np.nanmedian(b24['n_bands'][use])) if use.any() else None)
    if 'b24c' in r:
        bc = r['b24c']
        uc = (b9['zero'] <= ZERO_MAX) & (bc['zero'] <= ZERO_MAX)
        m.update(nc=int(uc.sum()), s24c=int((uc & (bc['second'] <= NEAR)).sum()),
                 s9c=int((uc & (b9['second'] <= NEAR)).sum()))
    return m


def pooled(sites, keys, den='n', nboot=2000, seed=SEED):
    """Pooled shares over sites with a site-bootstrap 95 % interval."""
    D = np.array([s[den] for s in sites], float)
    out = {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sites), size=(nboot, len(sites))) if len(sites) else None
    for k in keys:
        X = np.array([s[k] for s in sites], float)
        sh = float(X.sum() / D.sum()) if D.sum() else None
        ci = None
        if len(sites) > 1:
            b = X[idx].sum(1) / np.maximum(D[idx].sum(1), 1)
            ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        out[k] = dict(share=sh, ci95=ci)
    out['points'] = int(D.sum())
    out['sites'] = len(sites)
    return out


def ratio_ci(sites, a, b, nboot=2000):
    rng = np.random.default_rng(SEED + 1)
    A = np.array([s[a] for s in sites], float)
    B = np.array([s[b] for s in sites], float)
    idx = rng.integers(0, len(sites), size=(nboot, len(sites)))
    r = A[idx].sum(1) / np.maximum(B[idx].sum(1), 1)
    return dict(ratio=float(A.sum() / max(B.sum(), 1)), ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))])


# ---------------------------------------------------------------- detector check
def synthetic_check():
    """Gap ladder: band A on [-8, 0], band B on [g, g + 8], label at t = 0 (A's face), noise 3 % of contrast.
    Expected: g = 0.5 merged by fill (not seen), g = 1.5 seen (second <= 2 and gap < 2), g = 3, 4 not seen."""
    rng = np.random.default_rng(SEED)
    t = np.arange(-H9, H9 + 0.5 * DT, DT)
    rows, ok = [], True
    for g, want in ((0.5, False), (1.5, True), (3.0, False), (4.0, False)):
        for rep in range(20):
            p = 40.0 + 100.0 * (((t >= -8) & (t <= 0)) | ((t >= g) & (t <= g + 8))) + rng.normal(0, 3.0, len(t))
            r = band.sheet_faces(t, p, 1.0, at=0.0)
            gap = np.fmin(r['gap_recto'], r['gap_verso'])
            got_s, got_g = bool(r['second'] <= NEAR), bool(gap < NEAR)
            ok &= (got_s == want) and (got_g == want)
            if rep == 0:
                rows.append(dict(gap=g, want=want, second=r['second'], gap_found=float(gap), n_bands=r['n_bands']))
    return dict(passed=bool(ok), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='results_t23.json', help='output file (readings i-ii rerun: results_t23r.json)')
    ap.add_argument('--limit', type=int, default=0, help='debug: first N jobs only (not a scored run)')
    a = ap.parse_args()
    chk = synthetic_check()
    print('detector check', json.dumps(chk))
    if a.check:
        return
    assert chk['passed'], 'detector check failed: tests 2-3 VOID'
    t0 = time.time()
    jobs = build_jobs()
    if a.limit:
        jobs = jobs[:a.limit]
    print(f'{len(jobs)} sites, {sum(len(j["P"]) for j in jobs)} points, built in {time.time() - t0:.0f} s', flush=True)
    sites = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(site_job, jobs, chunksize=1)):
            m = site_metrics(r)
            m['key'] = list(r['key'])
            sites.append(m)
            if i % 25 == 0:
                print(f'  {i + 1}/{len(jobs)} {r["key"]} n {m["n"]} s9 {m["s9"]} s24 {m["s24"]} g9 {m["g9"]} '
                      f'{time.time() - t0:.0f} s', flush=True)
    by = lambda s: [x for x in sites if x['key'][1] == s]
    K = ['s9', 's24', 's24s', 'g9', 'g24', 'g24s', 'amb9', 'amb24', 'split9', 'split24', 'noband9', 'noband24',
         's24p', 'g9na', 'g9amb']
    res = dict(detector_check=chk, band_sha256=BAND_SHA, scaled_options=SCALED,
               sets={s: pooled(by(s), K) for s in ('label', 'N1_nonlabel', 'N2_shifted', 'R_surf')})
    lab = by('label')
    res['catalogue_reading'] = pooled(lab, ['s9c', 's24c'], den='nc')
    S9, S24 = res['sets']['label']['s9']['share'], res['sets']['label']['s24']['share']
    G9 = res['sets']['label']['g9']['share']
    res['t2'] = dict(S9=S9, S24=S24, ratio=ratio_ci(lab, 's24', 's9'),
                     verdict='DROP' if (S9 is None or S9 == 0 or S24 > 0.5 * S9) else 'PASS')
    res['t3'] = dict(G9=G9, bar=DROP_T3, verdict='DROP' if G9 < DROP_T3 else 'PASS')
    res['per_window'] = {w: {s: pooled([x for x in by(s) if x['key'][0] == w], ['s9', 's24', 'g9'], nboot=200)
                             for s in ('label', 'N1_nonlabel', 'N2_shifted')} for w in sorted(DEV)}
    res['dropped_profiles'] = int(sum(s['n_all'] - s['n'] for s in sites))
    res['seconds'] = time.time() - t0
    res['sites'] = sites
    (HERE / a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ('sites', 'per_window')}, indent=1, default=float))


if __name__ == '__main__':
    main()
