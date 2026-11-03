"""VES-2344 step 0.2: does Will's fresh PHerc1667 surface lie on papyrus? CT only, no model, no ink.

Criteria: README "Step 0.2" (committed 8d697015 before any CT read of Will's surface). Per node one CT profile along
the grid normal at pyramid level 2 of scan 20251217075048 (t = -24 .. 24 level-2 vx); on a sheet peak = the nearest
band peak (villa #2368 band finder, vendored in VES-2199) within 3 vx of t = 0; null = same rule at a random wrong
depth. Writes results_ct.json and ct_profiles.png.

  python -I ctcheck.py [--limit N]     (N nodes per set, smoke test only)
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import numpy as np
import zarr
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root; data defaults live under ROOT / 'data'
WILL = Path(os.environ.get('WILL1667_DIR', ROOT / 'data/will1667/x'))   # the surface upload, patch_{0..5}
MESH0 = Path(os.environ.get('REPO_DATA', ROOT / 'data')) / 'pherc1667_meshes'   # level-0 2.399 um meshes (VES-2321 input)
L2 = Path(os.environ.get('DATA_1667', ROOT / 'data/pherc1667')) / 'ves170'   # VES-2321 level-2 copies and labels
TF = Path(os.environ.get('TRANSFORM_2399', ROOT / 'data/transform_2399.json'))
VOL = ('https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/'
       '20251217075048-2.399um-0.2m-78keV-masked.zarr')
LEVEL, F = 2, 4
SEGS = ('w013', 'w018', 'w023', 'w028', 'w029', 'w031')
OVERLAP_SHA = '666f80f72ef42389a61d12f3ba99636b020e81f7131d346625229abd2083d623'
PITCH791 = 26.770074366096566                          # F-407 wrap pitch, 7.91 um vx (overlap.json pitch_vx.median)
SCALE = 7.91 / 9.596                                   # 7.91 um vx -> level-2 vx
SEED = 2344
N_WILL, MIN_PATCH, N_REF = 1200, 60, 100
HALF, NEAR_PEAK = 24.0, 3.0
NULL_LO, NULL_HI = 4.0, 12.0
ZERO_MAX = 0.10
THREADS = 8
# README criteria
CT0_SEP, CT1_TOL, CT1_SEP, CT2_FRAC, CT2_MIN_N = 0.15, 0.10, 0.15, 0.5, 50


def load(name, path, sha=None):
    if sha:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, f'{path} changed'
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


if not hasattr(zarr.errors, 'PathNotFoundError'):     # zarr 3: run_t23's retry clause names it (VES-2332)
    zarr.errors.PathNotFoundError = zarr.errors.NodeNotFoundError
OV = load('ves2337_overlap', HERE / 'vendor/overlap_ves2337.py', OVERLAP_SHA)
t23 = load('ves2199_run_t23', ROOT / 'shared/run_t23.py')
band = t23.band


def vol(url, tries=6):
    """run_t23.vol on pyramid level 2 (a retried read reopens level 2, not level 0)."""
    assert url == VOL, url
    for k in range(tries):
        if url in t23._VOLS:
            return t23._VOLS[url]
        try:
            t23._VOLS[url] = zarr.open(url, mode='r')[str(LEVEL)]
        except (OSError, aiohttp.ClientError, asyncio.TimeoutError, zarr.errors.PathNotFoundError) as e:
            if k == tries - 1:
                raise
            print(f'  open retry {k + 1} after {type(e).__name__}: {e}', flush=True)
            time.sleep(2.0 * 2 ** k)
    return t23._VOLS[url]


t23.vol = vol


def to_l2(p791, Ainv, t):
    """7.91 um frame -> 2.399 um level 0 (inverse public affine) -> level 2 (voxel-centre rule)."""
    p0 = (p791 - t) @ Ainv.T
    return (p0 - (F - 1) / 2) / F


def grid_normals_any(g):
    """(R, C, 3) grid with NaN holes -> unit grid normals (rows x cols), NaN where undefined."""
    dr = np.gradient(g, axis=0)
    dc = np.gradient(g, axis=1)
    n = np.cross(dr, dc)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.where(ln > 0, ln, np.nan)


def peaks_of(t, p):
    """Band peaks: local maxima of the band-smoothed profile that lie inside a band of band.bands."""
    o = band.FACE_DEFAULTS
    dt = float(t[1] - t[0])
    ps = band.smooth(np.asarray(p, np.float64), o['sigma'] / dt)
    bl, level, _ = band.bands(t, p)
    i = np.flatnonzero((ps[1:-1] >= ps[:-2]) & (ps[1:-1] > ps[2:]) & (ps[1:-1] >= level)) + 1
    out = []
    for k in i:
        for lo, hi in bl:
            lo = t[0] if not math.isfinite(lo) else lo
            hi = t[-1] if not math.isfinite(hi) else hi
            if lo <= t[k] <= hi:
                out.append(float(t[k]))
                break
    return np.asarray(out)


def node_measure(P, N, u):
    t, pr = t23.profiles(VOL, P[None], N[None], HALF)
    p = pr[0]
    zero = float((p <= 0).mean())
    pk = peaks_of(t, p)
    d0 = float(np.min(np.abs(pk - 0.0))) if len(pk) else math.inf
    du = float(np.min(np.abs(pk - u))) if len(pk) else math.inf
    s0 = float(pk[np.argmin(np.abs(pk))]) if len(pk) else math.nan
    return dict(zero=zero, off_scan=zero > ZERO_MAX, d0=d0, du=du, signed0=s0, n_peaks=int(len(pk)),
                prof=p[::4].astype(np.float32))


def share(rs, key):
    ok = [(not r['off_scan']) and r[key] <= NEAR_PEAK for r in rs]
    return float(np.mean(ok)) if rs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    A = np.asarray(json.load(open(TF))['transformation_matrix'], np.float64)
    L, tt = A[:, :3], A[:, 3]
    Ainv = np.linalg.inv(L)

    # ---- VES-2321 labelled nodes in the 7.91 frame (the fresh-area rule of overlap.py) and our level-2 meshes
    lab_pts, ours_p2, ours_n2 = [], [], []
    for w in SEGS:
        xyz, valid = OV.load_xyz(MESH0 / w)
        lab = np.load(L2 / 'labels' / f'{w}.npz')
        sup = lab['pos'] | lab['neg']
        H, W = valid.shape
        assert sup.shape == (H * 5, W * 5), (w, sup.shape)
        sup = sup.reshape(H, 5, W, 5).any((1, 3))
        keep = valid & sup
        lab_pts.append(OV.apply_affine(A, xyz[keep].astype(np.float64)))
        g2 = np.where(valid[::2, ::2, None], xyz[::2, ::2].astype(np.float64), np.nan)
        g2 = (g2 - (F - 1) / 2) / F                           # level-0 mesh at stride 2 (10 level-2 vx), level 2
        n2 = grid_normals_any(g2)
        ok = np.isfinite(g2).all(-1) & np.isfinite(n2).all(-1)
        ours_p2.append(g2[ok]); ours_n2.append(n2[ok])
        del xyz
    t_lab = cKDTree(np.concatenate(lab_pts))
    ours_p2, ours_n2 = np.concatenate(ours_p2), np.concatenate(ours_n2)
    t_ours = cKDTree(ours_p2)
    print('trees', t_lab.n, t_ours.n, f'{time.time() - t0:.0f}s', flush=True)

    # ---- Will's fresh nodes (16 vx grid of overlap.py), level-2 points and grid normals
    cand = {}
    for p in range(6):
        xyz, valid = OV.load_xyz(WILL / f'patch_{p}')
        g = xyz[::OV.STRIDE, ::OV.STRIDE].astype(np.float64)
        sv = valid[::OV.STRIDE, ::OV.STRIDE]
        area = OV.cell_area(xyz[::OV.STRIDE, ::OV.STRIDE], sv) * OV.VX_UM ** 2 / 1e8
        del xyz, valid
        g2 = np.full(g.shape, np.nan)
        g2[sv] = to_l2(g[sv], Ainv, tt)
        n2 = grid_normals_any(g2)
        ok = sv & np.isfinite(n2).all(-1)
        dl, _ = t_lab.query(g[ok], k=1, workers=OV.WORKERS)
        fresh = dl > 2 * PITCH791
        cand[p] = dict(P=g2[ok][fresh], N=n2[ok][fresh], a=area[ok][fresh], area_fresh=float(area[ok][fresh].sum()))
        print(f'patch_{p}: {ok.sum()} nodes, fresh {fresh.sum()} ({cand[p]["area_fresh"]:.1f} cm2)', flush=True)
    A_tot = sum(c['area_fresh'] for c in cand.values())
    alloc = {p: max(MIN_PATCH, int(round(N_WILL * c['area_fresh'] / A_tot))) for p, c in cand.items()}
    jobs = []
    for p, c in cand.items():
        n = min(alloc[p], len(c['P'])) if a.limit is None else min(a.limit, len(c['P']))
        k = rng.choice(len(c['P']), n, replace=False, p=c['a'] / c['a'].sum())
        for i in k:
            jobs.append(dict(set='will', patch=p, P=c['P'][i], N=c['N'][i]))

    # ---- reference: 100 random labelled px per VES-2321 segment on its level-2 mesh
    for wi, w in enumerate(SEGS):
        g, v = t23.load_grid(L2 / 'tifxyz' / w)
        nrm = t23.grid_normals(g, v)
        lab = np.load(L2 / 'labels' / f'{w}.npz')
        valid = t23.pixel_valid(v, lab['pos'].shape)
        rr, cc = np.nonzero((lab['pos'] | lab['neg']) & valid)
        k = rng.choice(len(rr), N_REF if a.limit is None else a.limit, replace=False)
        P, Nn = t23.at_pixels(g, nrm, rr[k], cc[k])
        for i in range(len(k)):
            if np.isfinite(P[i]).all() and np.isfinite(Nn[i]).all():
                jobs.append(dict(set='ref', patch=w, P=P[i], N=Nn[i]))
    for j in jobs:                                                   # null depth, drawn in job order
        j['u'] = float(rng.choice([-1, 1]) * rng.uniform(NULL_LO, NULL_HI))
    print(f'{len(jobs)} profiles to read', f'{time.time() - t0:.0f}s', flush=True)

    # ---- normal orientation of Will's nodes vs our six meshes (same sheet only)
    half_p2, inpl = PITCH791 * SCALE / 2, 8.0
    W_ = [j for j in jobs if j['set'] == 'will']
    q = np.stack([j['P'] for j in W_])
    d, i = t_ours.query(q, k=1)
    v = q - ours_p2[i]
    dn = np.einsum('ij,ij->i', v, ours_n2[i])
    ip = np.sqrt(np.maximum(d ** 2 - dn ** 2, 0))
    same = (np.abs(dn) <= half_p2) & (ip <= inpl)
    dots = np.einsum('ij,ij->i', np.stack([j['N'] for j in W_]), ours_n2[i])
    orient = {}
    for p in range(6):
        m = same & np.array([j['patch'] == p for j in W_])
        orient[f'patch_{p}'] = dict(n_same_sheet=int(m.sum()), frac_positive=float((dots[m] > 0).mean()) if m.any() else None,
                                    flip=bool(m.any() and (dots[m] > 0).mean() < 0.5))

    # ---- CT profiles (one box read per node)
    def run(j):
        return node_measure(j['P'], j['N'], j['u'])
    with ThreadPoolExecutor(THREADS) as ex:
        rs = list(ex.map(run, jobs))
    for j, r in zip(jobs, rs):
        r.update(set=j['set'], patch=j['patch'])
    print('profiles done', f'{time.time() - t0:.0f}s', flush=True)

    R = [r for r in rs if r['set'] == 'ref']
    Wr = [r for r in rs if r['set'] == 'will']
    S_ref, N_ref, S_will, N_will = share(R, 'd0'), share(R, 'du'), share(Wr, 'd0'), share(Wr, 'du')
    ct0 = bool(S_ref - N_ref >= CT0_SEP)
    ct1 = bool(S_will >= S_ref - CT1_TOL and S_will - N_will >= CT1_SEP)
    per = {}
    for p in range(6):
        rp = [r for r in Wr if r['patch'] == p]
        Sp, Np = share(rp, 'd0'), share(rp, 'du')
        drop = bool(len(rp) >= CT2_MIN_N and Sp < Np + CT2_FRAC * (S_ref - N_ref))
        per[f'patch_{p}'] = dict(n=len(rp), area_fresh_cm2=cand[p]['area_fresh'], S=Sp, N=Np, drop_step1=drop,
                                 off_scan=float(np.mean([r['off_scan'] for r in rp])) if rp else None,
                                 median_signed_peak=float(np.nanmedian([r['signed0'] for r in rp])) if rp else None)
    per_ref = {w: dict(n=len([r for r in R if r['patch'] == w]), S=share([r for r in R if r['patch'] == w], 'd0'),
                       N=share([r for r in R if r['patch'] == w], 'du')) for w in SEGS}
    res = dict(
        criteria_commit='8d697015', limit=a.limit, near_peak_vx=NEAR_PEAK, null_depth=[NULL_LO, NULL_HI],
        fresh_area_cm2=A_tot, n_will=len(Wr), n_ref=len(R),
        S_ref=S_ref, N_ref=N_ref, S_will=S_will, N_will=N_will,
        off_scan_will=float(np.mean([r['off_scan'] for r in Wr])), off_scan_ref=float(np.mean([r['off_scan'] for r in R])),
        median_signed_peak_will=float(np.nanmedian([r['signed0'] for r in Wr])),
        median_signed_peak_ref=float(np.nanmedian([r['signed0'] for r in R])),
        CT0=dict(sep=S_ref - N_ref, bar=CT0_SEP, ok=ct0),
        CT1=dict(gap_to_ref=S_will - S_ref, sep=S_will - N_will, ok=ct1),
        verdict='VOID' if not ct0 else ('PASS' if ct1 else 'FAIL'),
        per_patch=per, per_ref=per_ref, orientation=orient, runtime_s=time.time() - t0)
    out = HERE / ('results_ct.json' if a.limit is None else 'results_ct_smoke.json')
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('S_ref', 'N_ref', 'S_will', 'N_will', 'CT0', 'CT1', 'verdict')}, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tq = np.arange(-HALF, HALF + 0.5 * t23.DT, t23.DT)[::4]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (name, rr_) in enumerate((('reference (VES-2321 labelled)', R), ("Will's fresh area", Wr))):
        pr = np.stack([r['prof'] for r in rr_ if not r['off_scan']])
        ax[k].plot(tq, np.median(pr, 0), 'k', lw=2, label='median')
        ax[k].fill_between(tq, np.percentile(pr, 25, 0), np.percentile(pr, 75, 0), color='0.8', label='IQR')
        ax[k].axvline(0, color='r', lw=1)
        ax[k].set_title(f'{name}: on peak {share(rr_, "d0"):.2f}, null {share(rr_, "du"):.2f}', fontsize=10)
        ax[k].set_xlabel('t along the normal (level-2 vx, 9.6 µm)')
        ax[k].set_ylabel('CT grey value')
    fig.tight_layout()
    fig.savefig(HERE / ('ct_profiles.png' if a.limit is None else 'ct_profiles_smoke.png'), dpi=90)


if __name__ == '__main__':
    main()
