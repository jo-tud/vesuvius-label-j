"""VES-2578 step 0, G1 + G2 (CPU, no model): do the level-0 1667 meshes sit on the papyrus of the level-0 volume, and
does their grid normal point the same way as the level-2 copy F-418 used? (README "Step-0 checks".)

G2: level-0 grid normals (run_t23.grid_normals) at nodes (4i, 4j) vs the L2 copy's normals at (i, j), cos >= 0.9 at
    >= 95 % of jointly valid nodes.
G1: per segment 60 non-ink + 60 ink px drawn on F-418's level-2 label canvas (rng [2578, wi], non-ink first), placed
    at level-0 canvas px (4r + 2, 4c + 2); CT of `...-masked.zarr/0` along the level-0 normal, t = -96 .. 96 L0 vx,
    step 1 (trilinear). Profiles with > 10 % zero samples are dropped (run_t23.ZERO_MAX). Mean profile per class:
    non-ink argmax |t| <= 16 and contrast mean(|t| <= 4) - mean(64 <= |t| <= 96) >= 15.
    Trap control: the same level-2 px through the L2-copy mesh (L2 voxel xyz and normal) read on level 0 as if they
    were level-0 xyz; it must fail on every segment.
Reading (no bar): side of the non-ink peak; pitch-defined share and median pitches by the attempt-2 rule with the
band options x 4 (README recipe table).

  python step0_frame.py [--n 60] [--workers 8]   -> step0_frame.json
  python step0_frame.py --attempt 2               -> attempt2/step0_frame.json (attempt2/README.md: fresh draw
      rng [2578, wi, 2]; near window |t - argmax| <= 4 instead of |t| <= 4; every other rule unchanged)
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
_ROOT = Path(__file__).resolve().parents[2]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np
import zarr
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pherc1667'))
import prep_data as PD  # noqa: E402
import step0_labels as S  # noqa: E402

F389 = HERE.parents[1] / 'shared'
L2COPY = Path(os.environ.get('DATA_1667', str(_ROOT / 'data/pherc1667'))) / 'ves170' / 'tifxyz'
HALF, DT = 96, 1.0
ZERO_MAX = 0.10
BAND_X4 = ('sigma', 'fill', 'min_band', 'ambiguous')


def load_mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


t4 = load_mod('ves2199_run_t4', F389 / 'run_t4.py')       # pitches() (attempt-2 rule) and run_t23 helpers, unchanged
t23 = t4.t23
_V = {}


def vol0():
    if 'v' not in _V:
        _V['v'] = zarr.open(PD.VOL, mode='r')['0']
    return _V['v']


def profile(P, N):
    """CT along P + t N, t = -HALF .. HALF (level-0 voxels of the level-0 volume); xyz order."""
    t = np.arange(-HALF, HALF + 0.5 * DT, DT)
    q = P[None] + t[:, None] * N[None]
    V = vol0()
    lo = np.maximum(np.floor(q.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(q.max(0)).astype(int) + 3, np.array(V.shape[::-1]))
    if np.any(hi <= lo):
        return t, np.zeros(len(t), np.float32)
    for k in range(6):
        try:
            sub = np.asarray(V[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]).astype(np.float32)
            break
        except Exception as e:                                   # transport only
            if k == 5:
                raise
            print(f'  retry {type(e).__name__}', flush=True)
            _V.clear()
            time.sleep(2.0 * 2 ** k)
            V = vol0()
    c = np.stack([q[:, 2] - lo[2], q[:, 1] - lo[1], q[:, 0] - lo[0]])
    return t, ndimage.map_coordinates(sub, c, order=1, mode='constant', cval=0.0).astype(np.float32)


def class_stats(t, prs, near_at_peak=False):
    keep = [p for p in prs if (p <= 0).mean() <= ZERO_MAX]
    if not keep:
        return dict(n=len(prs), n_used=0, argmax_t=None, contrast=None, pass_=False)
    m = np.mean(keep, 0)
    am = float(t[int(np.argmax(m))])
    near, far = np.abs(t - (am if near_at_peak else 0.0)) <= 4, (np.abs(t) >= 64) & (np.abs(t) <= 96)
    con = float(m[near].mean() - m[far].mean())
    return dict(n=len(prs), n_used=len(keep), argmax_t=am, contrast=round(con, 2), peak=round(float(m.max()), 1),
                pass_=bool(abs(am) <= 16 and con >= 15), mean_profile=np.round(m[::4], 1).tolist())


def pitch_reading(t, prs):
    old = {k: t23.band.FACE_DEFAULTS[k] for k in BAND_X4}
    t23.band.FACE_DEFAULTS.update({k: v * S.F for k, v in old.items()})
    try:
        pp = [t4.pitches(t, p, 'a2') for p in prs if (p <= 0).mean() <= ZERO_MAX]
    finally:
        t23.band.FACE_DEFAULTS.update(old)
    if not pp:
        return dict(n=0)
    a = np.array(pp)
    dfn = ~np.isnan(a).any(1)
    fin = a[dfn & np.isfinite(a).all(1)]
    return dict(n=len(a), defined_share=round(float(dfn.mean()), 3),
                median_pm=float(np.median(fin[:, 0])) if len(fin) else None,
                median_pp=float(np.median(fin[:, 1])) if len(fin) else None)


def point_valid(v, r, c):
    """run_t23.pixel_valid at single canvas px."""
    i, j = np.floor((r + 0.5) / S.STEP).astype(int), np.floor((c + 0.5) / S.STEP).astype(int)
    ok = (i + 1 < v.shape[0]) & (j + 1 < v.shape[1])
    i, j = np.where(ok, i, 0), np.where(ok, j, 0)
    return ok & v[i, j] & v[i + 1, j] & v[i, j + 1] & v[i + 1, j + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--segs', nargs='*', default=list(PD.SEGS))
    ap.add_argument('--attempt', type=int, default=1, choices=(1, 2))
    a = ap.parse_args()
    zarr.config.set({'async.concurrency': 16})
    res = dict(segs={}, attempt=a.attempt)
    out = HERE / ('step0_frame.json' if a.attempt == 1 else 'attempt2/step0_frame.json')
    out.parent.mkdir(exist_ok=True)
    for w in a.segs:
        T0 = time.time()
        wi = list(PD.SEGS).index(w)
        g0, v0 = t23.load_grid(S.MESH0 / w)
        g2, v2 = t23.load_grid(L2COPY / w)
        n0, n2 = t23.grid_normals(g0, v0), t23.grid_normals(g2, v2)
        n0s = n0[::S.F, ::S.F][:n2.shape[0], :n2.shape[1]]
        n2s = n2[:n0s.shape[0], :n0s.shape[1]]
        cos = np.sum(n0s * n2s, -1)
        ok = np.isfinite(cos)
        g2r = dict(n_nodes=int(ok.sum()), share_cos_ge_0p9=round(float((cos[ok] >= 0.9).mean()), 4),
                   median_cos=round(float(np.median(cos[ok])), 4))
        g2r['pass_'] = bool(g2r['share_cos_ge_0p9'] >= 0.95)
        lab = np.load(S.L2_LABELS / f'{w}.npz')
        valid2 = t23.pixel_valid(v2, lab['pos'].shape)
        rng = np.random.default_rng([2578, wi] + ([2] if a.attempt == 2 else []))
        picks = {}
        for cls, m in (('nonink', lab['neg']), ('ink', lab['pos'])):
            cand = np.flatnonzero(m & valid2)
            r2, c2 = np.divmod(rng.choice(cand, size=4 * a.n, replace=False), m.shape[1])
            r0, c0 = S.F * r2 + 2, S.F * c2 + 2
            k = np.flatnonzero(point_valid(v0, r0, c0))[:a.n]               # level-0 nodes valid too
            picks[cls] = (r2[k], c2[k], r0[k], c0[k])
        jobs = []
        for cls, (r2, c2, r0, c0) in picks.items():
            P0, N0 = t23.at_pixels(g0, n0, r0.astype(float), c0.astype(float))
            P2, N2 = t23.at_pixels(g2, n2, r2.astype(float), c2.astype(float))
            for i in range(len(r0)):
                if np.all(np.isfinite(P0[i])) and np.all(np.isfinite(N0[i])):
                    jobs.append(('mesh0', cls, P0[i], N0[i]))
                if np.all(np.isfinite(P2[i])) and np.all(np.isfinite(N2[i])):
                    jobs.append(('l2copy_as_l0', cls, P2[i], N2[i]))
        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(lambda j: profile(j[2], j[3]), jobs))
        t = outs[0][0]
        seg = dict(g2=g2r)
        for src in ('mesh0', 'l2copy_as_l0'):
            seg[src] = {cls: class_stats(t, [o[1] for j, o in zip(jobs, outs) if j[0] == src and j[1] == cls],
                                         near_at_peak=a.attempt == 2)
                        for cls in ('nonink', 'ink')}
        seg['mesh0']['pitch_reading'] = pitch_reading(t, [o[1] for j, o in zip(jobs, outs) if j[0] == 'mesh0'])
        seg['g1_mesh_pass'] = seg['mesh0']['nonink']['pass_']
        seg['g1_control_pass'] = seg['l2copy_as_l0']['nonink']['pass_']
        seg['seconds'] = round(time.time() - T0, 1)
        res['segs'][w] = seg
        print(w, json.dumps({k: (v if k != 'mesh0' and k != 'l2copy_as_l0' else
                                 {c: {kk: vv for kk, vv in x.items() if kk != 'mean_profile'} if isinstance(x, dict)
                                  else x for c, x in v.items()}) for k, v in seg.items()}), flush=True)
        out.write_text(json.dumps(res, indent=1))
    if len(res['segs']) == len(PD.SEGS):
        s = res['segs']
        ctl_any = any(s[w]['g1_control_pass'] for w in PD.SEGS)
        res['G1'] = dict(mesh_pass_all=all(s[w]['g1_mesh_pass'] for w in PD.SEGS), control_pass_any=ctl_any,
                         status='VOID' if ctl_any else ('PASS' if all(s[w]['g1_mesh_pass'] for w in PD.SEGS)
                                                        else 'FAIL'))
        res['G2'] = dict(pass_=all(s[w]['g2']['pass_'] for w in PD.SEGS))
        out.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ('G1', 'G2')}), flush=True)


if __name__ == '__main__':
    main()
