"""VES-2344 step 1 after VES-2321's WIN (README "Step 1 run"): G2, Will's per-point arrays + hecate, net inference.
Every rule is VES-2321's code, imported unchanged after the sha256 pins of score_lf.PINS: attempt4/sites4.job and
hecate_points (render, per-point arrays, hecate), attempt4/infer4.outputs / net_of (net input, planes, phases,
shift windows), score.f395_points / site_list, score_lf.g2_measure (G2). New here: the Will grid loader handed to
sites4.job, the order of the passes, the file layout score_lf.Bundle reads, and the self-test on old sites.

  python -I step1.py g2         G2 on VES-2321's 480 old sites: held-out J/D outputs of RUN2321, S from s_old.npz
                                -> DATA/gates_lf.json g2, g2_lf.json (no Will output)
  python -I step1.py selftest   old sites only (no Will output): the points path must reproduce VES-2321's stored
                                renders and per-point arrays, and every net its stored outputs -> selftest_step1.json
  python -I step1.py points     Will's sites: 44-plane tiles, per-point arrays, hecate -> DATA/tiles44_lf.npz,
                                DATA/points_lf.npz (site_sha, n_sites)
  python -I step1.py infer      26 nets: replay on old sites first, then outputs on every Will site
                                -> DATA/out_<n>_h<f>_lf.npz, out_<n>_h<f>_lf_shift.npz, out_P<s>_lf_shift.npz, infer_lf.json
Run with RUN_1667=RUN2321 and CODE_1667=VES-2321's experiment directory at the pinned commit.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_lf as SL  # noqa: E402

CODE = Path(os.environ.get('CODE_1667', HERE.parent / 'pherc1667'))
DATA = SL.DATA
C, S, SH = SL.load_rules(CODE)                       # asserts every pinned sha256 (incl. sites4, infer4, prep, train)
assert Path(C.DATA).resolve() == SL.RUN2321.resolve(), f'RUN_1667 must be {SL.RUN2321}, is {C.DATA}'
T4 = C.f389()
T23 = T4.t23
PREP = C.load('ves2321_prep', CODE / 'prep.py')
M4 = C.load('ves2321_sites4', CODE / 'attempt4/sites4.py')
HO = C.DATA / 'heldout'
NAMES = [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
HEC_TOL = 2 / 255                                    # self-test: hecate u8 output, CPU thread-count rounding
N_SELF = 2                                           # self-test: first label and first N1 site of every segment


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


_W = {}


def seg_any(name):
    """(g, nrm, pixel-valid canvas) like prep.seg; Will grids ('will<p>', prep1.py grid, flips applied) on a canvas of
    nodes x STEP px; VES-2321 segments go to prep.seg unchanged."""
    if not name.startswith('will'):
        return PREP.seg(name)
    if name not in _W:
        g, v = T23.load_grid(DATA / 'tifxyz' / name)
        shape = (int(v.shape[0] * T23.STEP), int(v.shape[1] * T23.STEP))
        _W[name] = (g, T23.grid_normals(g, v), T23.pixel_valid(v, shape))
    return _W[name]


M4.seg = seg_any                                     # sites4.job reads its module global `seg`


def job(j):
    r = M4.job(j)
    r['n_in'] = len(j['rr'])
    return r


def pool(workers):
    return ProcessPoolExecutor(workers, mp_context=mp.get_context('fork'))


def old_keys():
    keys = S.site_list()
    assert json.loads((C.DATA / 'prep.json').read_text())['site_keys'] == keys
    return keys


def hecate():
    import torch
    hec = T4.load('hecate_ref', T4.HEC_DIR / 'hecate.py')
    return torch, hec, hec.load_model(str(T4.CKPT), 'cpu')


def site_sha_committed():
    """The G1 sha file must be committed (tracked, clean) and match sites_lf.json before any Will output is made."""
    s = SL.sha(DATA / 'sites_lf.json')
    g = SL.git_g1(HERE / 'sites_lf.sha256', s, [])
    assert g['sha_match'] and g['clean'] and g['commit'], f'G1 not committed: {g}'
    return s


# ---------------------------------------------------------------- G2
def g2(a):
    keys = old_keys()
    pts = S.f395_points()                            # asserts points_old sha (C.F395_POINTS_SHA)
    assert len(pts) == len(keys) == 480
    so = np.load(DATA / 's_old.npz')
    src = [so[f'{w}_{cl}_{k:03d}'] for w, cl, k, _, _ in keys]
    bad = [i for i, (s, d) in enumerate(zip(src, pts)) if len(s) != len(d['pm'])]
    assert not bad, f'S and points differ in length at old sites {bad[:5]}'
    flags = [k[1] == 'label' for k in keys]
    outs = {}
    for n in NAMES:                                  # site i scored by the fold net that held its segment out
        zs = {f: np.load(HO / f'out_{n}_h{f}.npz') for f in SH.FOLDS}
        outs[n] = [zs[k[0]][f'p_{i}'] for i, k in enumerate(keys)]
    r = SL.g2_measure(S, pts, flags, outs, src)
    nl = int(sum(d['dfn'].sum() for d, f in zip(pts, flags) if f))
    ns = int(sum((d['dfn'] & (s >= 0.5)).sum() for d, s, f in zip(pts, src, flags) if f))
    r.update(n_label_sites=int(sum(flags)), n_label_pts_defined=nl, n_src_pts_defined=ns, run=str(C.DATA))
    gj = json.loads((DATA / 'gates_lf.json').read_text())
    gj['g2'] = r
    (DATA / 'gates_lf.json').write_text(json.dumps(gj, indent=1, default=float))
    (HERE / 'g2_lf.json').write_text(json.dumps(r, indent=1, default=float))
    log(f"G2: dL_src {r['dL_src']} dL_lab {r['dL_lab']} bar {r['bar']} ok {r['ok']} "
        f"(label pts {nl}, of them S >= 0.5: {ns})")


# ---------------------------------------------------------------- self-test (old sites only)
def selftest(a):
    keys = old_keys()
    st, w44 = np.load(C.DATA / 'sites.npz'), np.load(C.DATA / 'sites44.npz')['vol']
    zo = np.load(C.F395_POINTS)
    idx = [next(i for i, k in enumerate(keys) if k[0] == w and k[1] == c) for w in T23.DEV
           for c in ('label', 'N1_nonlabel')]
    assert len(idx) == N_SELF * len(T23.DEV)
    jobs = [dict(id=i, w=keys[i][0], r0=keys[i][3], c0=keys[i][4], rr=st[f'rr_{i}'], cc=st[f'cc_{i}']) for i in idx]
    torch, hec, model = hecate()
    torch.set_num_threads(a.threads)
    res = {}
    with pool(a.workers) as ex:
        for r in ex.map(job, jobs):
            i = r['id']
            ph = M4.hecate_points(hec, model, r)
            eq = dict(vol=bool(np.array_equal(r['vol'], w44[i])), ok_mask=bool(np.array_equal(r['ok'], st['ok'][i])),
                      ra=int(r['ra']) == int(st['ra'][i]), ca=int(r['ca']) == int(st['ca'][i]),
                      rr=bool(np.array_equal(r['rr'], st[f'rr_{i}'])), cc=bool(np.array_equal(r['cc'], st[f'cc_{i}'])))
            eq.update({k: bool(np.array_equal(r[k], zo[f'{k}_{i}'], equal_nan=True)) for k in ('dfn', 'use', 'pm', 'pp')})
            hd = float(np.abs(ph.astype(np.float32) - zo[f'p_hecate_fwd_{i}'].astype(np.float32)).max())
            eq['hecate'] = hd <= HEC_TOL
            res[str(i)] = dict(key=keys[i], hecate_max_diff=hd, **eq, ok=all(eq.values()))
            log(f'old site {i} {keys[i][:2]}: ' + ('ok' if res[str(i)]['ok'] else f'FAIL {eq}') + f', hecate {hd:.4f}')
    nets = replay_all(a)
    out = dict(points=res, points_ok=all(v['ok'] for v in res.values()), nets=nets['replay'], device=nets['device'],
               nets_ok=nets['ok'], ok=all(v['ok'] for v in res.values()) and nets['ok'])
    (HERE / 'selftest_step1.json').write_text(json.dumps(out, indent=1, default=float))
    log('SELF-TEST', 'PASS' if out['ok'] else 'FAIL')
    sys.exit(0 if out['ok'] else 1)


# ---------------------------------------------------------------- Will points
def torn_reading(g, v, ra, ca, n):
    """Max level-2 grid edge (vx) between valid nodes under the net tile [ra, ra + n) x [ca, ca + n) (reading only;
    VES-2375's torn cells; the 33 x 33 boxes hold none, R9 VES-2384)."""
    a, b = max(int((ra + 0.5) // T23.STEP), 0), int((ra + n - 0.5) // T23.STEP) + 2
    c, d = max(int((ca + 0.5) // T23.STEP), 0), int((ca + n - 0.5) // T23.STEP) + 2
    gg, vv = g[:, a:b, c:d], v[a:b, c:d]
    er = np.linalg.norm(np.diff(gg, axis=1), axis=0)[(vv[1:] & vv[:-1])]
    ec = np.linalg.norm(np.diff(gg, axis=2), axis=0)[(vv[:, 1:] & vv[:, :-1])]
    e = np.concatenate([er, ec])
    return float(e.max()) if len(e) else None


def points(a):
    s_sha = site_sha_committed()
    info = json.loads((DATA / 'sites_lf.json').read_text())
    assert info['limit'] is None
    keys = info['site_keys']
    jobs = []
    for j, (k, rr, cc, u) in enumerate(zip(keys, info['rr'], info['cc'], info['use'])):
        u = np.asarray(u, bool)                      # README: a site's points = its stored use px (not S >= 0.5)
        jobs.append(dict(id=j, w=f'will{k[0]}', r0=k[3], c0=k[4], rr=np.asarray(rr)[u], cc=np.asarray(cc)[u]))
    if a.limit:
        sys.exit('points: no --limit (a partial points_lf.npz would be a Will output; use selftest)')
    torch, hec, model = hecate()
    torch.set_num_threads(a.threads)
    grids = {p: T23.load_grid(DATA / 'tifxyz' / f'will{p}') for p in sorted({k[0] for k in keys})}
    n2 = 2 * C.TILE + C.PHASE
    vol = np.zeros((len(keys), C.NPL_W, n2, n2), np.uint8)
    okm = np.zeros((len(keys), n2, n2), bool)
    ra, ca, pts, rc, lost, torn = np.zeros(len(keys), int), np.zeros(len(keys), int), {}, {}, {}, {}
    T0 = time.time()
    with pool(a.workers) as ex:
        for n, r in enumerate(ex.map(job, jobs, chunksize=2)):
            j = r['id']
            ph = M4.hecate_points(hec, model, r)
            vol[j], okm[j], ra[j], ca[j] = r['vol'], r['ok'], r['ra'], r['ca']
            pts.update({f'{k}_{j}': r[k] for k in ('dfn', 'use', 'pm', 'pp')})
            pts[f'p_hecate_fwd_{j}'] = ph
            rc.update({f'rr_{j}': r['rr'], f'cc_{j}': r['cc']})
            if len(r['rr']) != r['n_in']:
                lost[j] = r['n_in'] - len(r['rr'])
            torn[j] = torn_reading(*grids[keys[j][0]], r['ra'], r['ca'], n2)
            if n % 50 == 0:
                log(f'{n + 1}/{len(jobs)} {time.time() - T0:.0f} s')
    np.savez(DATA / 'tiles44_lf.npz', vol=vol, ok=okm, ra=ra, ca=ca, **rc)
    np.savez_compressed(DATA / 'points_lf.npz', site_sha=np.array(s_sha), n_sites=np.array(len(keys)), **pts)
    tv = [t for t in torn.values() if t is not None]
    rep = dict(n_sites=len(keys), site_sha=s_sha, n_sites_lost_points=len(lost), lost=lost,
               torn_tile=dict(bar_vx=2 * T23.STEP, n_over=int(sum(t > 2 * T23.STEP for t in tv)),
                              max=max(tv) if tv else None),
               n_nan_hecate_sites=int(sum(not np.isfinite(pts[f'p_hecate_fwd_{j}'].astype(np.float32)).all()
                                          for j in range(len(keys)))),
               seconds=round(time.time() - T0, 1))
    (HERE / 'points_lf.json').write_text(json.dumps(rep, indent=1))
    log(json.dumps({k: v for k, v in rep.items() if k != 'lost'}))


# ---------------------------------------------------------------- nets
class OnDev:
    """infer4.outputs calls net(x) on a CPU tensor; this runs the net on `dev` and returns CPU output."""

    def __init__(self, net, dev):
        self.net, self.dev = net.to(dev), dev

    def __call__(self, x):
        return self.net(x.to(self.dev)).cpu()


def nets_of(I4, dev):
    out = {}
    for f in SH.FOLDS:
        for n in NAMES:
            p = HO / f'net_{n}_h{f}.pt'
            tr = json.loads((HO / f'train_{n}_h{f}.json').read_text())
            out[f'{n}_h{f}'] = dict(f=f, path=p, mu=tr['norm']['mean'], sd=tr['norm']['std'])
    meta = json.loads((C.DATA / 'prep.json').read_text())
    for n in S.FRAME_PC:                                # P0, P1: the known-artefact control
        out[n] = dict(f=None, path=C.DATA / f'net_{n}.pt', mu=meta['norm']['mean'], sd=meta['norm']['std'])
    for v in out.values():
        v['sha'] = SL.sha(v['path'])
        v['net'] = OnDev(I4.net_of(v['path']), dev)
    return out


def replay_all(a, I4=None, nets=None):
    """infer4's replay rule on every net: fold nets on their fold's first N_REPLAY old sites (site + shift outputs),
    P0 / P1 on the first N_REPLAY old sites (shift outputs); max |diff| <= REPLAY_TOL."""
    import torch
    torch.set_num_threads(a.threads)
    I4 = I4 or C.load('ves2321_infer4', CODE / 'attempt4/infer4.py')
    dev = a.device
    nets = nets or nets_of(I4, dev)
    keys = old_keys()
    st, o44 = np.load(C.DATA / 'sites.npz'), np.load(C.DATA / 'sites44.npz')['vol']
    rep = {}
    for name, v in nets.items():
        oi = ([i for i, k in enumerate(keys) if k[0] == v['f']] if v['f'] else list(range(len(keys))))[:SL.N_REPLAY]
        if v['f']:
            zo, zs = np.load(HO / f'out_{name}.npz'), np.load(HO / f'out_{name}_shift.npz')
        else:
            zo, zs = None, np.load(C.DATA / f'out_{name}_shift.npz')
        ds = []
        for i in oi:
            site, sh = I4.outputs(v['net'], o44[i], st['ok'][i], st['ra'][i], st['ca'][i], st[f'rr_{i}'],
                                  st[f'cc_{i}'], v['mu'], v['sd'])
            if zo is not None:
                ds.append(np.abs(site - zo[f'p_{i}']).max())
            ds += [np.abs(sh[k].astype(np.float32) - zs[f'p{k}_{i}'].astype(np.float32)).max() for k in sh]
        d = float(np.max(np.asarray(ds, dtype=np.float64)))          # NaN-propagating
        rep[name] = dict(d=d, n_sites=len(oi))
        log(f'replay {name}: max |diff| {d:.2e} on {len(oi)} old sites')
    ok = all(np.isfinite(v['d']) and v['d'] <= SL.REPLAY_TOL and v['n_sites'] >= SL.N_REPLAY for v in rep.values())
    return dict(replay=rep, ok=bool(ok), device=dev, nets=nets, I4=I4)


def infer(a):
    s_sha = site_sha_committed()
    z = np.load(DATA / 'points_lf.npz')
    assert str(z['site_sha']) == s_sha and int(z['n_sites']) == len(json.loads(
        (DATA / 'sites_lf.json').read_text())['site_keys']), 'points_lf.npz is not of the committed site list'
    T0 = time.time()
    R = replay_all(a)                                # pass 1: every replay before any Will output
    assert R['ok'], f"replay fails: {R['replay']}"
    I4, nets = R['I4'], R['nets']
    t = np.load(DATA / 'tiles44_lf.npz')
    V, OK, RA, CA = t['vol'], t['ok'], t['ra'], t['ca']
    n = len(RA)
    RR, CC = [t[f'rr_{j}'] for j in range(n)], [t[f'cc_{j}'] for j in range(n)]
    secs = {}
    for name, v in nets.items():                     # pass 2: every net on every Will site
        o, s = {}, {}
        for j in range(n):
            site, sh = I4.outputs(v['net'], V[j], OK[j], RA[j], CA[j], RR[j], CC[j], v['mu'], v['sd'])
            if v['f']:
                o[f'p_{j}'] = site
            s.update({f'p{k}_{j}': x for k, x in sh.items()})
        if v['f']:                                   # P0 / P1: shift outputs only (score_lf.Bundle.pshift)
            np.savez_compressed(DATA / f'out_{name}_lf.npz', **o)
        np.savez_compressed(DATA / f'out_{name}_lf_shift.npz', **s)
        secs[name] = round(time.time() - T0, 1)
        log(f'{name}: {n} Will sites, {time.time() - T0:.0f} s')
    info = dict(run_dir=str(C.DATA), site_sha=s_sha, nets={k: v['sha'] for k, v in nets.items()},
                replay=R['replay'], replay_tol=SL.REPLAY_TOL, device=a.device, threads=a.threads, seconds=secs,
                total_seconds=round(time.time() - T0, 1), code=str(CODE), pins_commit=SL.PINS_COMMIT)
    (DATA / 'infer_lf.json').write_text(json.dumps(info, indent=1))
    (HERE / 'infer_lf.json').write_text(json.dumps(info, indent=1))
    log(f"infer done: {info['total_seconds']} s on {a.device}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('g2', 'selftest', 'points', 'infer'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    dict(g2=g2, selftest=selftest, points=points, infer=infer)[a.cmd](a)
