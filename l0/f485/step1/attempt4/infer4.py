"""VES-2264 (VES-2241 attempt 4) inference of the fold nets on the sites (attempt4/sites4.py).
Per fold f: J0, J1, D0, D1 trained without segment f (DATA/heldout/net_<n>_h<f>.pt, own norm from train_<n>_h<f>.json)
on the sites of f only; P0, P1 (DATA/net_P<s>.pt, all-block norm) on the sites of every inferred fold (known-artefact
control). Same input, planes and point sampling as F-418 (train.outputs):
  site output   p_<u>   (2 phases, n, 64)  slab planes SCORE_K of the offset-0 window (planes JIT .. JIT + 127)
  shift output  p<k>_<u> (n, 128) float16  window at mesh offset k, mean of the 2 phases; k in (-16, 0, 16) for J, P;
                                           k = 0 only for D (F-418 read D's frame test as a reading only; no rule uses
                                           it, and k = +-16 for D would cost 0.94 GPU-h at level 0)
u = index into sites4.json site_keys.
VES-2578 step 1 (level 0), changes against F-418's infer4.py: runs on the GPU (a level-0 window is 64 x the voxels
of a level-2 one); the site tiles (176 planes TWIDE, 400 x 400 px, common.render) are rendered here by CPU worker
processes in step with the GPU and not stored. Gates first (pass 1, before any site output): net replay = each net
loaded from disk reproduces, on its replay sites (prep.py replay.npz), the outputs it wrote in its training process
within REPLAY_TOL; site replay = the points of the replay sites equal points.npz, and (pass 2) a replay site's tile
rendered here equals prep.py's render byte for byte. A fail stops the run.

  python attempt4/infer4.py [--folds w028 ...] [--workers 10] [--device cuda]   (VES2578_SMOKE=N: smoke subset)
Writes DATA/attempt4/out_<n>_h<f>.npz, out_<n>_h<f>_shift.npz, out_P<s>_shift_h<f>.npz, infer4.json.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
import train as T  # noqa: E402

OUT = C.DATA / 'attempt4'
HO = C.DATA / 'heldout'
FOLDS = C.FOLDS
NAMES = [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
PNETS = ('P0', 'P1')
KS = (C.SHIFT_K[0], 0, C.SHIFT_K[1])
KS_OF = {'J': KS, 'D': (0,), 'P': KS}
REPLAY_TOL = 5e-3
_G = {}


def net_of(path, dev):
    net = T.UNet3D()
    net.load_state_dict(torch.load(path, map_location='cpu'))
    return net.to(dev).eval()


def render_job(j):
    """One site tile (CPU worker): 176 planes, 2 x TILE + PHASE px square, as prep.py's replay tiles."""
    t23 = C.f389().t23
    if j['w'] not in _G:
        _G.clear()
        g, v, nrm = C.grid0(j['w'])
        _G[j['w']] = (g, nrm, v)
    g, nrm, v = _G[j['w']]
    n2 = 2 * C.TILE + C.PHASE
    vol, ok = C.render(t23, g, nrm, v, j['ra'], j['ra'] + n2, j['ca'], j['ca'] + n2, T=C.TWIDE)
    return j['u'], vol, ok


def tiles(ex, jobs, k):
    """render_job results with at most k jobs in flight (completion order)."""
    it, fut = iter(jobs), set()
    for j in it:
        fut.add(ex.submit(render_job, j))
        if len(fut) >= k:
            done, fut = wait(fut, return_when=FIRST_COMPLETED)
            yield from (f.result() for f in done)
    while fut:
        done, fut = wait(fut, return_when=FIRST_COMPLETED)
        yield from (f.result() for f in done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='*', default=list(FOLDS))
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    ex = ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'))     # before any CUDA use
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    dev = torch.device(a.device)
    T0 = time.time()
    info = json.loads((OUT / 'sites4.json').read_text())
    assert (info['power']['ok'] and info['limit'] is None) or C.SMOKE, 'sites4 not a full, passing prep'
    assert C.F395_POINTS_SHA != 'not-built' or C.SMOKE
    keys = info['site_keys']
    p4 = np.load(C.F395_POINTS)
    RR = [p4[f'rr_{u}'] for u in range(len(keys))]
    CC = [p4[f'cc_{u}'] for u in range(len(keys))]
    meta = json.loads((C.DATA / 'prep.json').read_text())
    assert meta['site_keys'] == keys, 'prep site order != sites4 site order'
    rp = np.load(C.DATA / 'replay.npz')
    rep = {int(u): i for i, u in enumerate(rp['u'])}
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / 'infer4.json'
    log = json.loads(logf.read_text()) if logf.exists() else dict(replay={}, replay_sites={}, seconds={}, done=[])

    def check(name, d):
        log['replay'][name] = d
        assert np.isfinite(d) and d <= REPLAY_TOL, f'replay {name}: max |diff| {d} > {REPLAY_TOL}: step 1 VOID'

    def dmax(ds):                                                          # NaN-propagating max (builtin max drops NaN)
        return float(np.max(np.asarray(list(ds), dtype=np.float64)))

    for u, i in rep.items():                                               # site replay: points
        same = bool(np.array_equal(rp[f'rr_{i}'], RR[u]) and np.array_equal(rp[f'cc_{i}'], CC[u]))
        log['replay_sites'][f'points_{u}'] = same
        assert same, f'replay site {u}: points != points.npz'

    def replay(name, net, mu, sd, path):
        z = np.load(path)
        ds = []
        for u in sorted(int(k[2:]) for k in z.files if k.startswith('p_')):
            i = rep[u]
            site, sh = T.outputs(net, dev, rp['vol'][i], rp['ok'][i], rp['ra'][i], rp['ca'][i], RR[u], CC[u], mu, sd)
            ds.append(np.abs(site - z[f'p_{u}']).max())
            ds += [np.abs(sh[k].astype(np.float32) - z[f'p{k}_{u}'].astype(np.float32)).max() for k in sh]
        check(name, dmax(ds))
        print(f'{name}: replay max |diff| {dmax(ds):.2e}, {time.time() - T0:.0f} s', flush=True)

    nets = {}
    for n in PNETS:                                                        # pass 1: every replay, before any output
        tr = json.loads((C.DATA / f'train_{n}.json').read_text())
        net = net_of(C.DATA / f'net_{n}.pt', dev)
        replay(n, net, tr['norm']['mean'], tr['norm']['std'], C.DATA / f'out_{n}_replay.npz')
        nets[n] = (net, tr['norm']['mean'], tr['norm']['std'])
    folds = [f for f in a.folds if all((HO / f'train_{n}_h{f}.json').exists() for n in NAMES)]
    for f in folds:
        for n in NAMES:
            tr = json.loads((HO / f'train_{n}_h{f}.json').read_text())
            net = net_of(HO / f'net_{n}_h{f}.pt', dev)
            replay(f'{n}_h{f}', net, tr['norm']['mean'], tr['norm']['std'], HO / f'out_{n}_h{f}_replay.npz')
            del net
    logf.write_text(json.dumps(log, indent=1))
    for f in folds:                                                        # pass 2: site outputs, fold by fold
        if f in log['done']:
            continue
        t = time.time()
        fn = {}
        for n in NAMES:
            tr = json.loads((HO / f'train_{n}_h{f}.json').read_text())
            fn[n] = (net_of(HO / f'net_{n}_h{f}.pt', dev), tr['norm']['mean'], tr['norm']['std'])
        uu = [u for u, k in enumerate(keys) if k[0] == f]
        jobs = [dict(u=u, w=f, ra=keys[u][3] - C.TILE, ca=keys[u][4] - C.TILE) for u in uu]
        o = {n: {} for n in NAMES}
        s = {n: {} for n in NAMES + list(PNETS)}
        for u, vol, ok in tiles(ex, jobs, 2 * a.workers):
            if u in rep:
                i = rep[u]
                same = bool(np.array_equal(vol, rp['vol'][i]) and np.array_equal(ok, rp['ok'][i]))
                log['replay_sites'][f'tile_{u}'] = same
                assert same, f'replay site {u}: tile != prep.py render'
            ra, ca = keys[u][3] - C.TILE, keys[u][4] - C.TILE
            for n, (net, mu, sd) in list(fn.items()) + [(p, nets[p]) for p in PNETS]:
                site, sh = T.outputs(net, dev, vol, ok, ra, ca, RR[u], CC[u], mu, sd, ks=KS_OF[n[0]])
                if n in o:
                    o[n][f'p_{u}'] = site
                s[n].update({f'p{k}_{u}': v for k, v in sh.items()})
        for n, (net, _, _) in list(fn.items()) + [(p, nets[p]) for p in PNETS]:
            s[n]['const'] = T.const_output(net, dev)
        for n in NAMES:
            np.savez_compressed(OUT / f'out_{n}_h{f}.npz', **o[n])
            np.savez_compressed(OUT / f'out_{n}_h{f}_shift.npz', **s[n])
        for n in PNETS:
            np.savez_compressed(OUT / f'out_{n}_shift_h{f}.npz', **s[n])
        log['seconds'][f] = round(time.time() - t, 1)
        log['done'].append(f)
        logf.write_text(json.dumps(log, indent=1))
        print(f'fold {f}: {len(uu)} sites, {time.time() - t:.0f} s', flush=True)
        del fn
    ex.shutdown()
    log.update(folds=log['done'], replay_tol=REPLAY_TOL, total_seconds=round(time.time() - T0, 1),
               gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    logf.write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'replay_sites'}, indent=1))


if __name__ == '__main__':
    main()
