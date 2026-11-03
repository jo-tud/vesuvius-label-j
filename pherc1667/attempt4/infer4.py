"""VES-2264 (VES-2241 attempt 4) CPU inference of the unchanged F-400 nets on the new sites (attempt4/sites4.py).
Per fold f: J0, J1, D0, D1 trained without segment f (DATA/heldout/net_<n>_h<f>.pt, own norm from train_<n>_h<f>.json)
on the new sites of f only; P0, P1 (attempt 2, DATA/net_P<s>.pt, attempt-1 norm) on all new sites (known-artefact
control). Same input, planes and point sampling as train_fold.py / train.frame_outputs:
  site output   p_<j>   (2 phases, n, 16)  planes SCORE_K of the offset-0 window (44-plane render, planes JIT..JIT+31,
                                           which prep3.py asserted equals the 32-plane render byte for byte)
  shift output  p<k>_<j> (n, 32) float16   window at mesh offset k in (-4, 0, 4), mean of the 2 phases
j = index into sites4.json site_keys. First, the replay check: each fold net run through this code on its fold's first
4 old sites must reproduce its stored F-400 outputs within REPLAY_TOL (GPU vs CPU float); a fail stops the run.

  python attempt4/infer4.py [--threads 12] [--folds w035 ...]
Writes DATA/attempt4/out_<n>_h<f>.npz, out_<n>_h<f>_shift.npz, out_P<s>_shift.npz, infer4.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
import train as T  # noqa: E402

OUT = C.DATA / 'attempt4'
HO = C.DATA / 'heldout'
FOLDS = C.FOLDS                                      # VES-2321
NAMES = [f'{A}{s}' for A in ('J', 'D') for s in C.SEEDS]
REPLAY_TOL = 5e-3
N_REPLAY = 4


def net_of(path):
    net = T.UNet3D()
    net.load_state_dict(torch.load(path, map_location='cpu'))
    return net.eval()


def outputs(net, vol44, ok, ra, ca, rr, cc, mu, sd):
    """(site output (2, n, 16) float32, {k: shift output (n, 32) float16}) of one site."""
    ks = (C.SHIFT_K[0], 0, C.SHIFT_K[1])
    x = np.stack([T.site_input(vol44, ok, C.JIT + k, off, mu, sd) for k in ks for off in (0, C.PHASE)])
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(x[:, None]))).numpy()[:, 0]          # (6, 32, 96, 96)
    sh, site = {}, None
    for a, k in enumerate(ks):
        ph = [p[2 * a + b][:, rr - ra - off, cc - ca - off].T for b, off in enumerate((0, C.PHASE))]   # (n, 32) x 2
        sh[k] = np.mean(ph, 0).astype(np.float16)
        if k == 0:
            site = np.stack([q[:, C.SCORE_K] for q in ph]).astype(np.float32)
    return site, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--folds', nargs='*', default=list(FOLDS))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    T0 = time.time()
    info = json.loads((OUT / 'sites4.json').read_text())
    assert info['power']['ok'] and info['replay']['ok'] and info['limit'] is None, 'sites4 not a full, passing prep'
    assert C.F395_POINTS_SHA != 'not-built'
    keys = info['site_keys']
    s4 = np.load(OUT / 'sites4.npz')
    p4 = np.load(OUT / 'points4.npz')
    V, OK, RA, CA = s4['vol'], s4['ok'], s4['ra'], s4['ca']
    RR = [p4[f'rr_{j}'] for j in range(len(keys))]
    CC = [p4[f'cc_{j}'] for j in range(len(keys))]
    meta = json.loads((C.DATA / 'prep.json').read_text())
    old_st, old44 = np.load(C.DATA / 'sites.npz'), np.load(C.DATA / 'sites44.npz')['vol']
    log = dict(replay={}, seconds={})

    def check(name, d):
        log['replay'][name] = d
        assert np.isfinite(d) and d <= REPLAY_TOL, f'replay {name}: max |diff| {d} > {REPLAY_TOL}: attempt 4 VOID'

    def dmax(ds):                                                          # NaN-propagating max (builtin max drops NaN)
        return float(np.max(np.asarray(list(ds), dtype=np.float64)))

    runs = {}
    for f in a.folds:                                                      # pass 1: every replay, before any new-site output
        oi = [i for i, k in enumerate(meta['site_keys']) if k[0] == f][:N_REPLAY]
        for n in NAMES:
            tr = json.loads((HO / f'train_{n}_h{f}.json').read_text())
            mu, sd = tr['norm']['mean'], tr['norm']['std']
            net = net_of(HO / f'net_{n}_h{f}.pt')
            zo, zs = np.load(HO / f'out_{n}_h{f}.npz'), np.load(HO / f'out_{n}_h{f}_shift.npz')
            ds = []
            for i in oi:
                site, sh = outputs(net, old44[i], old_st['ok'][i], old_st['ra'][i], old_st['ca'][i],
                                   old_st[f'rr_{i}'], old_st[f'cc_{i}'], mu, sd)
                ds.append(np.abs(site - zo[f'p_{i}']).max())
                ds += [np.abs(sh[k].astype(np.float32) - zs[f'p{k}_{i}'].astype(np.float32)).max() for k in sh]
            check(f'{n}_h{f}', dmax(ds))
            runs[f'{n}_h{f}'] = (f, net, mu, sd)
            print(f'{n}_h{f}: replay max |diff| {dmax(ds):.2e}, {time.time() - T0:.0f} s', flush=True)
    mu0, sd0 = meta['norm']['mean'], meta['norm']['std']
    pnets = {}
    for n in ('P0', 'P1'):
        net = net_of(C.DATA / f'net_{n}.pt')
        zs = np.load(C.DATA / f'out_{n}_shift.npz')
        ds = []
        for i in range(N_REPLAY):
            _, sh = outputs(net, old44[i], old_st['ok'][i], old_st['ra'][i], old_st['ca'][i], old_st[f'rr_{i}'],
                            old_st[f'cc_{i}'], mu0, sd0)
            ds += [np.abs(sh[k].astype(np.float32) - zs[f'p{k}_{i}'].astype(np.float32)).max() for k in sh]
        check(n, dmax(ds))
        pnets[n] = net
        print(f'{n}: replay max |diff| {dmax(ds):.2e}, {time.time() - T0:.0f} s', flush=True)
    for name, (f, net, mu, sd) in runs.items():                            # pass 2: new-site outputs
        jj = [j for j, k in enumerate(keys) if k[0] == f]
        o, s = {}, {}
        for j in jj:
            site, sh = outputs(net, V[j], OK[j], RA[j], CA[j], RR[j], CC[j], mu, sd)
            o[f'p_{j}'] = site
            s.update({f'p{k}_{j}': v for k, v in sh.items()})
        np.savez_compressed(OUT / f'out_{name}.npz', **o)
        np.savez_compressed(OUT / f'out_{name}_shift.npz', **s)
        log['seconds'][name] = round(time.time() - T0, 1)
        print(f'{name}: {len(jj)} new sites, {time.time() - T0:.0f} s', flush=True)
    for n, net in pnets.items():
        s = {}
        for j in range(len(keys)):
            if keys[j][0] in a.folds:
                _, sh = outputs(net, V[j], OK[j], RA[j], CA[j], RR[j], CC[j], mu0, sd0)
                s.update({f'p{k}_{j}': v for k, v in sh.items()})
        np.savez_compressed(OUT / f'out_{n}_shift.npz', **s)
        print(f'{n}: new-site shifts, {time.time() - T0:.0f} s', flush=True)
    log.update(folds=a.folds, threads=a.threads, total_seconds=round(time.time() - T0, 1), replay_tol=REPLAY_TOL)
    (OUT / 'infer4.json').write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
