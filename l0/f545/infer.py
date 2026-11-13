"""VES-3226 GPU inference: the frozen F-485 attempt-2 nets of folds w029 (J0, J1) and w028 (J0) and the known-artefact
controls P0, P1 on the held-out sites of those two folds. No training. Code = step1/attempt4/infer4.py (F-485's
frozen inference, R9 VES-2643/2646) with three changes, all bookkeeping or cost (README "Inference"):
  1. nets: only the trained J nets of the two folds (infer4 wants J0, J1, D0, D1 per fold; D was never trained on
     w029, and w028 J1 / D1 were never trained);
  2. windows: shift windows k = -16, 0, +16 only on label sites (score.frame_control reads label sites only); on N1
     sites J gets the k = 0 window (site output, 2 phases) and P none (P's only rule is its frame test). Every
     output a frozen rule reads is computed exactly as infer4 does (same train.outputs call per window);
  3. outputs go to OUT (default LABELJ_DATA/ves3226), inputs are read from VES2578_RUN (step1a2), read only;
     a wall-time guard stops before a fold that would pass --cap-h (folds not run are NOT-RUN).
Gates first, as infer4 (pass 1, before any site output): net replay of every net used (<= 5e-3), site replay
(points, then tiles byte-equal to prep.py's render).

  VES2578_RUN=.../ves2578/step1a2 python infer.py [--folds w029 w028] [--workers 10] [--cap-h 0.6] [--out DIR]
Writes OUT/out_<n>_h<f>.npz (site outputs), OUT/out_<n>_h<f>_shift.npz (J shift), OUT/out_P<s>_shift_h<f>.npz,
OUT/infer.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np
import torch

STEP1 = Path(__file__).resolve().parent.parent / 'f485' / 'step1'
sys.path.insert(0, str(STEP1))
sys.path.insert(0, str(STEP1 / 'attempt4'))
import common as C  # noqa: E402
import train as T  # noqa: E402
import infer4 as I4  # noqa: E402   (net_of, render_job, tiles, REPLAY_TOL: unchanged)

HO = C.DATA / 'heldout'
NETS = {'w029': ('J0', 'J1'), 'w028': ('J0',)}          # every trained F-485 attempt-2 J net (run_folds.json)
PNETS = ('P0', 'P1')
KS = I4.KS                                               # (-16, 0, 16)
S_PER_WINDOW = 0.25                                      # guard estimate before the first fold (bench 0.219 s, gpu1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='*', default=['w029', 'w028'])
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--cap-h', type=float, default=0.6)
    ap.add_argument('--out', default=os.environ.get('VES3226_OUT', _VD + '/ves3226'))
    a = ap.parse_args()
    assert set(a.folds) <= set(NETS), a.folds
    OUT = Path(a.out)
    ex = ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'))     # before any CUDA use
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    dev = torch.device(a.device)
    T0 = time.time()
    info = json.loads((C.DATA / 'attempt4' / 'sites4.json').read_text())
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
    logf = OUT / 'infer.json'
    log = json.loads(logf.read_text()) if logf.exists() else dict(replay={}, replay_sites={}, seconds={}, done=[],
                                                                  windows={}, not_run=[])

    def check(name, d):
        log['replay'][name] = d
        assert np.isfinite(d) and d <= I4.REPLAY_TOL, f'replay {name}: max |diff| {d} > {I4.REPLAY_TOL}: VOID'

    def dmax(ds):
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

    def load(path_net, path_tr):
        tr = json.loads(path_tr.read_text())
        return I4.net_of(path_net, dev), tr['norm']['mean'], tr['norm']['std']

    nets = {}
    for n in PNETS:                                                        # pass 1: every replay, before any output
        nets[n] = load(C.DATA / f'net_{n}.pt', C.DATA / f'train_{n}.json')
        replay(n, *nets[n], C.DATA / f'out_{n}_replay.npz')
    for f in a.folds:
        for n in NETS[f]:
            net, mu, sd = load(HO / f'net_{n}_h{f}.pt', HO / f'train_{n}_h{f}.json')
            replay(f'{n}_h{f}', net, mu, sd, HO / f'out_{n}_h{f}_replay.npz')
            del net
    logf.write_text(json.dumps(log, indent=1))
    rate = S_PER_WINDOW
    for f in a.folds:                                                      # pass 2: site outputs, fold by fold
        if f in log['done']:
            continue
        uu = [u for u, k in enumerate(keys) if k[0] == f]
        lab = {u for u in uu if keys[u][1] == 'label'}
        nwin = sum((len(NETS[f]) + len(PNETS)) * len(KS) * 2 if u in lab else len(NETS[f]) * 2 for u in uu)
        if (time.time() - T0) + 1.1 * rate * nwin > a.cap_h * 3600:
            print(f'BUDGET STOP before fold {f}: {time.time() - T0:.0f} s used, est {1.1 * rate * nwin:.0f} s, '
                  f'cap {a.cap_h} h', flush=True)
            log['not_run'].append(f)
            break
        t = time.time()
        fn = {n: load(HO / f'net_{n}_h{f}.pt', HO / f'train_{n}_h{f}.json') for n in NETS[f]}
        jobs = [dict(u=u, w=f, ra=keys[u][3] - C.TILE, ca=keys[u][4] - C.TILE) for u in uu]
        o = {n: {} for n in NETS[f]}
        s = {n: {} for n in NETS[f] + PNETS}
        for u, vol, ok in I4.tiles(ex, jobs, 2 * a.workers):
            if u in rep:
                i = rep[u]
                same = bool(np.array_equal(vol, rp['vol'][i]) and np.array_equal(ok, rp['ok'][i]))
                log['replay_sites'][f'tile_{u}'] = same
                assert same, f'replay site {u}: tile != prep.py render'
            ra, ca = keys[u][3] - C.TILE, keys[u][4] - C.TILE
            for n, (net, mu, sd) in fn.items():                           # J: site output always, shift on label
                site, sh = T.outputs(net, dev, vol, ok, ra, ca, RR[u], CC[u], mu, sd, ks=KS if u in lab else (0,))
                o[n][f'p_{u}'] = site
                if u in lab:
                    s[n].update({f'p{k}_{u}': v for k, v in sh.items()})
            if u in lab:                                                   # P0, P1: frame test on label sites
                for n in PNETS:
                    pnet, pmu, psd = nets[n]
                    _, sh = T.outputs(pnet, dev, vol, ok, ra, ca, RR[u], CC[u], pmu, psd, ks=KS)
                    s[n].update({f'p{k}_{u}': v for k, v in sh.items()})
        for n, (net, _, _) in list(fn.items()) + [(p, nets[p]) for p in PNETS]:
            s[n]['const'] = T.const_output(net, dev)
        for n in NETS[f]:
            np.savez_compressed(OUT / f'out_{n}_h{f}.npz', **o[n])
            np.savez_compressed(OUT / f'out_{n}_h{f}_shift.npz', **s[n])
        for n in PNETS:
            np.savez_compressed(OUT / f'out_{n}_shift_h{f}.npz', **s[n])
        sec = time.time() - t
        rate = sec / nwin
        log['seconds'][f] = round(sec, 1)
        log['windows'][f] = nwin
        log['done'].append(f)
        logf.write_text(json.dumps(log, indent=1))
        print(f'fold {f}: {len(uu)} sites, {nwin} windows, {sec:.0f} s ({rate:.3f} s/window)', flush=True)
        del fn
    ex.shutdown()
    log.update(folds=log['done'], nets={f: list(NETS[f]) for f in log['done']}, replay_tol=I4.REPLAY_TOL,
               total_seconds=round(time.time() - T0, 1), cap_h=a.cap_h, data=str(C.DATA),
               gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    logf.write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'replay_sites'}, indent=1))


if __name__ == '__main__':
    main()
