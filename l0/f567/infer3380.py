"""VES-3380 (attempt 3) = ../2026-11-07-ves3347-f485-jd-folds/infer3347.py with D0_hw029 trained for STEPS = 24 000
steps (the one registered change) and paths VES3380_OUT / the ves3380 overlay. Text of infer3347 below.
VES-3347 step 1 GPU inference: the new F-485-recipe net D0_hw029 (arm D, seed 0, trained without w029 by F-485's
heldout/train_fold.py, unchanged) on the 320 held-out sites of fold w029. No training. Code = ../2026-11-06-ves3226-
f485-regate/infer.py (F-545's inference, = attempt4/infer4.py's calls) with the D rule of infer4.py: arm D gets the
k = 0 window only (infer4.KS_OF['D'] = (0,): D has no frame test, F-485 README), both grid phases, on label and N1
sites. The J nets (J0, J1) and P0 / P1 of w029 are NOT re-run: their w029 outputs exist in LABELJ_DATA/ves3226
(F-545) and score3347.py reads them there.
Gates first, as infer4 / infer.py (before any site output): net replay of D0_hw029 (<= 5e-3 against the outputs
train_fold.py stored on the replay sites), site replay (replay points equal points.npz; every w029 replay tile byte-
equal to prep.py's render). A wall-time guard refuses to start the site pass if it would pass --cap-h (NOT-RUN).

  VES2578_RUN=.../ves3380/run python infer3380.py [--workers 10] [--cap-h 0.9] [--out DIR]
Writes OUT/out_D0_hw029.npz (site outputs p_<u>), OUT/out_D0_hw029_shift.npz (k = 0 mean of 2 phases, const),
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
import infer4 as I4  # noqa: E402   (net_of, tiles, REPLAY_TOL, KS_OF: unchanged)

HO = C.DATA / 'heldout'
FOLD = 'w029'
NETS = ('D0',)
STEPS = 24000                                             # attempt 3 (was 12000)
S_PER_WINDOW = 0.94                                       # F-545 measured rate class (S3 tile render bound), guard only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--cap-h', type=float, default=0.9)
    ap.add_argument('--out', default=os.environ.get('VES3380_OUT', _VD + '/ves3380/out'))
    a = ap.parse_args()
    assert I4.KS_OF['D'] == (0,)
    OUT = Path(a.out)
    ex = ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'))     # before any CUDA use
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    dev = torch.device(a.device)
    T0 = time.time()
    info = json.loads((C.DATA / 'attempt4' / 'sites4.json').read_text())
    assert (info['power']['ok'] and info['limit'] is None) or C.SMOKE, 'sites4 not a full, passing prep'
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
    assert not (logf.exists() and FOLD in json.loads(logf.read_text()).get('done', [])), \
        f'{logf}: {FOLD} already inferred (one read only, HELDOUT.md): no re-run'
    log = dict(replay={}, replay_sites={}, seconds={}, done=[], windows={}, not_run=[])

    for u, i in rep.items():                                               # site replay: points
        same = bool(np.array_equal(rp[f'rr_{i}'], RR[u]) and np.array_equal(rp[f'cc_{i}'], CC[u]))
        log['replay_sites'][f'points_{u}'] = same
        assert same, f'replay site {u}: points != points.npz'

    fn = {}
    for n in NETS:                                                         # net replay, before any site output
        tr = json.loads((HO / f'train_{n}_h{FOLD}.json').read_text())
        assert (tr['fold'], tr['arm'], tr['seed'], tr['steps'], tr['patch'], tr['batch']) == \
            (FOLD, n[0], int(n[1]), STEPS, T.PATCH, 16), f'{n}: not the registered run'
        net, mu, sd = I4.net_of(HO / f'net_{n}_h{FOLD}.pt', dev), tr['norm']['mean'], tr['norm']['std']
        z = np.load(HO / f'out_{n}_h{FOLD}_replay.npz')
        ds = []
        for u in sorted(int(k[2:]) for k in z.files if k.startswith('p_')):
            i = rep[u]
            site, sh = T.outputs(net, dev, rp['vol'][i], rp['ok'][i], rp['ra'][i], rp['ca'][i], RR[u], CC[u], mu, sd)
            ds.append(np.abs(site - z[f'p_{u}']).max())
            ds += [np.abs(sh[k].astype(np.float32) - z[f'p{k}_{u}'].astype(np.float32)).max() for k in sh]
        d = float(np.max(np.asarray(ds, dtype=np.float64)))
        log['replay'][f'{n}_h{FOLD}'] = d
        assert np.isfinite(d) and d <= I4.REPLAY_TOL, f'replay {n}: max |diff| {d} > {I4.REPLAY_TOL}: VOID'
        print(f'{n}_h{FOLD}: replay max |diff| {d:.2e}, {time.time() - T0:.0f} s', flush=True)
        fn[n] = (net, mu, sd)
    logf.write_text(json.dumps(log, indent=1))

    uu = [u for u, k in enumerate(keys) if k[0] == FOLD]
    nwin = len(NETS) * 2 * len(uu)                                        # k = 0, 2 phases, every site
    if (time.time() - T0) + 1.1 * S_PER_WINDOW * nwin > a.cap_h * 3600:
        print(f'BUDGET STOP before fold {FOLD}: est {1.1 * S_PER_WINDOW * nwin:.0f} s, cap {a.cap_h} h', flush=True)
        log['not_run'].append(FOLD)
    else:
        t = time.time()
        jobs = [dict(u=u, w=FOLD, ra=keys[u][3] - C.TILE, ca=keys[u][4] - C.TILE) for u in uu]
        o = {n: {} for n in NETS}
        s = {n: {} for n in NETS}
        for u, vol, ok in I4.tiles(ex, jobs, 2 * a.workers):
            if u in rep:
                i = rep[u]
                same = bool(np.array_equal(vol, rp['vol'][i]) and np.array_equal(ok, rp['ok'][i]))
                log['replay_sites'][f'tile_{u}'] = same
                assert same, f'replay site {u}: tile != prep.py render'
            ra, ca = keys[u][3] - C.TILE, keys[u][4] - C.TILE
            for n, (net, mu, sd) in fn.items():
                site, sh = T.outputs(net, dev, vol, ok, ra, ca, RR[u], CC[u], mu, sd, ks=I4.KS_OF['D'])
                o[n][f'p_{u}'] = site
                s[n].update({f'p{k}_{u}': v for k, v in sh.items()})
        for n, (net, _, _) in fn.items():
            s[n]['const'] = T.const_output(net, dev)
            np.savez_compressed(OUT / f'out_{n}_h{FOLD}.npz', **o[n])
            np.savez_compressed(OUT / f'out_{n}_h{FOLD}_shift.npz', **s[n])
        sec = time.time() - t
        log['seconds'][FOLD] = round(sec, 1)
        log['windows'][FOLD] = nwin
        log['done'].append(FOLD)
        print(f'fold {FOLD}: {len(uu)} sites, {nwin} windows, {sec:.0f} s ({sec / nwin:.3f} s/window)', flush=True)
    ex.shutdown()
    log.update(folds=log['done'], nets={f: list(NETS) for f in log['done']}, replay_tol=I4.REPLAY_TOL,
               total_seconds=round(time.time() - T0, 1), cap_h=a.cap_h, data=str(C.DATA),
               gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    logf.write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'replay_sites'}, indent=1))


if __name__ == '__main__':
    main()
