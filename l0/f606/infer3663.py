"""VES-3663 (F-567 control) GPU inference on fold w029: the two D nets of this task on the 320 held-out w029 sites.
  D12s0 = F-560's D0_hw029 (arm D, seed 0, 12 000 steps; LABELJ_DATA/ves3347; trained 2026-11-07, never read);
  D24s1 = this task's D1_hw029 (arm D, seed 1, 24 000 steps; F-485's heldout/train_fold.py unchanged).
No training. Code = ../f567/step2/infer_s2.py with explicit net paths and the D rule only
(infer4.KS_OF['D']: k = 0 window, both phases, every site; constant-input output). J0, J1 (F-545) and D24s0 (F-567
step 1) outputs on w029 exist and are not re-run.
Gates first (before any site output): net replay of each net (<= 5e-3 against its stored replay outputs) and the
registered run of each net (fold, arm, seed, steps, patch, batch); site replay (points, then every w029 replay tile
byte-equal to prep.py's render). A wall-time guard refuses the site pass if it would pass --cap-h (NOT-RUN).
One read: refuses if infer.json says done. Only run by unit.sh stage R, after the R9 PASS.
Smoke (before R9, no new read): --smoke N runs F-567's D24s0 on the first N label and first N N1 w029 sites into
--out (a scratch dir) and checks its outputs equal F-567's stored ones (ves3380/out) within 5e-3.

  VES2578_RUN=.../ves3663/run python infer3663.py [--workers 10] [--cap-h 2.0] [--out DIR] [--smoke N]
Writes OUT/out_<n>_hw029.npz (site outputs p_<u>), OUT/out_<n>_hw029_shift.npz (k = 0 windows, const), OUT/infer.json.
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

FOLD = 'w029'
VD = Path(_VD + '')
HO = C.DATA / 'heldout'
# name: (net dir, file stem, seed, steps)
NETS = {'D12s0': (VD / 'ves3347/run/heldout', 'D0_hw029', 0, 12000),
        'D24s1': (HO, 'D1_hw029', 1, 24000)}
SMOKE_NETS = {'D24s0': (VD / 'ves3380/run/heldout', 'D0_hw029', 0, 24000)}
REF_D24S0 = VD / 'ves3380/out'                            # F-567 step 1 D0 w029 outputs (smoke reference)
S_PER_TILE = 13.5                                         # guard only; F-567 D0 w029: 4 136 s for 320 tiles (render bound)
OUT_DEF = Path(os.environ.get('VES3663_OUT', _VD + '/ves3663/out'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--cap-h', type=float, default=2.0)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--out', default=str(OUT_DEF))
    a = ap.parse_args()
    OUT = Path(a.out)
    if a.smoke:
        assert OUT.resolve() != OUT_DEF.resolve(), 'smoke: a scratch dir only'
    # D24s1 killed at its training cap or never trained (NOT-RUN): the read runs D12s0 alone (Q1 does not need it)
    nets_reg = SMOKE_NETS if a.smoke else {n: v for n, v in NETS.items() if (v[0] / f'train_{v[1]}.json').exists()}
    assert a.smoke or 'D12s0' in nets_reg, 'D12s0 (F-560) missing'
    assert I4.KS_OF['D'] == (0,)
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
    assert not (logf.exists() and json.loads(logf.read_text()).get('done')), f'{logf}: w029 already read: no re-run'
    log = dict(fold=FOLD, replay={}, replay_sites={}, registered={}, done=False, not_run=False, smoke=a.smoke)

    for u, i in rep.items():                                               # site replay: points
        same = bool(np.array_equal(rp[f'rr_{i}'], RR[u]) and np.array_equal(rp[f'cc_{i}'], CC[u]))
        log['replay_sites'][f'points_{u}'] = same
        assert same, f'replay site {u}: points != points.npz'

    nets = {}
    kept = sorted(set(C.FOLDS) - {FOLD})
    for n, (d, stem, seed, steps) in nets_reg.items():                     # every replay before any site output
        tr = json.loads((d / f'train_{stem}.json').read_text())
        reg = bool((tr['fold'], tr['arm'], tr['seed'], tr['steps'], tr['patch'], tr['batch']) ==
                   (FOLD, 'D', seed, steps, T.PATCH, 16) and tr['kept_segments'] == kept and tr['limit_blocks'] is None)
        log['registered'][n] = reg
        assert reg, f'{n}: not the registered run'
        net, mu, sd = I4.net_of(d / f'net_{stem}.pt', dev), tr['norm']['mean'], tr['norm']['std']
        z = np.load(d / f'out_{stem}_replay.npz')
        ds = []
        for u in sorted(int(k[2:]) for k in z.files if k.startswith('p_')):
            i = rep[u]
            site, sh = T.outputs(net, dev, rp['vol'][i], rp['ok'][i], rp['ra'][i], rp['ca'][i], RR[u], CC[u], mu, sd)
            ds.append(np.abs(site - z[f'p_{u}']).max())
            ds += [np.abs(sh[k].astype(np.float32) - z[f'p{k}_{u}'].astype(np.float32)).max() for k in sh]
        dmax = float(np.max(np.asarray(ds, dtype=np.float64)))
        log['replay'][n] = dmax
        assert np.isfinite(dmax) and dmax <= I4.REPLAY_TOL, f'replay {n}: max |diff| {dmax} > {I4.REPLAY_TOL}: VOID'
        print(f'{n}: replay max |diff| {dmax:.2e}, {time.time() - T0:.0f} s', flush=True)
        nets[n] = (net, mu, sd)
    logf.write_text(json.dumps(log, indent=1))

    uu = [u for u, k in enumerate(keys) if k[0] == FOLD]
    if a.smoke:                                                            # first N label + first N N1 sites
        uu = [u for u in uu if keys[u][1] == 'label'][:a.smoke] + [u for u in uu if keys[u][1] != 'label'][:a.smoke]
    est = 1.1 * S_PER_TILE * len(uu)
    if (time.time() - T0) + est > a.cap_h * 3600:
        print(f'BUDGET STOP before the w029 read: est {est:.0f} s, cap {a.cap_h} h', flush=True)
        log['not_run'] = True
    else:
        t = time.time()
        jobs = [dict(u=u, w=FOLD, ra=keys[u][3] - C.TILE, ca=keys[u][4] - C.TILE) for u in uu]
        o = {n: {} for n in nets}
        s = {n: {} for n in nets}
        for u, vol, ok in I4.tiles(ex, jobs, 2 * a.workers):
            if u in rep:
                i = rep[u]
                same = bool(np.array_equal(vol, rp['vol'][i]) and np.array_equal(ok, rp['ok'][i]))
                log['replay_sites'][f'tile_{u}'] = same
                assert same, f'replay site {u}: tile != prep.py render'
            ra, ca = keys[u][3] - C.TILE, keys[u][4] - C.TILE
            for n, (net, mu, sd) in nets.items():
                site, sh = T.outputs(net, dev, vol, ok, ra, ca, RR[u], CC[u], mu, sd, ks=I4.KS_OF['D'])
                o[n][f'p_{u}'] = site
                s[n].update({f'p{k}_{u}': v for k, v in sh.items()})
        for n in nets:
            s[n]['const'] = T.const_output(nets[n][0], dev)
            np.savez_compressed(OUT / f'out_{n}_h{FOLD}.npz', **o[n])
            np.savez_compressed(OUT / f'out_{n}_h{FOLD}_shift.npz', **s[n])
        sec = time.time() - t
        log.update(seconds=round(sec, 1), windows=2 * len(nets) * len(uu), sites=len(uu), done=True)
        print(f'w029: {len(uu)} sites, {sec:.0f} s', flush=True)
    ex.shutdown()
    log.update(nets=list(nets_reg), replay_tol=I4.REPLAY_TOL, total_seconds=round(time.time() - T0, 1),
               cap_h=a.cap_h, data=str(C.DATA), gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    logf.write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'replay_sites'}, indent=1))
    if a.smoke:
        r = smoke_compare(OUT, uu)
        print(json.dumps(r, indent=1))
        (OUT / 'smoke_compare.json').write_text(json.dumps(r, indent=1))
        sys.exit(0 if r['ok'] else 5)


def smoke_compare(out, uu):
    """Smoke outputs (D24s0 on the first N w029 sites) vs F-567 step 1's stored D0_hw029 outputs."""
    a, b = np.load(out / f'out_D24s0_h{FOLD}.npz'), np.load(REF_D24S0 / f'out_D0_h{FOLD}.npz')
    ds = [float(np.abs(a[f'p_{u}'] - b[f'p_{u}']).max()) for u in uu]
    sa, sb = np.load(out / f'out_D24s0_h{FOLD}_shift.npz'), np.load(REF_D24S0 / f'out_D0_h{FOLD}_shift.npz')
    dc = float(np.abs(sa['const'] - sb['const']).max())
    ok = bool(ds and max(ds) <= I4.REPLAY_TOL and dc <= I4.REPLAY_TOL)
    return dict(sites=len(uu), site_max_abs_diff=max(ds) if ds else None, const_max_abs_diff=dc, ok=ok)


if __name__ == '__main__':
    main()
