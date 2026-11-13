"""VES-3416 (F-567 step 2) GPU inference of one fold: the step-2 nets J0_h<f> (12 000 steps) and D0_h<f> (24 000
steps), both trained by F-485's heldout/train_fold.py unchanged, and the known-artefact controls P0, P1 on the 320
held-out sites of fold f (w013, w018, w023). Fold w028: D0_hw028 only (as ../infer3380.py on w029): its J0 (F-485)
and P0 / P1 outputs exist from F-545 (LABELJ_DATA/ves3226) and are not re-run. No training. Code = ../../f545/infer.py (F-545: J and P windows)
plus the D rule of ../infer3380.py (D: k = 0 window only, both phases, every site; infer4.KS_OF['D']):
  J0: k = 0 site output on every site, shift windows k = -16, 0, +16 on label sites (frame test);
  D0: k = 0 site output on every site (D has no frame test);
  P0, P1: shift windows k = -16, 0, +16 on label sites (their only rule is the frame test).
Gates first (before any site output): net replay of P0, P1, J0_h<f>, D0_h<f> (<= 5e-3 against their stored replay
outputs), site replay (points, then every replay tile of the fold byte-equal to prep.py's render). A wall-time guard
refuses the site pass if it would pass --cap-h (fold NOT-RUN). One read per fold: refuses if infer_<f>.json says done.
Only run by unit.sh after the fold's gate 1 passed (score_s2.py --gate1 <f>).
Smoke (before R9, no new read): --smoke N runs fold w029 with its frozen nets (J0 F-485, D0 F-567) on the first N
label and first N N1 w029 sites into --out (a scratch dir); smoke_compare() then checks the outputs equal F-545's / F-567's stored w029
outputs (same sites) within 5e-3.

  VES2578_RUN=.../ves3416/run python infer_s2.py --fold w013 [--workers 10] [--cap-h 2.0] [--out DIR]
Writes OUT/out_<J0|D0>_h<f>.npz (site outputs p_<u>), OUT/out_<J0|D0>_h<f>_shift.npz (shift windows, const),
OUT/out_P<s>_shift_h<f>.npz, OUT/infer_<f>.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np
import torch

STEP1 = Path(__file__).resolve().parent.parent.parent / 'f485' / 'step1'
sys.path.insert(0, str(STEP1))
sys.path.insert(0, str(STEP1 / 'attempt4'))
import common as C  # noqa: E402
import train as T  # noqa: E402
import infer4 as I4  # noqa: E402   (net_of, render_job, tiles, REPLAY_TOL, KS, KS_OF: unchanged)

HO = C.DATA / 'heldout'
NEW = ('w013', 'w028', 'w018', 'w023')                    # step-2 folds, run order (w031: hecate F1 fails, not run)
NETS_OF = {'w013': ('J0', 'D0'), 'w028': ('D0',), 'w018': ('J0', 'D0'), 'w023': ('J0', 'D0')}
PNETS_OF = {'w013': ('P0', 'P1'), 'w028': (), 'w018': ('P0', 'P1'), 'w023': ('P0', 'P1')}
STEPS = {'J': 12000, 'D': 24000}                          # J = F-485's frozen recipe; D = F-567's 24 000 steps
KS = I4.KS                                                # (-16, 0, 16)
S_PER_WINDOW = 1.1                                        # guard only; F-545 0.94 s/window (w029, 4 480 windows)
S_PER_TILE = 13.5                                         # guard only; F-567 D0 w029: 4 136 s for 320 tiles (render bound)
S3416_OUT = Path(os.environ.get('VES3416_OUT', _VD + '/ves3416/out'))
REF = dict(J=Path(_VD + '/ves3226'), D=Path(_VD + '/ves3380/out'))


def windows(uu, lab, nets, pnets):
    """Windows per fold: J0 3 x 2 on label, 2 on N1; D0 2 on every site; P0, P1 3 x 2 on label."""
    nj, nd = sum(n[0] == 'J' for n in nets), sum(n[0] == 'D' for n in nets)
    return sum((len(KS) * 2 * (nj + len(pnets)) + 2 * nd) if u in lab else 2 * (nj + nd) for u in uu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--cap-h', type=float, default=2.0)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--out', default=str(S3416_OUT))
    a = ap.parse_args()
    f = a.fold
    if a.smoke:
        assert f == 'w029' and Path(a.out).resolve() != S3416_OUT.resolve(), 'smoke: fold w029 into a scratch dir only'
    else:
        assert f in NEW, f
    NETS, PNETS = (('J0', 'D0'), ('P0', 'P1')) if a.smoke else (NETS_OF[f], PNETS_OF[f])
    assert I4.KS_OF['D'] == (0,) and I4.KS_OF['J'] == KS
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
    logf = OUT / f'infer_{f}.json'
    assert not (logf.exists() and json.loads(logf.read_text()).get('done')), f'{logf}: {f} already read: no re-run'
    log = dict(fold=f, replay={}, replay_sites={}, done=False, not_run=False, smoke=a.smoke)

    def check(name, d):
        log['replay'][name] = d
        assert np.isfinite(d) and d <= I4.REPLAY_TOL, f'replay {name}: max |diff| {d} > {I4.REPLAY_TOL}: VOID'

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
        d = float(np.max(np.asarray(ds, dtype=np.float64)))
        check(name, d)
        print(f'{name}: replay max |diff| {d:.2e}, {time.time() - T0:.0f} s', flush=True)

    def load(path_net, path_tr):
        tr = json.loads(path_tr.read_text())
        return I4.net_of(path_net, dev), tr['norm']['mean'], tr['norm']['std'], tr

    nets = {}
    for n in PNETS:                                                        # every replay before any site output
        net, mu, sd, _ = load(C.DATA / f'net_{n}.pt', C.DATA / f'train_{n}.json')
        replay(n, net, mu, sd, C.DATA / f'out_{n}_replay.npz')
        nets[n] = (net, mu, sd)
    for n in NETS:
        net, mu, sd, tr = load(HO / f'net_{n}_h{f}.pt', HO / f'train_{n}_h{f}.json')
        assert (tr['fold'], tr['arm'], tr['seed'], tr['steps'], tr['patch'], tr['batch']) == \
            (f, n[0], 0, STEPS[n[0]], T.PATCH, 16), f'{n}_h{f}: not the registered run'
        replay(f'{n}_h{f}', net, mu, sd, HO / f'out_{n}_h{f}_replay.npz')
        nets[n] = (net, mu, sd)
    logf.write_text(json.dumps(log, indent=1))

    uu = [u for u, k in enumerate(keys) if k[0] == f]
    if a.smoke:                                                            # first N label + first N N1 sites
        uu = [u for u in uu if keys[u][1] == 'label'][:a.smoke] + [u for u in uu if keys[u][1] != 'label'][:a.smoke]
    lab = {u for u in uu if keys[u][1] == 'label'}
    nwin = windows(uu, lab, NETS, PNETS)
    est = 1.1 * max(S_PER_WINDOW * nwin, S_PER_TILE * len(uu))
    if (time.time() - T0) + est > a.cap_h * 3600:
        print(f'BUDGET STOP before fold {f}: est {est:.0f} s, cap {a.cap_h} h', flush=True)
        log['not_run'] = True
    else:
        t = time.time()
        jobs = [dict(u=u, w=f, ra=keys[u][3] - C.TILE, ca=keys[u][4] - C.TILE) for u in uu]
        o = {n: {} for n in NETS}
        s = {n: {} for n in NETS + PNETS}
        for u, vol, ok in I4.tiles(ex, jobs, 2 * a.workers):
            if u in rep:
                i = rep[u]
                same = bool(np.array_equal(vol, rp['vol'][i]) and np.array_equal(ok, rp['ok'][i]))
                log['replay_sites'][f'tile_{u}'] = same
                assert same, f'replay site {u}: tile != prep.py render'
            ra, ca = keys[u][3] - C.TILE, keys[u][4] - C.TILE
            for n in NETS:
                net, mu, sd = nets[n]
                ks = KS if (n[0] == 'J' and u in lab) else (0,)
                site, sh = T.outputs(net, dev, vol, ok, ra, ca, RR[u], CC[u], mu, sd, ks=ks)
                o[n][f'p_{u}'] = site
                if n[0] == 'D' or u in lab:
                    s[n].update({f'p{k}_{u}': v for k, v in sh.items()})
            if u in lab:
                for n in PNETS:
                    net, mu, sd = nets[n]
                    _, sh = T.outputs(net, dev, vol, ok, ra, ca, RR[u], CC[u], mu, sd, ks=KS)
                    s[n].update({f'p{k}_{u}': v for k, v in sh.items()})
        for n in NETS + PNETS:
            s[n]['const'] = T.const_output(nets[n][0], dev)
        for n in NETS:
            np.savez_compressed(OUT / f'out_{n}_h{f}.npz', **o[n])
            np.savez_compressed(OUT / f'out_{n}_h{f}_shift.npz', **s[n])
        for n in PNETS:
            np.savez_compressed(OUT / f'out_{n}_shift_h{f}.npz', **s[n])
        sec = time.time() - t
        log.update(seconds=round(sec, 1), windows=nwin, sites=len(uu), done=True)
        print(f'fold {f}: {len(uu)} sites, {nwin} windows, {sec:.0f} s ({sec / nwin:.3f} s/window)', flush=True)
    ex.shutdown()
    log.update(nets=list(NETS) + list(PNETS), replay_tol=I4.REPLAY_TOL, total_seconds=round(time.time() - T0, 1),
               cap_h=a.cap_h, data=str(C.DATA), gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    logf.write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'replay_sites'}, indent=1))
    if a.smoke:
        r = smoke_compare(OUT, f, uu, lab)
        print(json.dumps(r, indent=1))
        (OUT / 'smoke_compare.json').write_text(json.dumps(r, indent=1))
        sys.exit(0 if r['ok'] else 5)


def smoke_compare(out, f, uu, lab):
    """Smoke outputs (w029, first N sites) vs F-545's (J0, P0, P1; ves3226) and F-567's (D0; ves3380/out)."""
    d = {}
    pairs = [('J0', f'out_J0_h{f}.npz', REF['J'] / f'out_J0_h{f}.npz', uu, ('p_',)),
             ('J0_shift', f'out_J0_h{f}_shift.npz', REF['J'] / f'out_J0_h{f}_shift.npz', sorted(lab), ('p-16_', 'p0_', 'p16_')),
             ('D0', f'out_D0_h{f}.npz', REF['D'] / f'out_D0_h{f}.npz', uu, ('p_',))]
    pairs += [(f'{n}_shift', f'out_{n}_shift_h{f}.npz', REF['J'] / f'out_{n}_shift_h{f}.npz', sorted(lab),
               ('p-16_', 'p0_', 'p16_')) for n in ('P0', 'P1')]
    for name, mine, ref, us, pre in pairs:
        a, b = np.load(out / mine), np.load(ref)
        ds = [float(np.abs(a[f'{p}{u}'].astype(np.float32) - b[f'{p}{u}'].astype(np.float32)).max()) for u in us for p in pre]
        d[name] = dict(n=len(ds), max_abs_diff=max(ds) if ds else None)
    ok = all(v['n'] > 0 and v['max_abs_diff'] <= I4.REPLAY_TOL for v in d.values())
    return dict(sites=len(uu), label_sites=len(lab), diffs=d, ok=bool(ok))


if __name__ == '__main__':
    main()
