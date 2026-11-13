#!/usr/bin/env python3
"""Weight replay of the released level-0 nets: each net in weights/ is rerun on its four stored replay tiles and
compared with the outputs stored when it was trained (the net replay gate of the scored runs: max |diff| <= 5e-3,
`attempt4/infer4.REPLAY_TOL`, over the site output and the three frame-shift outputs). This ties the released weight
files to the outputs that the scorers read. It also checks that the replay tiles' points equal the scored points
(`points.npz`). Uses the unchanged `train.outputs` and `infer4.net_of`; no new place is read.

  python l0/check_weights.py [--weights weights] [--device cuda|cpu]
Needs LABELJ_DATA (default data/) with ves3951/replay_l0.npz and ves3416/run/points.npz (the stored asset).
Writes l0/check_weights.json; exit 1 if a net fails.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VD = Path(os.environ.get('LABELJ_DATA', ROOT / 'data'))
os.environ.setdefault('VES2578_RUN', str(VD / 'ves3416/run'))     # the step-2 run folder (points.npz)
sys.path.insert(0, str(HERE / 'f485' / 'step1'))
sys.path.insert(0, str(HERE / 'f485' / 'step1' / 'attempt4'))
import common as C  # noqa: E402
import train as T  # noqa: E402
import infer4 as I4  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(ROOT / 'weights'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
    W = Path(a.weights)
    meta = json.loads((HERE / 'weights.json').read_text())
    rp = np.load(VD / 'ves3951' / 'replay_l0.npz')
    rep = {int(u): i for i, u in enumerate(rp['u'])}
    p4 = np.load(C.F395_POINTS)
    points_ok = {u: bool(np.array_equal(rp[f'rr_{i}'], p4[f'rr_{u}']) and np.array_equal(rp[f'cc_{i}'], p4[f'cc_{u}']))
                 for u, i in rep.items()}
    dev = torch.device(a.device)
    res, t0 = {}, time.time()
    for name, m in meta.items():
        pt = W / f'{name}.pt'
        sha_ok = sha(pt) == m['sha256']
        tr = json.loads((W / f'{name}.train.json').read_text())
        net = I4.net_of(pt, dev)
        z = np.load(W / f'{name}.replay.npz')
        ds = []
        for u in sorted(int(k[2:]) for k in z.files if k.startswith('p_')):
            i = rep[u]
            site, sh = T.outputs(net, dev, rp['vol'][i], rp['ok'][i], rp['ra'][i], rp['ca'][i], rp[f'rr_{i}'],
                                 rp[f'cc_{i}'], tr['norm']['mean'], tr['norm']['std'])
            ds.append(np.abs(site - z[f'p_{u}']).max())
            ds += [np.abs(sh[k].astype(np.float32) - z[f'p{k}_{u}'].astype(np.float32)).max() for k in sh]
        d = float(np.max(np.asarray(ds, dtype=np.float64)))
        res[name] = dict(max_abs_diff=d, sha256_ok=sha_ok, sites=m['replay_sites'],
                         ok=bool(sha_ok and np.isfinite(d) and d <= I4.REPLAY_TOL))
        print(f'{name}: max |diff| {d:.2e} (tol {I4.REPLAY_TOL}), sha256 {"ok" if sha_ok else "MISMATCH"}, '
              f'{time.time() - t0:.0f} s', flush=True)
    ok = bool(all(r['ok'] for r in res.values()) and all(points_ok.values()) and len(res) == len(meta))
    out = dict(ok=ok, tol=I4.REPLAY_TOL, device=a.device, nets=res, replay_points_equal_scored_points=points_ok,
               seconds=round(time.time() - t0, 1))
    (HERE / 'check_weights.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print('weight replay', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
