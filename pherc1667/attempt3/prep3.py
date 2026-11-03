"""VES-2241 attempt 3 (VES-2251) data prep, CPU, no model, no score: re-render attempt 1's training blocks and site
tiles (same windows, read from DATA/prep.json and DATA/sites.npz) on the 44 planes T44 = -21.5 .. 21.5 vx, so a
32-plane window can be cut at any mesh offset j in [-6, 6]. Asserts that the central 32 planes (offset 0) equal
attempt 1's renders byte for byte. Writes DATA/train44.npz, DATA/sites44.npz, DATA/prep3.json.

  python attempt3/prep3.py [--workers 10]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
from prep import seg  # noqa: E402

J0 = slice(C.JIT, C.JIT + C.NPL)            # offset-0 window inside the 44 planes


def job(j):
    t23 = C.f389().t23
    g, nrm, valid = seg(j['w'])
    vol, ok = C.render(t23, g, nrm, valid, j['ra'], j['rb'], j['ca'], j['cb'], T=C.T44)
    return j['id'], vol, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    assert np.array_equal(C.T44[J0], C.T32)
    T0 = time.time()
    meta = json.loads((C.DATA / 'prep.json').read_text())
    st = np.load(C.DATA / 'sites.npz')
    tr = np.load(C.DATA / 'train.npz')
    keys = meta['site_keys']
    n2 = 2 * C.TILE + C.PHASE
    jobs = [dict(id=('site', i), w=keys[i][0], ra=int(st['ra'][i]), rb=int(st['ra'][i]) + n2,
                 ca=int(st['ca'][i]), cb=int(st['ca'][i]) + n2) for i in range(meta['n_sites'])]
    jobs += [dict(id=('block', i), w=b['w'], ra=b['r'], rb=b['r'] + C.BLOCK, ca=b['c'], cb=b['c'] + C.BLOCK)
             for i, b in enumerate(meta['blocks'])]
    svol = np.zeros((meta['n_sites'], C.NPL_W, n2, n2), np.uint8)
    bvol = np.zeros((len(meta['blocks']), C.NPL_W, C.BLOCK, C.BLOCK), np.uint8)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (jid, vol, ok) in enumerate(ex.map(job, jobs, chunksize=4)):
            kind, i = jid
            if kind == 'site':
                assert np.array_equal(ok, st['ok'][i]), ('site ok', i)
                svol[i] = vol
            else:
                assert np.array_equal(ok, tr['ok'][i]), ('block ok', i)
                bvol[i] = vol
            if n % 50 == 0:
                print(f'  rendered {n + 1}/{len(jobs)} {time.time() - T0:.0f} s', flush=True)
    same_s = bool(np.array_equal(svol[:, J0], st['vol']))
    same_b = bool(np.array_equal(bvol[:, J0], tr['vol']))
    assert same_s and same_b, f'offset-0 window != attempt-1 render (sites {same_s}, blocks {same_b})'
    np.savez(C.DATA / 'sites44.npz', vol=svol)
    np.savez(C.DATA / 'train44.npz', vol=bvol)
    info = dict(planes=C.T44.tolist(), n_sites=len(svol), n_blocks=len(bvol), offset0_equals_attempt1=True,
                seconds=round(time.time() - T0, 1))
    (C.DATA / 'prep3.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
