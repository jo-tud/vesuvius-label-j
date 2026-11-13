"""VES-2578 step 1 attempt 2, prefetch check (VES-2741 grant term 2, CPU, before any attempt-2 training): the batch
stream of train.batches() with prefetch must be bit-identical to the plain loop. For P0 (train.data(), all blocks,
arm P, seed 0) and one fold net (heldout/train_fold.fold_data('w028'), arm J, seed 0, the jitter rng as well), the
first --steps batches are drawn twice from fresh rngs, plain and prefetched; every batch's X, Y, M bytes are hashed
(blake2b) and the jitter offsets compared. PASS = every hash and the offset lists equal for both nets.

  python prefetch_check.py [--steps 200]
Writes DATA/prefetch_check.json and step1/prefetch_check.json.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'heldout')]
import common as C  # noqa: E402
import train as T  # noqa: E402
import train_fold as TF  # noqa: E402


def stream(D, arm, seed, steps, prefetch):
    D = dict(D)
    D.pop('js', None)
    centres = np.argwhere(D['sup'])
    jrng = np.random.default_rng([2241, seed, 3]) if arm == 'J' else None   # as train.main / train_fold.main
    rng = np.random.default_rng([2241, seed])
    t, hs = time.time(), []
    for b in T.batches(rng, D, centres, C.SUP[arm], C.TGT[arm], jrng, arm, steps, prefetch):
        h = hashlib.blake2b(digest_size=16)
        for x in b:
            h.update(np.ascontiguousarray(x).tobytes())
        hs.append(h.hexdigest())
    return hs, D.get('js', []), round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    a = ap.parse_args()
    out = dict(steps=a.steps, patch=T.PATCH, batch=T.BATCH, nets={})
    for name, arm, seed, data in (('P0', 'P', 0, lambda: T.data()[0]),
                                  ('J0_hw028', 'J', 0, lambda: TF.fold_data('w028')[0])):
        D = data()
        h0, j0, s0 = stream(D, arm, seed, a.steps, False)
        h1, j1, s1 = stream(D, arm, seed, a.steps, True)
        r = dict(arm=arm, seed=seed, same_hashes=h0 == h1, n_equal=sum(x == y for x, y in zip(h0, h1)),
                 same_jitter=j0 == j1, n_jitter=len(j0), seconds_plain=s0, seconds_prefetch=s1,
                 first_hash=h0[0], last_hash=h0[-1])
        r['pass'] = bool(r['same_hashes'] and r['same_jitter'] and len(h0) == len(h1) == a.steps)
        out['nets'][name] = r
        print(name, json.dumps(r), flush=True)
    out['pass'] = all(r['pass'] for r in out['nets'].values())
    for p in (C.DATA / 'prefetch_check.json', HERE / 'prefetch_check.json'):
        p.write_text(json.dumps(out, indent=1))
    print('PASS' if out['pass'] else 'FAIL', flush=True)
    sys.exit(0 if out['pass'] else 1)


if __name__ == '__main__':
    main()
