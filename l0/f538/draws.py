"""VES-3199 (F-538): the fixed 64-patch fit draws, shared by check_draws.py (CPU, before any score) and score.py (GPU).

A fit draw is exactly the fit check of F-485 / F-526 (`../2026-11-04-ves2890-label-j-l0-a3s0/train_a3.py:97` fit_auc
== `../f485/step1/train.py:242-247`): frng = default_rng(22410), T.batch(frng, D, centres,
ARMS['S'], N_FIT = 64), offset-0 window (z0 = JIT), shell planes, centres = argwhere(D['sup']) of the kept blocks.
Sets (`kept` = blocks whose segment is not the fold; `all6` = every block, train.data()):
  all6  -> P0's own fit patches (F-485 train.py)
  w029  -> P5 w029's fit patches (F-526)
  w028  -> P5 w028's fit patches (F-502)
The patch positions, rotations, labels and masks depend only on the rng and the kept blocks, not on the input
normalisation, so a set can be drawn with any net's (mean, std): `norm` overrides D['mu'], D['sd'] only.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-04-ves2890-label-j-l0-a3s0'))
import train_a3 as A  # noqa: E402,F401
from train_a3 import C, T, TF  # noqa: E402

FRNG_SEED = 22410
SETS = ('all6', 'w029', 'w028')


def h(*arrs):
    m = hashlib.blake2b(digest_size=16)
    for x in arrs:
        m.update(np.ascontiguousarray(x).tobytes())
    return m.hexdigest()


def data(name):
    """(D, mu, sd, kept segment per kept block) of a set, as the training script of its net built it."""
    if name == 'all6':
        D, mu, sd, meta = T.data()
        return D, mu, sd, [b['w'] for b in meta['blocks']]
    D, mu, sd, meta, kept_w = TF.fold_data(name)
    return D, mu, sd, kept_w


def draw(D, norm=None):
    """The 64-patch fit draw of D (X, Y, M); norm = (mean, std) replaces the set's own normalisation."""
    D0 = dict(D, z0=C.JIT)
    if norm is not None:
        D0['mu'], D0['sd'] = norm
    centres = np.argwhere(D0['sup'])
    return T.batch(np.random.default_rng(FRNG_SEED), D0, centres, C.ARMS['S'], T.N_FIT)


def patch_blocks(D, kept_w):
    """Segment of each of the 64 drawn patches: replays T.batch's rng calls (integers(len), integers(4), integers(2)
    per patch; no jitter rng), descriptive only."""
    centres = np.argwhere(D['sup'])
    rng = np.random.default_rng(FRNG_SEED)
    segs = []
    for _ in range(T.N_FIT):
        b = centres[rng.integers(len(centres))][0]
        rng.integers(4); rng.integers(2)
        segs.append(kept_w[b])
    return segs
