"""VES-2241 shared constants and the along-normal renderer (F-389 frame: 9.362 um tifxyz, grid normals)."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root; data defaults live under ROOT / 'data'
F389 = ROOT / 'shared'
DATA = Path(os.environ.get('RUN_0139', ROOT / 'data/run0139'))
F395_POINTS = Path(os.environ.get('POINTS_0139', ROOT / 'data/points0139')) / 'points.npz'    # F-395 scored run, 957b1d63
F395_POINTS_SHA = '8987c20a9be633828d9de6bede487f6f2c1d7ac7341f953b0fd700e4afe88cb7'

NPL = 32
T32 = np.arange(NPL) - 15.5                 # render / net planes, own vx along the grid normal
SCORE_K = np.arange(8, 24)                  # planes t = -7.5 .. 7.5 = F-389's 16 planes (T32[8:24] == run_t4.T_PL)
ARMS = {'S': np.abs(T32) <= 1.5,            # shell: 4 planes, no loss elsewhere
        'D': np.abs(T32) <= 7.5}            # depth extent: the 16-plane slab
# attempt 2 (F-396): arm P = S' = slab supervised, label value on the shell, target 0 on the rest of the slab
SUP = dict(ARMS, P=np.abs(T32) <= 7.5)      # supervised planes per arm
TGT = {'S': np.ones(NPL, bool), 'D': np.ones(NPL, bool),
       'P': np.abs(T32) <= 1.5}             # planes that carry the 2D label value; target 0 on other supervised planes
# attempt 3 (F-397): arm J = P with frame jitter: the 32-plane window is cut from a wider render at a random mesh
# offset j in [-JIT, JIT] vx per patch; supervised / label planes follow the mesh (|T32 + j| rules), not the frame
SUP['J'], TGT['J'] = SUP['P'], TGT['P']
ATTEMPT_ARMS = {1: ('S', 'D'), 2: ('P', 'D'), 3: ('J', 'D')}   # (informative arm, comparison arm); D = attempt-1 runs
JIT = 6                                     # training frame jitter, +-6 vx (integer planes)
NPL_W = NPL + 2 * JIT                       # wide render: 44 planes, t = -21.5 .. 21.5
T44 = np.arange(NPL_W) - (NPL_W - 1) / 2
SHIFT_K = (-4, 4)                           # frame-shift control: window rendered at mesh offset k (mesh t = T32 + k)
SEARCH = np.abs(T32) <= 10                  # frame planes searched for the peak (covers 15.5 +- 4 with margin)


def masks(arm, j=0):
    """(supervised, label-carrying) planes of the 32-plane window whose mesh t is T32 + j. The rules are those of
    SUP / TGT, written in mesh t; j = 0 gives SUP[arm], TGT[arm] exactly."""
    t = T32 + j
    sup = {'S': np.abs(t) <= 1.5, 'D': np.abs(t) <= 7.5, 'P': np.abs(t) <= 7.5, 'J': np.abs(t) <= 7.5}[arm]
    tgt = np.abs(t) <= 1.5 if arm in ('P', 'J') else np.ones(NPL, bool)
    return sup, tgt


for _a in SUP:
    assert all(np.array_equal(x, y) for x, y in zip(masks(_a), (SUP[_a], TGT[_a]))), _a
SEEDS = (0, 1)
TILE = 48                                   # site tile half size (96 x 96), as run_t4
PHASE = 4                                   # second grid phase: window shifted by 4 px (half a level-3 cell)
EXCL = 48                                   # no training label within +-48 px of a site centre (33 px box + 32 px)
BLOCK = 256
N_BLOCKS = 48                               # training blocks per segment
MIN_TRAIN_FRAC = 0.10                       # block kept if >= 10 % of its px are trainable labelled px


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def f389():
    """run_t4 (and run_t23 as .t23), imported unchanged."""
    if 'ves2199_run_t4' in sys.modules:
        return sys.modules['ves2199_run_t4']
    return load('ves2199_run_t4', F389 / 'run_t4.py')


def render(t23, g, nrm, v_pix, ra, rb, ca, cb, T=T32):
    """CT u8 at canvas px rows ra:rb, cols ca:cb (clipped to the canvas, zero-padded outside) on the planes T
    (default the 32 planes; attempt 3: T44). Same sampling as run_t4.render_tile (trilinear, cval 0);
    returns (vol (len(T), h, w) u8, ok (h, w) bool)."""
    H, W = v_pix.shape
    h, w = rb - ra, cb - ca
    npl = len(T)
    vol = np.zeros((npl, h, w), np.uint8)
    okf = np.zeros((h, w), bool)
    a0, a1, b0, b1 = max(ra, 0), min(rb, H), max(ca, 0), min(cb, W)
    if a1 <= a0 or b1 <= b0:
        return vol, okf
    rr, cc = np.mgrid[a0:a1, b0:b1]
    P, Nn = t23.at_pixels(g, nrm, rr.ravel(), cc.ravel())
    ok = v_pix[a0:a1, b0:b1].ravel() & np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
    if ok.any():
        q = P[None, ok] + T[:, None, None] * Nn[None, ok]                   # (npl, n_ok, xyz)
        qq = q.reshape(-1, 3)
        V = t23.vol(t23.VOL9)
        lo = np.maximum(np.floor(qq.min(0)).astype(int) - 2, 0)
        hi = np.minimum(np.ceil(qq.max(0)).astype(int) + 3, np.array(V.shape[::-1]))
        sub = t23.read_box(t23.VOL9, lo, hi)
        coords = np.stack([q[..., 2] - lo[2], q[..., 1] - lo[1], q[..., 0] - lo[0]]).reshape(3, -1)
        val = ndimage.map_coordinates(sub, coords, order=1, mode='constant', cval=0.0).reshape(npl, -1)
        full = np.zeros((npl, ok.size), np.float32)
        full[:, ok] = val
        vol[:, a0 - ra:a1 - ra, b0 - ca:b1 - ca] = np.clip(np.rint(full), 0, 255).astype(np.uint8).reshape(
            npl, a1 - a0, b1 - b0)
    okf[a0 - ra:a1 - ra, b0 - ca:b1 - ca] = ok.reshape(a1 - a0, b1 - b0)
    return vol, okf


def sites(t23, w, wi, pos, neg, valid):
    """The F-389 / F-395 label and N1 site draws of one segment, in F-395's order (label first, then N1)."""
    out = []
    for si, (sname, m) in enumerate((('label', pos), ('N1_nonlabel', neg))):
        rng = np.random.default_rng([t23.SEED, wi, si])
        for k, (r0, c0, rr, cc) in enumerate(t23.draw_sites(m, valid, rng, t23.N_SITES)):
            if len(rr):
                out.append(dict(key=[w, sname, k, int(r0), int(c0)], rr=rr, cc=cc))
    return out
