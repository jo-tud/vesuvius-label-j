"""VES-2241 shared constants and the along-normal renderer (F-389 frame: 9.362 um tifxyz, grid normals).
VES-2321: run on PHerc1667 level 2 (9.596 um): f389() points run_t23 at the 1667 data (patch1667.apply), DATA and the
per-point file (F-395 points.npz on 0139 -> points_old.npz built by attempt4/sites4.py on 1667) change; nothing else.
VES-2578 step 1: the same on PHerc1667 LEVEL 0 (2.399 um), README "Level-0 recipe" / "Step 1 criteria": every length
x 4 (planes, shell, slab, jitter, shift, search, tile, phase, exclusion, block 768 by the frozen recipe). f389() is
the level-0 run_t4 instance (patchl0.apply); f418_l2() the unchanged F-418 level-2 instance, used only for the hecate
positive control on the level-2 render (as F-418). Data access: level-0 meshes data/pherc1667_meshes/<w>, level-0
labels and sites cached by ../step0_labels.py (DATA0/labels0, DATA0/sites0). render() works on the node validity
and samples in row strips (level-0 canvas masks are up to 4.3 G px); values equal F-418's render (tested byte for byte)."""
import importlib.util
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
F389 = HERE.parents[2] / 'shared'
F418 = HERE.parents[2] / 'pherc1667'
DATA0 = Path(os.environ.get('VES2578_DATA', _VD + '/ves2578'))    # step-0 caches
DATA = Path(os.environ.get('VES2578_RUN', str(DATA0 / 'step1')))
SMOKE = int(os.environ.get('VES2578_SMOKE', '0'))   # smoke test only: old-draw sites k < SMOKE (gates refuse it)
MESH0 = Path(os.environ.get('REPO_DATA', str(_ROOT / 'data'))) / 'pherc1667_meshes'                  # level-0 tifxyz
L2COPY = Path(os.environ.get('DATA_1667', str(_ROOT / 'data/pherc1667'))) / 'ves170'     # F-418's level-2 copy: hecate PC only
# The F-395 role (per-point use / pm / pp / dfn / p_hecate_fwd) is played by DATA/points.npz, written by
# attempt4/sites4.py for every step-0 site; its sha256 is written next to this file at build time and committed.
F395_POINTS = DATA / 'points.npz'
F395_SHA_FILE = HERE / 'points.sha256'
F395_POINTS_SHA = F395_SHA_FILE.read_text().split()[0] if F395_SHA_FILE.exists() else 'not-built'
F = 4                                       # level 2 -> level 0
STEP = 20.0                                 # canvas px per grid node (level-0 mesh): canvas px = L0 vx

NPL = 128
TWIN = np.arange(NPL) - 63.5                # render / net planes, L0 vx along the grid normal (F-418: T32, 32 planes)
SHELL, SLAB = 7.5, 31.5                     # F-418 1.5 / 7.5 on the half-integer grid, extent-matched x 4
SCORE_K = np.flatnonzero(np.abs(TWIN) <= SLAB)   # the 64 slab planes t = -31.5 .. 31.5 (== f389().T_PL, patchl0)
ARMS = {'S': np.abs(TWIN) <= SHELL,         # shell: 16 planes, no loss elsewhere
        'D': np.abs(TWIN) <= SLAB}          # depth extent: the 64-plane slab
# attempt 2 (F-396): arm P = S' = slab supervised, label value on the shell, target 0 on the rest of the slab
SUP = dict(ARMS, P=np.abs(TWIN) <= SLAB)    # supervised planes per arm
TGT = {'S': np.ones(NPL, bool), 'D': np.ones(NPL, bool),
       'P': np.abs(TWIN) <= SHELL}          # planes that carry the 2D label value; target 0 on other supervised planes
# attempt 3 (F-397): arm J = P with frame jitter: the 128-plane window is cut from a wider render at a random mesh
# offset j in [-JIT, JIT] vx per patch; supervised / label planes follow the mesh (|TWIN + j| rules), not the frame
SUP['J'], TGT['J'] = SUP['P'], TGT['P']
ATTEMPT_ARMS = {1: ('S', 'D'), 2: ('P', 'D'), 3: ('J', 'D')}   # (informative arm, comparison arm); D = attempt-1 runs
JIT = 24                                    # training frame jitter, +-24 L0 vx (F-418: 6)
NPL_W = NPL + 2 * JIT                       # wide render: 176 planes, t = -87.5 .. 87.5
TWIDE = np.arange(NPL_W) - (NPL_W - 1) / 2
SHIFT_K = (-16, 16)                         # frame-shift control: window rendered at mesh offset k (mesh t = TWIN + k)
SEARCH = np.abs(TWIN) <= 40                 # frame planes searched for the peak (F-418: |t| <= 10)


def masks(arm, j=0):
    """(supervised, label-carrying) planes of the 128-plane window whose mesh t is TWIN + j. The rules are those of
    SUP / TGT, written in mesh t; j = 0 gives SUP[arm], TGT[arm] exactly."""
    t = TWIN + j
    sup = {'S': np.abs(t) <= SHELL, 'D': np.abs(t) <= SLAB, 'P': np.abs(t) <= SLAB, 'J': np.abs(t) <= SLAB}[arm]
    tgt = np.abs(t) <= SHELL if arm in ('P', 'J') else np.ones(NPL, bool)
    return sup, tgt


for _a in SUP:
    assert all(np.array_equal(x, y) for x, y in zip(masks(_a), (SUP[_a], TGT[_a]))), _a
assert np.array_equal(TWIDE[JIT:JIT + NPL], TWIN) and len(SCORE_K) == 64 and SUP['J'].sum() == 64
assert TGT['J'].sum() == 16
SEEDS = (0, 1)
# The six 1667 folds in F-418's registered order (README "Step 1 criteria": folds and order as F-418).
FOLDS = ('w028', 'w029', 'w013', 'w018', 'w023', 'w031')
TILE = 192                                  # site tile half size (384 x 384), F-418 48 x 4
PHASE = 16                                  # second grid phase: window shifted by 16 px (F-418: 4)
EXCL = 192                                  # no training label within +-192 px of a site centre (F-418: 48)
BLOCK = 768                                 # training block (frozen recipe: 768, 0.56 x F-418's 1024 L0 px area)
N_BLOCKS = 48                               # training blocks per segment
MIN_TRAIN_FRAC = 0.10                       # block kept if >= 10 % of its px are trainable labelled px
SEED_BLOCKS = 2578                          # block draw rng [2578, wi] (F-418: [2241, wi])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def f389():
    """run_t4 (and run_t23 as .t23), level-0 instance (patchl0.apply: data, profile and scored-plane constants)."""
    if 'ves2578_run_t4' in sys.modules:
        return sys.modules['ves2578_run_t4']
    sys.path.insert(0, str(HERE))
    import patchl0                                                        # VES-2578: 1667 level-0 data
    t4 = load('ves2578_run_t4', F389 / 'run_t4.py')
    patchl0.apply(t4)
    return t4


def f418_l2():
    """F-418's level-2 run_t4 instance (patch1667.apply, unchanged): only for hecate's level-2 render_tile."""
    if 'ves2321_run_t4_l2' in sys.modules:
        return sys.modules['ves2321_run_t4_l2']
    load('prep_data', F418 / 'prep_data.py')                             # patch1667's `import prep_data`
    p1667 = load('patch1667', F418 / 'patch1667.py')
    t4 = load('ves2321_run_t4_l2', F389 / 'run_t4.py')
    p1667.apply(t4.t23)
    return t4


def segs():
    """{w: S3 segment dir}, F-418's prep_data.SEGS (the order is run_t23.DEV and step0_labels' segment index wi)."""
    if 'prep_data' not in sys.modules:
        load('prep_data', F418 / 'prep_data.py')
    return dict(sys.modules['prep_data'].SEGS)


def grid0(w):
    """Level-0 mesh grid, node validity, grid normals (run_t23 rules)."""
    t23 = f389().t23
    g, v = t23.load_grid(MESH0 / w)
    return g, v, t23.grid_normals(g, v)


def labels0(w):
    """Level-0 (pos, neg) of one segment from step0_labels.py's packbits cache (pos = ink & sup, neg = sup & ~ink)."""
    z = np.load(DATA0 / 'labels0' / f'{w}.npz')
    sh = tuple(int(x) for x in z['shape'])
    return tuple(np.unpackbits(z[k], count=sh[0] * sh[1]).reshape(sh).view(bool) for k in ('pos', 'neg'))


def canvas_shape(v):
    return (v.shape[0] * int(STEP), v.shape[1] * int(STEP))


def pixel_valid_box(v, ra, rb, ca, cb):
    """run_t23.pixel_valid(v, canvas_shape(v))[ra:rb, ca:cb] for 0 <= ra < rb <= H, 0 <= ca < cb <= W, without the
    full canvas (px p <-> node floor((p + 0.5) / STEP); valid if its 4 surrounding nodes are)."""
    vv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    r = np.floor((np.arange(ra, rb) + 0.5) / STEP).astype(int)
    c = np.floor((np.arange(ca, cb) + 0.5) / STEP).astype(int)
    ok_r, ok_c = r < vv.shape[0], c < vv.shape[1]
    out = np.zeros((rb - ra, cb - ca), bool)
    out[np.ix_(ok_r, ok_c)] = vv[np.ix_(r[ok_r], c[ok_c])]
    return out


STRIP = 64                                  # sampled rows per strip (RAM: 176 planes x 64 x 768 x 3 float64 = 208 MB)


def render(t23, g, nrm, v, ra, rb, ca, cb, T=TWIN):
    """CT u8 at canvas px rows ra:rb, cols ca:cb (clipped to the canvas, zero-padded outside) on the planes T
    (default the 128 planes; TWIDE: 176). Same sampling as run_t4.render_tile (trilinear, cval 0) on the level-0
    volume; returns (vol (len(T), h, w) u8, ok (h, w) bool). v = node validity (pixel_valid_box). One box read per
    call, as F-418 (lo / hi from the end planes: q is linear in t, so its extremes are at T[0], T[-1]); the trilinear
    sampling runs in row strips of STRIP px, which gives the same values as one map_coordinates call."""
    H, W = canvas_shape(v)
    h, w = rb - ra, cb - ca
    npl = len(T)
    vol = np.zeros((npl, h, w), np.uint8)
    okf = np.zeros((h, w), bool)
    a0, a1, b0, b1 = max(ra, 0), min(rb, H), max(ca, 0), min(cb, W)
    if a1 <= a0 or b1 <= b0:
        return vol, okf
    rr, cc = np.mgrid[a0:a1, b0:b1]
    P, Nn = t23.at_pixels(g, nrm, rr.ravel(), cc.ravel())
    ok = pixel_valid_box(v, a0, a1, b0, b1).ravel() & np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
    okf[a0 - ra:a1 - ra, b0 - ca:b1 - ca] = ok.reshape(a1 - a0, b1 - b0)
    if not ok.any():
        return vol, okf
    ends = np.concatenate([P[ok] + T[0] * Nn[ok], P[ok] + T[-1] * Nn[ok]])
    V = t23.vol(t23.VOL9)
    lo = np.maximum(np.floor(ends.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(ends.max(0)).astype(int) + 3, np.array(V.shape[::-1]))
    sub = t23.read_box(t23.VOL9, lo, hi)
    P, Nn, ok = (x.reshape(a1 - a0, b1 - b0, *x.shape[1:]) for x in (P, Nn, ok))
    for s0 in range(0, a1 - a0, STRIP):
        s1 = min(s0 + STRIP, a1 - a0)
        k = ok[s0:s1].ravel()
        if not k.any():
            continue
        q = P[s0:s1].reshape(-1, 3)[k][None] + T[:, None, None] * Nn[s0:s1].reshape(-1, 3)[k][None]   # (npl, n, 3)
        coords = np.stack([q[..., 2] - lo[2], q[..., 1] - lo[1], q[..., 0] - lo[0]]).reshape(3, -1)
        del q
        val = ndimage.map_coordinates(sub, coords, order=1, mode='constant', cval=0.0).reshape(npl, -1)
        full = np.zeros((npl, k.size), np.float32)
        full[:, k] = val
        vol[:, a0 - ra + s0:a0 - ra + s1, b0 - ca:b1 - ca] = np.clip(np.rint(full), 0, 255).astype(np.uint8).reshape(
            npl, s1 - s0, b1 - b0)
    return vol, okf


def sites0(w):
    """step0_labels.py's site draws of one segment, in its order (old label, old N1, new label, new N1):
    dict(key=[w, cls, k, r0, c0], kind, rr, cc); k < 1000 old draw, k >= 1000 spacing draw (README "Step 1 criteria":
    Sites). Points are box +-64 px, every 8th px, class & valid (rr, cc before the finite-geometry filter)."""
    z = np.load(DATA0 / 'sites0' / f'{w}.npz')
    meta = json.loads(str(z['meta']))
    return [dict(key=[w, m['cls'], int(m['k']), int(m['r0']), int(m['c0'])], kind=m['kind'],
                 rr=z[f'rr_{i}'], cc=z[f'cc_{i}']) for i, m in enumerate(meta) if not SMOKE or int(m['k']) < SMOKE]
