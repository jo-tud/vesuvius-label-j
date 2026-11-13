"""VES-2241: train the fixed 3D U-Net for one (arm, seed). Only the target's depth extent (common.ARMS) differs
between arms; data, net, schedule, augmentation and seeds are the same.
VES-2578 step 1 (level 0), changes against F-418's train.py:
- patch (128, 32, 32) by the frozen patch rule (../bench_gpu.json), 12 000 steps (F-418 attempt 2), batch 16;
  attempt 2 (VES-2741): patch (128, 128, 128) (pilot PASS) and the batch stream cut one step ahead in a worker thread
  (batches(); same stream, prefetch_check.py);
- input: the 176-plane u8 blocks (prep.py, memory-mapped), normalised per patch as it is cut ((u8 - mean) / std x ok,
  the F-418 formula; a float32 copy of the blocks would be ~120 GB); non-jitter arms use the offset-0 window
  (planes JIT .. JIT + 127, == F-418's offset-0 window);
- site outputs are not written here (attempt4/infer4.py renders the tiles and runs every net); the trained net
  stores its outputs on its replay sites (prep.py replay.npz) for the net replay gate. outputs() (moved here from
  attempt4/infer4.py) is the one function that turns a 176-plane tile into site and frame-shift outputs.

  python train.py --arm P --seed 0|1 [--steps 12000] [--device cuda]    (P: known-artefact control, all six segments)
Writes DATA/net_<arm><seed>.pt, DATA/out_<arm><seed>_replay.npz, DATA/train_<arm><seed>.json.
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C

STEPS, BATCH, PATCH, LR = 12000, 16, 128, 1e-3   # attempt 2: patch 128 (pilot PASS, README); attempt 1 used 32
PATCH = int(os.environ.get('VES2578_PATCH', PATCH))   # pilot override (run_pilot_patch.sh)
WIDTHS = (16, 32, 64, 128)
N_FIT = 64                                  # fit check: training patches, fixed seed


class Block(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True),
                         nn.Conv3d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class UNet3D(nn.Module):
    """3D in (1, 128, h, w), 3D out (1, 128, h, w) logits; h, w divisible by 8."""

    def __init__(self, w=WIDTHS):
        super().__init__()
        self.enc = nn.ModuleList([Block(1, w[0])] + [Block(w[k - 1], w[k]) for k in range(1, len(w))])
        self.dec = nn.ModuleList([Block(w[k] + w[k - 1], w[k - 1]) for k in range(len(w) - 1, 0, -1)])
        self.head = nn.Conv3d(w[0], 1, 1)

    def forward(self, x):
        skips = []
        for k, e in enumerate(self.enc):
            x = e(x if k == 0 else F.max_pool3d(x, 2))
            skips.append(x)
        for d, s in zip(self.dec, reversed(skips[:-1])):
            x = d(torch.cat([F.interpolate(x, size=s.shape[2:], mode='trilinear', align_corners=False), s], 1))
        return self.head(x)


def data(keep=None):
    """Training data of the kept blocks (default all): D for batch(), with the prep.py norm of the kept blocks."""
    import prep
    meta = json.loads((C.DATA / 'prep.json').read_text())
    nb = len(meta['blocks'])
    keep = np.ones(nb, bool) if keep is None else keep
    mu, sd = prep.norm_of([b for b, k in zip(meta['blocks'], keep) if k])
    bidx = np.flatnonzero(keep)
    D = dict(u8=np.load(C.DATA / 'train176.npy', mmap_mode='r'), bidx=bidx, mu=mu, sd=sd, z0=C.JIT,
             **{k: np.load(C.DATA / f'train_{k}.npy')[bidx] for k in ('ok', 'pos', 'sup')})
    assert D['u8'].shape == (nb, C.NPL_W, C.BLOCK, C.BLOCK)
    return D, mu, sd, meta


def batch(rng, D, centres, arm_planes, n, tgt_planes=None, jrng=None, arm=None):
    """n patches (128, 32, 32) centred on trainable labelled px; random in-plane rot90 / flip (never along t).
    jrng (attempt 3, arm J): per patch a mesh offset j in [-JIT, JIT] from its own rng (the patch stream of rng is
    unchanged); the window is planes JIT + j .. JIT + j + 127 of the 176-plane render, i.e. mesh t = TWIN + j, and the
    supervised / label planes are C.masks(arm, j), so the label stays on the mesh while the frame moves.
    Input = (u8 - mean) / std x ok of the patch (float32), as F-418's precomputed array."""
    X = np.zeros((n, 1, C.NPL, PATCH, PATCH), np.float32)
    Y = np.zeros((n, 1, C.NPL, PATCH, PATCH), np.float32)
    M = np.zeros((n, 1, C.NPL, PATCH, PATCH), np.float32)
    h = PATCH // 2
    tgt0 = np.ones(C.NPL, bool) if tgt_planes is None else tgt_planes
    js = []
    for i in range(n):
        b, r, c = centres[rng.integers(len(centres))]
        r = int(np.clip(r - h, 0, C.BLOCK - PATCH)); c = int(np.clip(c - h, 0, C.BLOCK - PATCH))
        if jrng is None:
            z0, pl, tgt = D['z0'], arm_planes, tgt0
        else:
            j = int(jrng.integers(-C.JIT, C.JIT + 1))
            js.append(j)
            z0, (pl, tgt) = C.JIT + j, C.masks(arm, j)
        okp = D['ok'][b, r:r + PATCH, c:c + PATCH]
        v = ((D['u8'][D['bidx'][b], z0:z0 + C.NPL, r:r + PATCH, c:c + PATCH].astype(np.float32) - D['mu'])
             / D['sd']) * okp[None]
        y = D['pos'][b, r:r + PATCH, c:c + PATCH].astype(np.float32)
        s = D['sup'][b, r:r + PATCH, c:c + PATCH]
        k, fl = int(rng.integers(4)), bool(rng.integers(2))
        v, y, s = (np.rot90(a, k, axes=(-2, -1)) for a in (v, y, s))
        if fl:
            v, y, s = (a[..., ::-1] for a in (v, y, s))
        X[i, 0], Y[i, 0] = v, y[None] * tgt[:, None, None]
        M[i, 0] = s[None] & pl[:, None, None]
    if jrng is not None:
        D.setdefault('js', []).extend(js)
    return X, Y, M


class _Prefetch:
    """One producer thread calls fn() n times and hands the results over a FIFO queue (depth 2). One producer and the
    same rng objects, so the batch stream is the one the plain loop draws (CPU check: prefetch_check.py)."""

    def __init__(self, fn, n, depth=2):
        self.q, self.n = queue.Queue(depth), n
        self.t = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self.t.start()

    def _run(self, fn):
        try:
            for _ in range(self.n):
                self.q.put(fn())
        except BaseException as e:                                          # re-raised in the training loop
            self.q.put(e)

    def __iter__(self):
        for _ in range(self.n):
            b = self.q.get()
            if isinstance(b, BaseException):
                raise b
            yield b
        self.t.join()


def batches(rng, D, centres, planes, tgt, jrng, arm, steps, prefetch=True):
    """The training loop's batch stream: steps x batch(rng, D, centres, planes, BATCH, tgt, jrng, arm), in order.
    prefetch (attempt 2, VES-2741): the next batches are cut in a worker thread while the GPU trains on this one."""
    fn = lambda: batch(rng, D, centres, planes, BATCH, tgt, jrng, arm)        # noqa: E731
    if prefetch:
        return iter(_Prefetch(fn, steps))
    return (fn() for _ in range(steps))


def auc(pos, neg):
    from scipy import stats
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def site_input(vol, ok, z0, off, mu, sd):
    """Normalised (128, 384, 384) net input of one site tile: planes z0 .. z0 + 127, grid phase off."""
    v = vol[z0:z0 + C.NPL, off:off + 2 * C.TILE, off:off + 2 * C.TILE].astype(np.float32)
    return (v - mu) / sd * ok[off:off + 2 * C.TILE, off:off + 2 * C.TILE]


def outputs(net, dev, vol176, ok, ra, ca, rr, cc, mu, sd, ks=(C.SHIFT_K[0], 0, C.SHIFT_K[1])):
    """(site output (2, n, 64) float32 or None, {k: shift output (n, 128) float16}) of one site tile (176 planes).
    Window at mesh offset k = planes JIT + k .. JIT + k + 127 (mesh t = TWIN + k), both grid phases; the site output
    is the k = 0 window's 64 slab planes (SCORE_K) per phase, the shift output the mean of the 2 phases on all 128
    planes (F-418 train.frame_outputs / attempt4/infer4.outputs). One window per forward (GPU memory)."""
    sh, site = {}, None
    with torch.no_grad():
        for k in ks:
            ph = []
            for off in (0, C.PHASE):
                x = torch.from_numpy(site_input(vol176, ok, C.JIT + k, off, mu, sd)[None, None]).to(dev)
                p = torch.sigmoid(net(x))[0, 0].cpu().numpy()
                ph.append(p[:, rr - ra - off, cc - ca - off].T)                     # (n, 128)
            sh[k] = np.mean(ph, 0).astype(np.float16)
            if k == 0:
                site = np.stack([q[:, C.SCORE_K] for q in ph]).astype(np.float32)
    return site, sh


def const_output(net, dev):
    """Constant-input reading: mean p by plane for an all-zero normalised input (no CT)."""
    with torch.no_grad():
        z = torch.zeros((1, 1, C.NPL, 2 * C.TILE, 2 * C.TILE), device=dev)
        return torch.sigmoid(net(z))[0, 0].mean((1, 2)).cpu().numpy().astype(np.float32)


def replay_outputs(net, dev, mu, sd, w, path):
    """Outputs of the trained net (in memory) on the replay sites of segment w -> path (keys p_<u>, p<k>_<u>)."""
    rp = np.load(C.DATA / 'replay.npz')
    keys = json.loads((C.DATA / 'prep.json').read_text())['site_keys']
    out = {}
    for i, u in enumerate(rp['u']):
        if keys[u][0] != w:
            continue
        site, sh = outputs(net, dev, rp['vol'][i], rp['ok'][i], rp['ra'][i], rp['ca'][i], rp[f'rr_{i}'],
                           rp[f'cc_{i}'], mu, sd)
        out[f'p_{u}'] = site
        out.update({f'p{k}_{u}': v for k, v in sh.items()})
    assert out, f'no replay sites of {w}'
    np.savez_compressed(path, **out)
    return sorted(int(k[2:]) for k in out if k.startswith('p_'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(C.SUP), required=True)
    ap.add_argument('--seed', type=int, choices=list(C.SEEDS), required=True)
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--tag', default='')
    ap.add_argument('--no-prefetch', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    name = f'{a.arm}{a.seed}{a.tag}'
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    D, mu, sd, meta = data()
    assert C.SMOKE or (meta['limit_blocks'] is None and not meta['smoke']), 'smoke prep: set VES2578_SMOKE'
    jit = a.arm == 'J'
    centres = np.argwhere(D['sup'])
    planes, tgt = C.SUP[a.arm], C.TGT[a.arm]
    jrng = np.random.default_rng([2241, a.seed, 3]) if jit else None               # own stream: patches as arm P
    net = UNet3D().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng([2241, a.seed])           # same patch stream for both arms of a seed
    T0, log = time.time(), []
    net.train()
    for step, b in enumerate(batches(rng, D, centres, planes, tgt, jrng, a.arm, a.steps, not a.no_prefetch)):
        X, Y, M = (torch.from_numpy(t).to(dev) for t in b)
        lo = net(X)
        loss = (F.binary_cross_entropy_with_logits(lo, Y, reduction='none') * M).sum() / M.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 200 == 0 or step == a.steps - 1:
            log.append([step, float(loss), round(time.time() - T0, 1)])
            print(f'{name} step {step} loss {float(loss):.4f} {time.time() - T0:.0f} s', flush=True)
    t_train = time.time() - T0
    net.eval()
    # fit check on training patches (shell planes, common to both arms; fixed patch draw)
    frng = np.random.default_rng(22410)
    X, Y, M = batch(frng, D, centres, C.ARMS['S'], N_FIT)
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(X).to(dev))).cpu().numpy()
    m = M > 0
    fit_auc = auc(P[m & (Y > 0.5)], P[m & (Y < 0.5)])
    torch.save(net.state_dict(), C.DATA / f'net_{name}.pt')
    rep_u = replay_outputs(net, dev, mu, sd, C.f389().t23.DEV[0], C.DATA / f'out_{name}_replay.npz')
    info = dict(arm=a.arm, seed=a.seed, steps=a.steps, batch=BATCH, patch=PATCH, lr=LR, widths=WIDTHS,
                norm=dict(mean=mu, std=sd), n_params=sum(p.numel() for p in net.parameters()), fit_auc_shell=fit_auc,
                loss_log=log, seconds=round(time.time() - T0, 1), seconds_train=round(t_train, 1), device=str(dev),
                replay_sites=rep_u, limit_blocks=meta['limit_blocks'],
                jitter=dict(range=C.JIT, n=len(D.get('js', [])),
                            counts=np.bincount(np.array(D.get('js', []), int) + C.JIT, minlength=2 * C.JIT + 1).tolist()),
                gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    (C.DATA / f'train_{name}.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'loss_log'}), flush=True)


if __name__ == '__main__':
    main()
