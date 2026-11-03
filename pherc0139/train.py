"""VES-2241: train the fixed 3D U-Net for one (arm, seed) and write its site outputs. Only the target's depth
extent (common.ARMS) differs between arms; data, net, schedule, augmentation and seeds are the same.

  python train.py --arm S|D|P|J --seed 0|1   (P = attempt 2 arm S': label on the shell, 0 on the rest of the slab;
                                             J = attempt 3: P with +-6 vx frame jitter) [--steps 6000] [--device cuda]
Writes DATA/out_<arm><seed>.npz (per site: p at the 16 F-389 planes, both grid phases) and DATA/train_<arm><seed>.json;
arm J also DATA/out_J<seed>_shift.npz (frame-shift control outputs, frame_outputs).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C

STEPS, BATCH, PATCH, LR = 6000, 16, 64, 1e-3
WIDTHS = (16, 32, 64, 128)
N_FIT = 64                                  # fit check: training patches, fixed seed


class Block(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True),
                         nn.Conv3d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class UNet3D(nn.Module):
    """3D in (1, 32, h, w), 3D out (1, 32, h, w) logits; h, w divisible by 8."""

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


def batch(rng, D, centres, arm_planes, n, tgt_planes=None, jrng=None, arm=None):
    """n patches (32, 64, 64) centred on trainable labelled px; random in-plane rot90 / flip (never along t).
    jrng (attempt 3, arm J): per patch a mesh offset j in [-JIT, JIT] from its own rng (the patch stream of rng is
    unchanged); the window is planes JIT + j .. JIT + j + 31 of the 44-plane render, i.e. mesh t = T32 + j, and the
    supervised / label planes are C.masks(arm, j), so the label stays on the mesh while the frame moves."""
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
        v = D['x'][b, z0:z0 + C.NPL, r:r + PATCH, c:c + PATCH]
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


def auc(pos, neg):
    from scipy import stats
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def site_input(vol, ok, z0, off, mu, sd):
    """Normalised (32, 96, 96) net input of one site tile: planes z0 .. z0 + 31, grid phase off."""
    v = vol[z0:z0 + C.NPL, off:off + 2 * C.TILE, off:off + 2 * C.TILE].astype(np.float32)
    return (v - mu) / sd * ok[off:off + 2 * C.TILE, off:off + 2 * C.TILE]


def frame_outputs(net, dev, mu, sd, meta, name):
    """Attempt 3 frame-shift control outputs (README attempt3 "Criteria"): for k in (-4, 0, +4) the site windows cut
    from the 44-plane render at mesh offset k (mesh t = T32 + k); per site, p at all 32 window planes at the F-389
    points, mean of the 2 grid phases, float16 -> DATA/out_<name>_shift.npz keys p<k>_<i> (n, 32). Also the
    constant-input reading: mean p by plane for an all-zero normalised input (no CT), key const."""
    st = np.load(C.DATA / 'sites.npz')
    w44 = np.load(C.DATA / 'sites44.npz')['vol']
    out = {}
    with torch.no_grad():
        for k in (C.SHIFT_K[0], 0, C.SHIFT_K[1]):
            for i in range(meta['n_sites']):
                rr, cc = st[f'rr_{i}'], st[f'cc_{i}']
                ph = []
                for off in (0, C.PHASE):
                    x = torch.from_numpy(site_input(w44[i], st['ok'][i], C.JIT + k, off, mu, sd)[None, None])
                    p = torch.sigmoid(net(x.to(dev)))[0, 0].cpu().numpy()
                    ph.append(p[:, rr - st['ra'][i] - off, cc - st['ca'][i] - off].T)        # (n, 32)
                out[f'p{k}_{i}'] = np.mean(ph, 0).astype(np.float16)
        z = torch.zeros((1, 1, C.NPL, 2 * C.TILE, 2 * C.TILE), device=dev)
        out['const'] = torch.sigmoid(net(z))[0, 0].mean((1, 2)).cpu().numpy().astype(np.float32)
    np.savez_compressed(C.DATA / f'out_{name}_shift.npz', **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(C.SUP), required=True)
    ap.add_argument('--seed', type=int, choices=list(C.SEEDS), required=True)
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    name = f'{a.arm}{a.seed}{a.tag}'
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    meta = json.loads((C.DATA / 'prep.json').read_text())
    mu, sd = meta['norm']['mean'], meta['norm']['std']
    tr = np.load(C.DATA / 'train.npz')
    jit = a.arm == 'J'
    vol = np.load(C.DATA / 'train44.npz')['vol'] if jit else tr['vol']           # J: 44 planes (attempt3/prep3.py)
    D = dict(x=((vol.astype(np.float32) - mu) / sd) * tr['ok'][:, None], pos=tr['pos'], sup=tr['sup'],
             z0=C.JIT if jit else 0)                                                # z0: offset-0 window
    del vol
    centres = np.argwhere(D['sup'])
    planes, tgt = C.SUP[a.arm], C.TGT[a.arm]
    jrng = np.random.default_rng([2241, a.seed, 3]) if jit else None               # own stream: patches as arm P
    net = UNet3D().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng([2241, a.seed])           # same patch stream for both arms of a seed
    T0, log = time.time(), []
    net.train()
    for step in range(a.steps):
        X, Y, M = (torch.from_numpy(t).to(dev) for t in batch(rng, D, centres, planes, BATCH, tgt, jrng, a.arm))
        lo = net(X)
        loss = (F.binary_cross_entropy_with_logits(lo, Y, reduction='none') * M).sum() / M.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 200 == 0 or step == a.steps - 1:
            log.append([step, float(loss), round(time.time() - T0, 1)])
            print(f'{name} step {step} loss {float(loss):.4f} {time.time() - T0:.0f} s', flush=True)
    net.eval()
    # fit check on training patches (shell planes, common to both arms; fixed patch draw)
    frng = np.random.default_rng(22410)
    X, Y, M = batch(frng, D, centres, C.ARMS['S'], N_FIT)
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(X).to(dev))).cpu().numpy()
    m = M > 0
    fit_auc = auc(P[m & (Y > 0.5)], P[m & (Y < 0.5)])
    # site outputs: both grid phases, planes 8..23, at the F-389 scored points
    st = np.load(C.DATA / 'sites.npz')
    out = {}
    with torch.no_grad():
        for i in range(meta['n_sites']):
            rr, cc = st[f'rr_{i}'], st[f'cc_{i}']
            ph = []
            for off in (0, C.PHASE):
                x = torch.from_numpy(site_input(st['vol'][i], st['ok'][i], 0, off, mu, sd)[None, None]).to(dev)
                p = torch.sigmoid(net(x))[0, 0].cpu().numpy()
                ph.append(p[C.SCORE_K][:, rr - st['ra'][i] - off, cc - st['ca'][i] - off].T)     # (n, 16)
            out[f'p_{i}'] = np.stack(ph).astype(np.float32)                                     # (2, n, 16)
    np.savez_compressed(C.DATA / f'out_{name}.npz', **out)
    torch.save(net.state_dict(), C.DATA / f'net_{name}.pt')
    if jit:
        frame_outputs(net, dev, mu, sd, meta, name)
    info = dict(arm=a.arm, seed=a.seed, steps=a.steps, batch=BATCH, patch=PATCH, lr=LR, widths=WIDTHS,
                n_params=sum(p.numel() for p in net.parameters()), fit_auc_shell=fit_auc, loss_log=log,
                seconds=round(time.time() - T0, 1), device=str(dev),
                jitter=dict(range=C.JIT, n=len(D.get('js', [])),
                            counts=np.bincount(np.array(D.get('js', []), int) + C.JIT, minlength=2 * C.JIT + 1).tolist()),
                gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    (C.DATA / f'train_{name}.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'loss_log'}), flush=True)


if __name__ == '__main__':
    main()
