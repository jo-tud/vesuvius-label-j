"""VES-2257 (VES-2241 held-out confirmation of F-399): train arm J or D for one (fold, seed) on the training blocks of
the four other dev segments, then write the site outputs and the frame-shift outputs. The loop is train.main's
(same net, steps, batch, lr, augmentation, patch rng [2241, seed], jitter rng [2241, seed, 3], fit check); the only
changes: (1) the blocks of the held-out segment are dropped, (2) the input normalisation (mean, std) is computed on
the kept blocks only (prep.py's formula), (3) both arms read the 44-plane render (D at the offset-0 window, which
attempt3/prep3.py asserted equals train.npz byte for byte).

  python heldout/train_fold.py --fold w035 --arm J|D --seed 0|1 [--steps 6000] [--device cuda]
Writes DATA/heldout/out_<arm><seed>_h<fold>.npz (all 400 sites, as train.py; only the fold's sites are scored),
out_<arm><seed>_h<fold>_shift.npz (train.frame_outputs) and train_<arm><seed>_h<fold>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
import train as T  # noqa: E402

OUT = C.DATA / 'heldout'


def fold_data(fold):
    """Kept blocks (w != fold) of the 44-plane render, normalised with their own mean / std (offset-0 planes, ok px)."""
    meta = json.loads((C.DATA / 'prep.json').read_text())
    keep = np.array([b['w'] != fold for b in meta['blocks']])
    assert fold in {b['w'] for b in meta['blocks']} and keep.any()
    tr = np.load(C.DATA / 'train.npz')
    vol = np.load(C.DATA / 'train44.npz')['vol'][keep]
    ok = tr['ok'][keep]
    v0 = vol[:, C.JIT:C.JIT + C.NPL].transpose(1, 0, 2, 3)[:, ok].astype(np.float64)    # prep.py's norm formula
    mu, sd = float(v0.mean()), float(v0.std())
    del v0
    D = dict(x=((vol.astype(np.float32) - mu) / sd) * ok[:, None], pos=tr['pos'][keep], sup=tr['sup'][keep], z0=C.JIT)
    return D, mu, sd, meta, [b['w'] for b, k in zip(meta['blocks'], keep) if k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--arm', choices=('J', 'D'), required=True)
    ap.add_argument('--seed', type=int, choices=list(C.SEEDS), required=True)
    ap.add_argument('--steps', type=int, default=T.STEPS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    name = f'{a.arm}{a.seed}_h{a.fold}'
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    D, mu, sd, meta, kept_w = fold_data(a.fold)
    centres = np.argwhere(D['sup'])
    planes, tgt = C.SUP[a.arm], C.TGT[a.arm]
    jit = a.arm == 'J'
    jrng = np.random.default_rng([2241, a.seed, 3]) if jit else None
    net = T.UNet3D().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=T.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=T.LR, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng([2241, a.seed])
    T0, log = time.time(), []
    net.train()
    for step in range(a.steps):
        X, Y, M = (torch.from_numpy(t).to(dev) for t in T.batch(rng, D, centres, planes, T.BATCH, tgt, jrng, a.arm))
        lo = net(X)
        loss = (F.binary_cross_entropy_with_logits(lo, Y, reduction='none') * M).sum() / M.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            log.append([step, float(loss), round(time.time() - T0, 1)])
            print(f'{name} step {step} loss {float(loss):.4f} {time.time() - T0:.0f} s', flush=True)
    net.eval()
    frng = np.random.default_rng(22410)                                   # fit check, as train.main (offset-0 window)
    D0 = dict(D, z0=C.JIT)
    X, Y, M = T.batch(frng, D0, centres, C.ARMS['S'], T.N_FIT)
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(X).to(dev))).cpu().numpy()
    m = M > 0
    fit_auc = T.auc(P[m & (Y > 0.5)], P[m & (Y < 0.5)])
    st = np.load(C.DATA / 'sites.npz')
    out = {}
    with torch.no_grad():
        for i in range(meta['n_sites']):
            rr, cc = st[f'rr_{i}'], st[f'cc_{i}']
            ph = []
            for off in (0, C.PHASE):
                x = torch.from_numpy(T.site_input(st['vol'][i], st['ok'][i], 0, off, mu, sd)[None, None]).to(dev)
                p = torch.sigmoid(net(x))[0, 0].cpu().numpy()
                ph.append(p[C.SCORE_K][:, rr - st['ra'][i] - off, cc - st['ca'][i] - off].T)
            out[f'p_{i}'] = np.stack(ph).astype(np.float32)
    np.savez_compressed(OUT / f'out_{name}.npz', **out)
    torch.save(net.state_dict(), OUT / f'net_{name}.pt')
    data_dir = C.DATA
    C.DATA = OUT                                                          # frame_outputs writes to C.DATA
    try:
        for f in ('sites.npz', 'sites44.npz'):
            if not (OUT / f).exists():
                (OUT / f).symlink_to(data_dir / f)
        T.frame_outputs(net, dev, mu, sd, meta, name)
    finally:
        C.DATA = data_dir
    info = dict(fold=a.fold, arm=a.arm, seed=a.seed, steps=a.steps, batch=T.BATCH, patch=T.PATCH, lr=T.LR,
                widths=T.WIDTHS, norm=dict(mean=mu, std=sd), kept_blocks=len(kept_w),
                kept_segments=sorted(set(kept_w)), n_params=sum(p.numel() for p in net.parameters()),
                fit_auc_shell=fit_auc, loss_log=log, seconds=round(time.time() - T0, 1), device=str(dev),
                jitter=dict(range=C.JIT, n=len(D.get('js', [])),
                            counts=np.bincount(np.array(D.get('js', []), int) + C.JIT, minlength=2 * C.JIT + 1).tolist()),
                gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None)
    (OUT / f'train_{name}.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'loss_log'}), flush=True)


if __name__ == '__main__':
    main()
