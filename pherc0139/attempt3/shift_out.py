"""VES-2241 attempt 3: frame-shift control outputs (train.frame_outputs) for an already trained net.
  P0, P1 (attempt 2, the known frame artefact F-397): the control's known-answer check (must be ARTEFACT, else VOID).
  D0, D1 (attempt 1): reading only.
  python attempt3/shift_out.py --name P0 [--device cuda]  ->  DATA/out_P0_shift.npz"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
from train import UNet3D, frame_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--device', default='cuda')
a = ap.parse_args()
dev = torch.device(a.device)
meta = json.loads((C.DATA / 'prep.json').read_text())
net = UNet3D().to(dev)
net.load_state_dict(torch.load(C.DATA / f'net_{a.name}.pt', map_location=dev))
net.eval()
frame_outputs(net, dev, meta['norm']['mean'], meta['norm']['std'], meta, a.name)
print('wrote', C.DATA / f'out_{a.name}_shift.npz', flush=True)
