# Third-party code

`shared/pr2368_band.py` is `band.py` from ScrollPrize/villa pull request #2368 (head 87223503), copied byte for byte
(`shared/run_t23.py` checks its sha256). villa is released under the MIT licence:

```
MIT License

Copyright (c) 2024 Vesuvius Challenge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The w028 release asset (see `PROVENANCE.md`) holds stored outputs of hecate (https://huggingface.co/scrollprize/hecate,
MIT licence) at the scored points; no hecate weights or code are in it.

## 2.4 µm release (`l0/`, weights and stored outputs on https://huggingface.co/j6sp5r/vesuvius-label-j): inputs and their licences

Checked 2026-11-13. No input forbids the release; the non-commercial term of the data carries over to the weights
and the stored outputs.

| input | source | licence / terms | how it is used |
|---|---|---|---|
| PHerc. 1667 CT, scan `20251217075048`, level 0 | `https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr` | Vesuvius Challenge open data, CC BY-NC 4.0 | read over https; 20 CT tiles (uint8, 176 × 400 × 400) at the stored replay spots are in `label-j-1667-l0-stored.tar` |
| PHerc. 1667 segment meshes w013, w018, w023, w028, w029, w031 (level 0) | `.../PHerc1667/segments/<seg>/mesh/<id>-on-20251217075048-2.399um.tifxyz` | as above | read over https; not redistributed |
| PHerc. 1667 ink labels, release `20260918`, level 0 | `.../PHerc1667/segments/<seg>/ink-labels/2.399um-volume-20251217075048/20260918/` | as above | the 13 nets were trained on them; spot lists and per-spot arrays derived from them are in the stored asset |
| hecate (public 3D ink model) | https://huggingface.co/scrollprize/hecate | MIT (Copyright 2026 Vesuvius Challenge), text below | positive control only: its stored outputs at the scored spots (`points.npz`, `p_hecate_fwd`) are in the stored asset; no hecate code or weights |
| villa `band.py` (PR #2368) | https://github.com/ScrollPrize/villa | MIT, text above | unchanged copy `shared/pr2368_band.py` (sheet finder) |
| the 13 label-J / label-D nets | this repository's code, trained from scratch | released by us under **CC BY-NC 4.0** | `label-j-1667-l0-weights.tar` |

No PHerc. Paris 4 or PHerc. 0139 data went into the 2.4 µm weights. No pre-trained model is a base of any of them
(`l0/f485/step1/train.py` builds the 3D U-Net from random initialisation).

### hecate licence text

```
MIT License

Copyright (c) 2026 Vesuvius Challenge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
