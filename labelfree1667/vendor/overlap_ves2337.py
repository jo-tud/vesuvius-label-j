"""VES-2337 step 0: do Will's six PHerc1667 patches overlap the six labelled segments of VES-2321?

All distances in the 7.91 um frame (scan 20231117161658), where Will's patches sit (bbox match).
Our six segments: 2.399 um meshes (VES-2321 input) mapped by the public 2.399 -> 7.91 affine
(`transform.json` of the 2.399 um zarr). The published 7.91 um meshes check that affine.
Labelled area = mesh points whose 5x5 label-px cell holds any pos/neg label (VES-2321 trains and scores
only there). No CT, no ink, no model.

usage: python -I overlap.py WILL_DIR MESH24_DIR MESH791_DIR LABEL_DIR TRANSFORM_JSON OUT_DIR
"""
import json
import sys
import time

import numpy as np
import tifffile
from scipy.spatial import cKDTree

SEGS = ["w013", "w018", "w023", "w028", "w029", "w031"]
VX_UM = 7.91
STRIDE = 4  # Will grid is 4 vx; stride 4 -> 16 vx between query points
WORKERS = 8


def load_xyz(d):
    xyz = np.stack([tifffile.imread(f"{d}/{a}.tif") for a in "xyz"], -1).astype(np.float32)
    valid = (xyz[..., 0] > 0) & np.isfinite(xyz).all(-1)
    return xyz, valid


def grid_normals(xyz, valid):
    """Unit normals from central differences; NaN where a neighbour is invalid."""
    p = np.where(valid[..., None], xyz, np.nan).astype(np.float64)
    du = np.full_like(p, np.nan)
    dv = np.full_like(p, np.nan)
    du[1:-1] = p[2:] - p[:-2]
    dv[:, 1:-1] = p[:, 2:] - p[:, :-2]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def cell_area(xyz, valid):
    """Area (vx^2) of the quad right/down of each node, 0 where any corner invalid."""
    a = np.zeros(valid.shape)
    p = xyz.astype(np.float64)
    ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    e1 = p[1:, :-1] - p[:-1, :-1]
    e2 = p[:-1, 1:] - p[:-1, :-1]
    e3 = p[1:, 1:] - p[1:, :-1]
    e4 = p[1:, 1:] - p[:-1, 1:]
    q = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=-1) + 0.5 * np.linalg.norm(np.cross(e3, e4), axis=-1)
    a[:-1, :-1] = np.where(ok, q, 0)
    return a


def apply_affine(m, pts):
    return pts @ m[:, :3].T + m[:, 3]


def perp_dist(tree, pts_ref, nrm_ref, q, inplane_max):
    """Nearest-point distance and normal distance (NaN if the nearest point's in-plane offset is too big)."""
    d, i = tree.query(q, k=1, workers=WORKERS)
    v = q - pts_ref[i]
    n = nrm_ref[i]
    dn = np.abs(np.einsum("ij,ij->i", v, n))
    ip = np.sqrt(np.maximum(d ** 2 - dn ** 2, 0))
    dp = np.where(np.isfinite(dn) & (ip <= inplane_max), dn, np.nan)
    return d, dp


def pitch_from_patch(xyz, valid, nrm, rng, n_query=20000, min_grid=15, inplane=10.0):
    """Wrap pitch (vx): normal offset to the nearest point of the same patch on another wrap.

    Another wrap = a point >= min_grid grid cells away on the patch grid, within `inplane` vx of the
    query's normal line. Inputs are the stride-STRIDE grids (16 vx). Returns per-query minimum offsets.
    """
    s = 1
    sub = xyz
    sv = valid
    rr, cc = np.nonzero(sv)
    pts = sub[rr, cc].astype(np.float64)
    tree = cKDTree(pts)
    qi = rng.choice(len(rr), size=min(n_query, len(rr)), replace=False)
    out = []
    for k in qi:
        r, c = rr[k] * s, cc[k] * s
        n = nrm[r, c]
        if not np.isfinite(n).all():
            continue
        p = xyz[r, c].astype(np.float64)
        idx = tree.query_ball_point(p, 120.0)
        if not idx:
            continue
        idx = np.asarray(idx)
        far = np.hypot(rr[idx] - r, cc[idx] - c) >= min_grid
        idx = idx[far]
        if not len(idx):
            continue
        v = pts[idx] - p
        dn = v @ n
        ip = np.linalg.norm(v - dn[:, None] * n, axis=1)
        m = ip <= inplane
        if m.any():
            out.append(np.abs(dn[m]).min())
    return np.asarray(out)


def main():
    will_dir, m24_dir, m791_dir, lab_dir, tf_json, out_dir = sys.argv[1:7]
    t0 = time.time()
    rng = np.random.default_rng(2337)
    m = np.asarray(json.load(open(tf_json))["transformation_matrix"], dtype=np.float64)
    res = {"frame": "PHerc1667 20231117161658 7.91 um voxels", "affine_2399_to_791": m.tolist(), "segs": {}, "will": {}}

    # --- our six segments, 2.399 um meshes mapped to 7.91 um; labelled subset; affine check
    ref = {}
    for w in SEGS:
        xyz, valid = load_xyz(f"{m24_dir}/{w}")
        lab = np.load(f"{lab_dir}/{w}.npz")
        sup = lab["pos"] | lab["neg"]
        H, W = valid.shape
        if sup.shape != (H * 5, W * 5):  # level-2 labels = 5 px per 20 vx node; refuse any other grid
            sys.exit(f"{w}: label shape {sup.shape} != 5 x mesh grid {(H * 5, W * 5)}; wrong mesh dir?")
        sup = sup.reshape(H, 5, W, 5).any((1, 3))
        p791 = np.full(xyz.shape, np.nan, np.float64)
        p791[valid] = apply_affine(m, xyz[valid].astype(np.float64))
        nrm = grid_normals(p791, valid)
        a24 = cell_area(xyz, valid) * 2.399 ** 2 / 1e8  # cm^2
        # affine check: published 7.91 um mesh vs mapped 2.399 mesh
        pxyz, pval = load_xyz(f"{m791_dir}/{w}")
        ok = valid & np.isfinite(nrm).all(-1)
        t_pub = cKDTree(pxyz[pval].astype(np.float64))
        sel = np.nonzero(ok.ravel())[0]
        sel = rng.choice(sel, size=min(20000, len(sel)), replace=False)
        q = p791.reshape(-1, 3)[sel]
        pn = grid_normals(pxyz.astype(np.float64), pval)
        dpub, dpub_perp = perp_dist(t_pub, pxyz[pval].astype(np.float64), pn[pval], q, 25.0)
        keep = valid & sup & np.isfinite(nrm).all(-1)
        if not keep.any():
            sys.exit(f"{w}: no labelled mesh node")
        allk = valid & np.isfinite(nrm).all(-1)
        ref[w] = {"lab_pts": p791[keep], "lab_nrm": nrm[keep], "all_pts": p791[allk], "all_nrm": nrm[allk]}
        res["segs"][w] = {
            "grid24": [H, W],
            "area_cm2_mesh": float(a24[valid].sum()),
            "area_cm2_labelled": float(a24[valid & sup].sum()),
            "z791_range_labelled": [float(np.nanmin(p791[keep][:, 2])), float(np.nanmax(p791[keep][:, 2]))],
            "affine_vs_published_791_normal_vx": {
                "median": float(np.nanmedian(dpub_perp)), "p90": float(np.nanpercentile(dpub_perp, 90)),
                "n_defined": int(np.isfinite(dpub_perp).sum())},
        }
        print(w, res["segs"][w], f"{time.time()-t0:.0f}s", flush=True)

    lab_pts = np.concatenate([ref[w]["lab_pts"] for w in SEGS])
    lab_nrm = np.concatenate([ref[w]["lab_nrm"] for w in SEGS])
    lab_seg = np.concatenate([np.full(len(ref[w]["lab_pts"]), i) for i, w in enumerate(SEGS)])
    all_pts = np.concatenate([ref[w]["all_pts"] for w in SEGS])
    all_nrm = np.concatenate([ref[w]["all_nrm"] for w in SEGS])
    t_lab = cKDTree(lab_pts)
    t_all = cKDTree(all_pts)
    print("trees", len(lab_pts), len(all_pts), f"{time.time()-t0:.0f}s", flush=True)

    # --- Will's patches
    pitches = []
    maps = {}
    for p in range(6):
        d = f"{will_dir}/patch_{p}"
        xyz, valid = load_xyz(d)
        meta = json.load(open(f"{d}/meta.json"))
        area = cell_area(xyz[::STRIDE, ::STRIDE], valid[::STRIDE, ::STRIDE]) * VX_UM ** 2 / 1e8
        nrm = grid_normals(xyz[::STRIDE, ::STRIDE], valid[::STRIDE, ::STRIDE])
        sv = valid[::STRIDE, ::STRIDE]
        if p in (0, 1):
            pitches.append(pitch_from_patch(xyz[::STRIDE, ::STRIDE], sv, nrm, rng, n_query=3000))
        q = xyz[::STRIDE, ::STRIDE][sv].astype(np.float64)
        dl, dlp = perp_dist(t_lab, lab_pts, lab_nrm, q, 8.0)
        da, dap = perp_dist(t_all, all_pts, all_nrm, q, 8.0)
        _, il = t_lab.query(q, k=1, workers=WORKERS)
        wa = area[sv]
        maps[p] = {"sv": sv, "dl": dl, "dlp": dlp, "da": da}
        res["will"][f"patch_{p}"] = {
            "uuid": meta.get("uuid"), "scale": meta.get("scale"), "grid": list(valid.shape),
            "valid_frac": float(valid.mean()), "area_cm2": float(area.sum()),
            "bbox_xyz": meta.get("bbox"), "_wa": wa, "_dl": dl, "_dlp": dlp, "_da": da, "_dap": dap,
            "_seg": lab_seg[il],
        }
        print(f"patch_{p}", valid.shape, f"{area.sum():.1f} cm2", f"{time.time()-t0:.0f}s", flush=True)

    pv = np.concatenate(pitches)
    pitch = float(np.median(pv))
    res["pitch_vx"] = {"median": pitch, "p25": float(np.percentile(pv, 25)), "p75": float(np.percentile(pv, 75)),
                       "n": int(len(pv)), "method": "normal offset to the same patch >= 15 cells (240 vx) away on its 16 vx grid, in-plane <= 10 vx (patch_0, patch_1)"}
    print("pitch", res["pitch_vx"], flush=True)

    tot = {"area": 0.0, "lab_same": 0.0, "lab_2": 0.0, "all_same": 0.0, "all_2": 0.0, "lab_4vx": 0.0}
    for p in range(6):
        r = res["will"][f"patch_{p}"]
        wa, dl, dlp, da, dap, seg = (r.pop(k) for k in ("_wa", "_dl", "_dlp", "_da", "_dap", "_seg"))
        # same sheet: normal distance <= pitch/2 with in-plane offset <= 8 vx (else not same sheet)
        same_l = np.nan_to_num(dlp, nan=np.inf) <= pitch / 2
        same_a = np.nan_to_num(dap, nan=np.inf) <= pitch / 2
        two_l = dl <= 2 * pitch
        two_a = da <= 2 * pitch
        c4 = dl <= 4.0
        A = wa.sum()
        r.update({
            "frac_area_same_sheet_as_labelled": float(wa[same_l].sum() / A),
            "frac_area_within_2_pitch_of_labelled": float(wa[two_l].sum() / A),
            "frac_area_within_4vx_of_labelled": float(wa[c4].sum() / A),
            "frac_area_same_sheet_as_any_mesh": float(wa[same_a].sum() / A),
            "frac_area_within_2_pitch_of_any_mesh": float(wa[two_a].sum() / A),
            "cm2_same_sheet_as_labelled": float(wa[same_l].sum()),
            "same_sheet_labelled_cm2_by_seg": {SEGS[i]: float(wa[same_l & (seg == i)].sum()) for i in range(6)},
            "median_dist_to_labelled_vx": float(np.median(dl)),
        })
        tot["area"] += A
        tot["lab_same"] += wa[same_l].sum()
        tot["lab_2"] += wa[two_l].sum()
        tot["all_same"] += wa[same_a].sum()
        tot["all_2"] += wa[two_a].sum()
        tot["lab_4vx"] += wa[c4].sum()
        maps[p]["cls"] = np.where(same_l, 0, np.where(two_l, 1, 2))
    res["total"] = {k: float(v) for k, v in tot.items()}
    res["total"]["frac_same_sheet_as_labelled"] = tot["lab_same"] / tot["area"]
    res["total"]["frac_within_2_pitch_of_labelled"] = tot["lab_2"] / tot["area"]
    res["total"]["frac_area_free_of_labelled_by_2_pitch"] = 1 - tot["lab_2"] / tot["area"]
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(f"{out_dir}/overlap.json", "w"), indent=1)

    # --- figure: each patch on its own grid, coloured by distance class to the labelled area
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d62728", "#ff9f40", "#4c78a8", "#dddddd"])
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    for p, ax in zip(range(6), axs.ravel()):
        sv = maps[p]["sv"]
        img = np.full(sv.shape, 3)
        img[sv] = maps[p]["cls"]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        r = res["will"][f"patch_{p}"]
        ax.set_title(f"patch_{p}: {r['area_cm2']:.0f} cm², same sheet as labelled "
                     f"{100*r['frac_area_same_sheet_as_labelled']:.1f} %, ≤2 pitch {100*r['frac_area_within_2_pitch_of_labelled']:.1f} %",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Will's PHerc1667 patches vs VES-2321 labelled area (7.91 µm frame, pitch {pitch:.1f} vx). "
                 "red = same sheet (≤ pitch/2), orange = ≤ 2 pitch, blue = farther, grey = no surface", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/overlap_map.png", dpi=80)
    print(json.dumps(res["total"], indent=1), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
