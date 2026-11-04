# Provenance

Every file below comes from the author's working repository at commit `60c99e44`. The only changed lines are: data
paths turned into environment variables with repository-relative defaults (variables and default folders renamed to
plain names, and the docstrings that name them), the ufsm Docker image name as a variable, import paths for the new
folder names, the Python paths in the shell scripts, a temp-file path, the label-free scorer's sha256 pins (re-
pinned to this tree), one internal branch name in a comment, and local paths in three stored result files (now
`$RUN_1667`). One default moved: the 1667 run folder defaults to attempt 2's run, the run behind the stored result.
No rule, threshold or number changed. One script fix: `pherc1667/attempt2/run_score_a2.sh` keeps the
attempt-1 `attempt4/results.json` aside only if it exists (it is not in this tree; the old script stopped there).
`shared/run_t23.py` is the latest version; its only other change since the copy the label-free run pinned is a
transport fix (the retry clause catches download errors instead of raising TypeError).

Checks run from a fresh clone (2026-11-03): `check_fold.py` re-scores PHerc. 1667 w028 and PHerc. 0139 w040 from the
stored fold outputs (local, not published yet) and matches `results.json` exactly; `labelfree1667/score_lf.py
--dryrun` passes and rewrites `results_dryrun.json` unchanged; `pherc0139/prep_data.py`, `pherc1667/fetch_meshes.py`
and `pherc1667/prep_data.py` rebuild meshes byte-identical and label masks equal to the scored-run inputs (compared
by the check script); `figure/depth_profile.py --scroll 1667` rewrites `figure/profile_1667.json` unchanged.

| file | change |
|---|---|
| `.gitignore` |  |
| `LICENSE` |  |
| `PROVENANCE.md` |  |
| `README.md` |  |
| `THIRD_PARTY.md` |  |
| `check_fold.py` |  |
| `depth_profile.png` |  |
| `figure/depth_profile.py` | 8 lines |
| `figure/plot.py` | identical |
| `figure/profile_0139.json` | identical |
| `figure/profile_1667.json` | identical |
| `labelfree1667/cands_lf.json` | identical |
| `labelfree1667/ctcheck.py` | 14 lines |
| `labelfree1667/g2_lf.json` | stored result, local paths redacted |
| `labelfree1667/grids_lf.json` | identical |
| `labelfree1667/infer_lf.json` | stored result, local paths redacted |
| `labelfree1667/points_lf.json` | identical |
| `labelfree1667/prep1.json` | identical |
| `labelfree1667/prep1.py` | 11 lines |
| `labelfree1667/r0.json` | identical |
| `labelfree1667/results.json` | stored result, local paths redacted |
| `labelfree1667/results_ct.json` | identical |
| `labelfree1667/results_dryrun.json` | identical |
| `labelfree1667/run_prep1_cpu.sh` | 4 lines |
| `labelfree1667/run_prep1_gpu.sh` | 2 lines |
| `labelfree1667/run_step1.sh` | 12 lines |
| `labelfree1667/score_lf.py` | 26 lines |
| `labelfree1667/selftest_step1.json` | identical |
| `labelfree1667/sites_lf.sha256` | identical |
| `labelfree1667/step1.py` | 7 lines |
| `labelfree1667/vendor/overlap_ves2337.py` | identical |
| `pherc0139/attempt3/prep3.py` | identical |
| `pherc0139/attempt3/run_gpu.sh` | 2 lines |
| `pherc0139/attempt3/shift_out.py` | identical |
| `pherc0139/attempt4/infer4.py` | identical |
| `pherc0139/attempt4/results.json` | identical |
| `pherc0139/attempt4/score4.py` | identical |
| `pherc0139/attempt4/sites4.py` | identical |
| `pherc0139/common.py` | 7 lines |
| `pherc0139/heldout/run_folds.py` | identical |
| `pherc0139/heldout/score_heldout.py` | identical |
| `pherc0139/heldout/train_fold.py` | identical |
| `pherc0139/prep.py` | identical |
| `pherc0139/prep_data.py` |  |
| `pherc0139/run_gpu.sh` | 2 lines |
| `pherc0139/score.py` | identical |
| `pherc0139/train.py` | identical |
| `pherc0139/xseg.py` | 2 lines |
| `pherc1667/attempt2/results.json` | identical |
| `pherc1667/attempt2/run_gpu_a2.sh` | 6 lines |
| `pherc1667/attempt2/run_score_a2.sh` | 10 lines |
| `pherc1667/attempt3/prep3.py` | identical |
| `pherc1667/attempt4/infer4.py` | identical |
| `pherc1667/attempt4/score4.py` | identical |
| `pherc1667/attempt4/sites4.py` | identical |
| `pherc1667/common.py` | 5 lines |
| `pherc1667/fetch_meshes.py` |  |
| `pherc1667/heldout/run_folds.py` | identical |
| `pherc1667/heldout/score_heldout.py` | identical |
| `pherc1667/heldout/train_fold.py` | identical |
| `pherc1667/patch1667.py` | identical |
| `pherc1667/points_old.sha256` | identical |
| `pherc1667/prep.py` | identical |
| `pherc1667/prep_data.py` | 5 lines |
| `pherc1667/run_gpu.sh` | 4 lines |
| `pherc1667/run_prep.sh` | 4 lines |
| `pherc1667/run_score.sh` | 2 lines |
| `pherc1667/run_sites.sh` | 2 lines |
| `pherc1667/score.py` | identical |
| `pherc1667/train.py` | identical |
| `pherc1667/xseg.py` | 2 lines |
| `points0139/run.py` | 11 lines |
| `requirements.txt` |  |
| `shared/pr2368_band.py` | identical |
| `shared/run_t23.py` | 11 lines |
| `shared/run_t4.py` | 5 lines |

## Release asset: stored outputs for PHerc. 1667 w028

`vesuvius-label-j-1667-w028-stored.tar.gz` (836 333 508 bytes, sha256
`cd62956d55460668a846a2f10b523ce86b9695e2ca6aef0867b4fa4f8b140c8e`) is not in git. It unpacks to `data/` and holds
exactly the 45 data files that `check_fold.py --scroll 1667 --fold w028` and `figure/depth_profile.py --scroll 1667`
open (listed by an audit hook during the check), plus `ARCHIVE_README.txt` and `SHA256SUMS`. Files are copied
unchanged from the scored run (attempt 2) and from `pherc1667/prep_data.py`'s output.

| files in the archive | what | terms |
|---|---|---|
| `run1667/heldout/out_*_hw028.npz`, `run1667/attempt4/out_*_hw028.npz`, `run1667/heldout/train_*_hw028.json` | outputs and training logs of the four held-out nets (J0, J1, D0, D1) | CC BY-NC 4.0 (made from the open data below) |
| `run1667/sites.npz`, `run1667/sites44.npz`, `run1667/attempt4/sites4.npz` | raw CT crops (uint8) around the scored spots, PHerc1667 20251217075048-2.399um-0.2m-78keV-masked, level 2 | Vesuvius Challenge open data, CC BY-NC 4.0 |
| `run1667/points_old.npz`, `run1667/attempt4/points4.npz` | per-point arrays (sheet spacings, depth masks) and hecate's 3D output at the points | CC BY-NC 4.0; hecate (scrollprize/hecate) is MIT |
| `run1667/prep.json`, `run1667/xseg.json`, `run1667/attempt4/sites4.json`, `run1667/attempt4/xseg4.json` | spot lists, segment-crossing lists | CC BY-NC 4.0 |
| `pherc1667/*/tifxyz/*`, `pherc1667/*/labels/*` | level-2 meshes and ink label masks of the five labelled segments | Vesuvius Challenge open data (release 20260918), CC BY-NC 4.0 |

Nothing from the fresh PHerc. 1667 surface upload is in the archive. Check run from a fresh clone of `b9941526`
(2026-11-04, `CUDA_VISIBLE_DEVICES=`, CPU torch): both checksums pass, `check_fold.py` prints MATCH for all four nets,
`figure/depth_profile.py` prints PASS, the tree is unchanged, and no file outside the clone and the unpacked archive
was read.
