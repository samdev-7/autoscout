# Local storage cleanup (2026-11-01)

What was kept, compressed and deleted on the development machine, and why.
Everything here lives under git-ignored paths (`data/`, `out/`, `runs/`), so
this is a record of local state, not of the repository.

Guiding rule: **keep anything that cannot be regenerated and is on the path
of the result we use; delete anything that is a copy, a superseded attempt,
or re-derivable from kept inputs with one command.**

## What the result actually depends on

| artefact | path | role |
|---|---|---|
| detector in use | `runs/runpod_5090/weights/epoch60.pt` (152 MB, resumable) | produced every detection the tracks are built from; sha256 `8c8beb0d8ebf53b5…` |
| same run, stripped | `runs/runpod_5090/best.pt`, `last.pt` (50 MB each) | inference-only copies; `best.pt` sha256 `5bcae832038e42e1…` |
| run record | `runs/runpod_5090/{results.csv,train.log,args.yaml,launch.sh,ckpt_sweep.*,*.png,*.jpg}` | metrics, curves, exact launch command |
| local baseline | `runs/detect/out/runs/fullB/weights/best.pt` (19 MB, yolo11s) | the comparison model in the earlier side-by-side videos |
| wide detections | `out/d_epoch60.npz` → `out/dets_e60c30{.npz,_xy.npy,_app.npz}` | raw and filtered detections the tracks use |
| station detections | `out/dets_red_station.npz`, `out/dets_blue_station.npz` | |
| tracks | `out/tracks_fused.npy` (result), `out/tracks_greedy.npy` (baseline) | |
| calibration | `out/cams.json`, `out/cam_params.npy`, `out/cam_*_solved.npy`, `out/wide_cam.json`, `out/lens_red_station.npy` | |
| calibration inputs | `out/points_multi.json`, `points_corrected.json`, `pairs_xview.json`, `lines_multi.json`, `tags_views.json`, `tag_marks.json`, `tag_guides.json`, `out/backup/` | hand-marked; irreplaceable |
| labels | `out/labels.json`, `out/trace.json`, `out/heldout_clips.json`, `out/label_field_pos.json` | hand-made; irreplaceable |
| validation sets | `out/val_combined/`, `out/val_multi/`, `out/yolo_val/`, `out/yolo_val_full/` (139 MB) | what checkpoints were selected on |
| plates / masks | `out/plate_*.png`, `out/frame_*.png`, `out/enh_*`, `out/top_plate.npy`, `out/carpet_mask.npy`, `out/field_box.png` | |
| published demo | `out/fused_full_demo.mp4` (88 MB) | the one on the Hack Club CDN |
| source video | `data/match1_qual.mp4` (78 MB) | |
| field assets | `data/field/` | from WPILib |

## Compressed

- **`data/rf_sources_used.tar.zst` (808 MB)** — the seven Roboflow sources that
  `src/build_final.py` actually uses (`frcrobots_v5, jerrywu_v2, t6036, grain,
  frc4419, capstone, scout_v7`; 14,996 files, 905 MB uncompressed). Verified by
  full decompression. JPEGs do not compress, so the gain is small; the point is
  one file instead of 15k, and the uncompressed copies can go. Restore with:

  ```bash
  mkdir -p data/rf && zstd -dc --long=27 data/rf_sources_used.tar.zst | tar -x -C data/rf
  ```

## Deleted (executed 2026-11-01)

Sizes are what `du` reported before deletion. Measured free-space gain: **13 GB**.
After deletion `data/` is 883 MB, `out/` 350 MB, `runs/` 512 MB, and the kept
weights' hashes were re-verified. `fuse_track.py` → `eval_trace.py` still
reproduces `out/tracks_fused.npy` exactly (1 ID switch).

| group | paths | size | why it is safe |
|---|---|---|---|
| A | `out/armA out/arm_A out/arm_A2 out/armB out/arm_B out/yolo_train out/yolo_train2 out/yolo_train3` | 4.5 GB | training-pool copies for the *local* runs; rebuilt from the archive by `src/build_final.py` / `build_arms*`. The model in use was trained on the pod straight from Roboflow (`src/train_kaggle.py`), not from these. |
| B | `data/rf/{alis,adamk,scout2025,pack1294}` | 2.6 GB | rejected sources (under-labelled, mosaic-augmented, bad boxes — see the comment in `build_final.py`) |
| B | `data/rf/jerrywu_v1` | 1.2 GB | superseded by `jerrywu_v2` |
| B′ | `data/rf/{frcrobots_v5,…,scout_v7}` uncompressed | 905 MB | now in the archive above |
| C | `runs/runpod_5090/weights/epoch{0,10,…,50,70,…,110}.pt`, `best_ep15_resumable.pt` | 1.75 GB | checkpoints of the same run that were *not* selected; `epoch60.pt` is kept resumable, `best.pt`/`last.pt` kept stripped |
| D | `runs/detect/out/runs/{sA,sB,smoke,smokeA,smokeB,speed_baseline}` | 219 MB | smoke tests and a speed benchmark |
| D | `epoch*.pt`, `last.pt` under `runs/detect/out/runs/{fullA,fullB,fullB_960,ext,mosaicoff}` | 1.2 GB | per-epoch resume points of superseded local runs; each run's `best.pt`, `results.csv`, `args.yaml` kept |
| D | `runs/detect/val*` (29 dirs) | 0 | empty |
| E | `out/baseline_weights/` | 145 MB | byte-identical (md5) to `runs/detect/out/runs/ext/weights/` |
| F | `kaggle_pkg/`, `autoscout_kaggle.zip` | 221 MB | packaging for the cloud run: its val sets are byte-identical to `out/val_*`, its scripts are now `src/train_kaggle.py` and `src/experiments/` |
| G | `yolo11s.pt yolo11m.pt yolo11l.pt` | 106 MB | COCO base weights; ultralytics re-downloads them on first use |
| H | every `out/*.mp4` except `fused_full_demo.mp4` | 1.1 GB | renders; regenerate from `out/tracks_fused.npy` with `render_cmp.render_one` (INSTRUCTIONS §6) |
| I | superseded sweep arrays in `out/` (`app960_*`, `d960_*`, `t640/t960`, `tracks_{0.10…0.25,em,asg,rep,win,…}.npy`, `appear*.npz`, `dets_fullB*`, `cam_try_*`, …) | 161 MB | outputs of detector/tracker sweeps whose winner is `dets_e60c30` → `tracks_fused.npy`; all listed in `src/experiments/` drivers |
| J | `out/pair/`, `out/trace_frames/` | 254 MB | frame dumps for the pairing and tracing tools; regenerate in seconds with `src/pair_frames.py` / `STEP=6 T0=8 T1=172 src/trace_frames.py` (needed only to *open* those tools, not for their saved data) |

Freed: about **13.5 GB** of the 14.2 GB that `data/`, `out/`, `runs/` and the
loose weights occupied.

## Not touched

`.venv/` (1.4 GB, recreated from `requirements.txt`), `out/frames/` (23 MB, the
frames `out/labels.json` refers to), the validation sets, every JSON, every
calibration `.npy`, `out/backup/`, the median plates, and the published demo.

## If you need to…

- **re-run detection on a new match** — `W=runs/runpod_5090/weights/epoch60.pt`
  (or `best.pt`) with `src/detect_match.py`, INSTRUCTIONS §3.
- **resume or fine-tune the detector** — `epoch60.pt` is a full resumable
  checkpoint; pin `ultralytics==8.4.138` (resuming across versions is unsafe).
- **rebuild the training pool** — restore the archive (above), then
  `src/build_final.py`. For a fresh cloud run, `src/train_kaggle.py` pulls the
  sources from Roboflow itself and needs only the key.
- **re-render** — INSTRUCTIONS §6 from `out/tracks_fused.npy`.
