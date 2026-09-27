# Instructions

How to take one broadcast match video through calibration, detection, tracking,
evaluation and rendering. Everything runs from the repository root. All paths
below are relative to it; every script reads and writes under `out/`.

Nothing event-specific is in the repo. You will produce, in order: a video, a
field diagram, calibrations, detections, tracks, and renders. Each step lists
what it needs from the previous ones.

---

## 0. Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p data/field out
```

**Video.** Put the match VOD at `data/match1_qual.mp4` (H.264, 1920×1080, 30 fps
is what the layout constants assume). Broadcasts from FIRST are a three-panel
composite: wide field camera across the top, red and blue driver-station cameras
below. The crop rectangles live in `src/detect_match.py` (`CROP`) and
`src/geom.py` (`BAND0`, `BAND1` — the rows of the wide panel that contain the
field). Check them against your video's first in-game frame before anything else.

**Field diagram.** The easiest source is WPILib's field assets, which ship a
top-down image plus a JSON with the field dimensions and the pixel box of the
playing area, and the AprilTag layout for the season:

- <https://github.com/wpilibsuite/allwpilib/tree/main/fieldImages/src/main/native/resources/edu/wpi/first/fields>
  — `<season>-<game>.png` and `<season>-<game>.json` (`field-dimensions`,
  `field-image` crop box)
- <https://github.com/wpilibsuite/allwpilib/tree/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag>
  — `<season>-<game>-welded.json`, the tag positions (also published as a CSV in
  the game manual's field drawings)

Put them in `data/field/`. Field length and width in metres are `LEN`, `WID` in
`src/geom.py` (WPILib's `field-dimensions`). The renderers read
`out/field_box.png`: the image cropped to exactly the `field-image` box (carpet
edge to carpet edge, no margin), so that pixel↔metre is a pure scale. The tag CSV
the solver reads is `ID, X, Y, Z` in inches, matching the manual's table.

**Weights.** Model weights are not distributed. Either train (§7) or place a
YOLO `.pt` file anywhere and pass its path with `W=`. The pipeline was developed
with a YOLO11l fine-tune.

**Start the tool server** whenever a browser tool is mentioned:

```bash
.venv/bin/python src/serve.py
```

It serves the repo on <http://127.0.0.1:8765>, saves tool output into `out/`
(every save first snapshots the previous file to `out/backup/`), and runs the
calibration solver live for `calib.html`.

---

## 1. Frames and plates

```bash
.venv/bin/python src/extract_frames.py          # out/frames/*.jpg, out/frames.json (every 2 s, for labelling)
T0=7.25 T1=172 .venv/bin/python src/plates.py   # out/plate_<view>.png median plates + first-frame plates
.venv/bin/python src/tags_views.py              # AprilTag corners in every panel -> out/tags_views.json
```

`T0`/`T1` are the in-game time range in seconds (after the broadcast switches to
the match layout, before the end-of-match graphics). Median plates remove the
robots so ground features are visible for calibration; AprilTags are detected on
many frames and median-filtered.

**If a camera moves during the match** (a station camera being bumped is common),
treat before and after as two views. The convention is `red_station` and
`red_station_b`, with the cut frames in `src/mcam.py` (`RED_CUT0`, `RED_CUT1`).
Same lens, different pose.

---

## 2. Calibration

The wide camera is solved first; the station cameras are then solved into the
same field frame.

### 2a. Wide camera

Open <http://127.0.0.1:8765/calib.html>, choose the **wide** view, and click
matching points between the plate and the field diagram — carpet features only
(tape corners, structure bases), never anything with height. Mark AprilTag
corners where they are detected; the tool shows candidate positions. Save.

Then solve:

```bash
.venv/bin/python src/calib3d_robust.py     # -> out/cam_params.npy (used by src/geom.py)
```

`geom.py` prints a self-check when run directly: round-trip error, the pixel
position of each field corner (must sit inside the band) and the predicted
measurement sigma across the field.

```bash
.venv/bin/python src/geom.py
```

> `calib3d_robust.py` reads `out/points_corrected.json` and `out/tag_marks.json`
> and contains three hand-verified tag pixel positions from the development
> video in its source. Replace those with your own detected tags (from
> `out/tags_views.json`) or remove that line.

### 2b. Station cameras

Three sources of constraint, all through browser tools. The solve needs at
least 6 constraints per view and gets better with more:

- **Straight lines** — <http://127.0.0.1:8765/lines.html>. Trace lines that are
  straight in the world (field border, wall edges, banners) and label their
  direction. This fits the lens distortion (`k1`, `k2`) and principal point,
  and the vanishing points fix the focal length. The purple overlay is the fit's
  own prediction of those lines; it should match the curvature you see.
- **Carpet points and tags** — `calib.html`, same as the wide view, per station.
- **Cross-view pairs** — <http://127.0.0.1:8765/pair.html>. Needs
  `.venv/bin/python src/pair_frames.py` first (1 Hz frame pairs). Scrub to a
  moment when a robot's floor contact is visible in both the wide view and a
  station view and click it in both; the wide click is mapped through the solved
  wide camera to a field coordinate and becomes a carpet point for the station.
  This is how you get ground constraints where the station view has no visible
  features.

`calib.html` re-solves live (via `/solve`) as you add constraints and reports
median and carpet reprojection residuals. Aim for a median under ~2 px on a
960×540 panel. Then write all cameras into one file:

```bash
LENS_FROM=red_station_b:red_station .venv/bin/python src/write_cams.py   # -> out/cams.json
```

`LENS_FROM` reuses a lens for a view of the same physical camera after it moved.
`out/cams.json` is what every later stage reads (through `src/mcam.py`).

Sanity-check the geometry of all views together:

```bash
.venv/bin/python src/mcam.py         # per-view residuals and view-axis angles
.venv/bin/python src/uncertainty.py  # 1-sigma ellipse per camera and fused, across the field
```

---

## 3. Detection

Run the detector on each panel. Decoding is done once per run; keep `T0`/`T1`
consistent with §1 (they are `T0, T1` in `src/detect_match.py`).

```bash
W=path/to/best.pt CONF=0.10 O=out/dets_wide.npz                             .venv/bin/python src/detect_match.py
W=path/to/best.pt CONF=0.10 O=out/dets_red_station.npz  PANEL=red_station   .venv/bin/python src/detect_match.py
W=path/to/best.pt CONF=0.10 O=out/dets_blue_station.npz PANEL=blue_station  .venv/bin/python src/detect_match.py
```

Detections are saved raw at a low confidence so thresholds can be chosen later
without re-running inference. Each row is
`frame, t, tid, x0, y0, x1, y1, conf, alliance` in panel pixels.

Then turn the wide detections into field positions and appearance descriptors
(the confidence threshold and field-space NMS are applied here):

```bash
DETS=out/dets_wide.npz CONF=0.30 OUT=out/dets_wide_c30 .venv/bin/python src/project_dets.py
# -> out/dets_wide_c30.npz, out/dets_wide_c30_xy.npy, out/dets_wide_c30_app.npz
```

Station detections are projected inside the tracker, because each needs its own
camera's corner correction.

---

## 4. Tracking

```bash
DETS=out/dets_wide_c30.npz XY=out/dets_wide_c30_xy.npy APP=out/dets_wide_c30_app.npz \
RED=out/dets_red_station.npz BLUE=out/dets_blue_station.npz \
OUT=out/tracks.npy .venv/bin/python src/fuse_track.py
```

This builds pure tracklets per camera, merges same-view duplicates, uses
station tracklets as bridges across wide-view gaps, solves identity exactly
(three paths per alliance) and smooths. The output is a list of six track
dicts (`alliance, xy, v, obs, P`) indexed by frame from `T0`.

Useful switches (environment variables; defaults are in the file header):

| var | meaning |
|---|---|
| `WIDE_ONLY=1` | ignore the station cameras (baseline) |
| `CONF_ST` | station detection confidence (default 0.30) |
| `MISS` | frames a tracklet may coast before closing |
| `DUP_M` | same-view duplicate distance, metres |
| `W_M W_H W_C W_R W_G` | link-cost weights: motion, height, colour, reach, unexplained gap |

The stdout summary per identity — tracklet count, observed %, frames filled from
stations, gaps > 1 s — is the first thing to look at. Two same-alliance tracks
closer than a robot footprint, any speed above 5 m/s, or a position off the
carpet are physically impossible and indicate a problem upstream.

---

## 5. Evaluation (ground truth by hand tracing)

The only evaluation that can see an identity switch is a trace that follows one
robot through time.

```bash
STEP=6 T0=8 T1=172 .venv/bin/python src/trace_frames.py   # 5 Hz frames -> out/trace_frames/
```

Open <http://127.0.0.1:8765/trace.html>. Trace **one robot at a time**: click its
floor-contact point (horizontal centre of the robot, nearest bumper edge on the
carpet), press `O` when it is occluded rather than guessing, `C` re-centres the
follow view. Saves to `out/trace.json`.

```bash
TRK=out/tracks.npy .venv/bin/python src/eval_trace.py
```

Reports per traced robot: matched %, position error p50/p90 split into depth and
lateral, ID switches and purity. Every tracker parameter was tuned on one match;
treat a second traced match as the real test.

For detector-level validation (boxes rather than trajectories) use
<http://127.0.0.1:8765/label.html> on the `extract_frames.py` output, then
`src/prep_val*.py` and `src/evaluate.py`.

---

## 6. Rendering

```bash
CW=1920 .venv/bin/python -c '
import numpy as np, sys; sys.path.insert(0, "src"); import render_cmp as R
T = list(np.load("out/tracks.npy", allow_pickle=True))
d = np.load("out/dets_wide_c30.npz", allow_pickle=True)["dets"]; xy = np.load("out/dets_wide_c30_xy.npy")
R.render_one(T, "fused tracker", 8.0, 172.0, "out/tracks.mp4", d, xy)
'
ffmpeg -i out/tracks.mp4 -c:v libx264 -crf 24 -pix_fmt yuv420p out/tracks_h264.mp4
```

`render_cmp.render(A, B, labelA, labelB, t0, t1, out, dA, xA, dB, xB, stations=True)`
draws two trackers side by side instead.

The render shows the broadcast panel with boxes coloured by claiming track, the
top-down field with 2σ ellipses and fading trails that break at observation
gaps, both station panels, and a line from each robot to every camera that saw
it, coloured by that camera's share of the fused information (green = most,
orange = least). To add the broadcast audio, mux it with `-ss <T0>` and `-shortest`.

---

## 7. Training the detector (optional)

The pipeline does not require you to train; any YOLO robot detector works with
`W=`. The training pool used external datasets for training and this project's
own labels for validation only, to keep the transfer measurement honest.

- `src/fetch_datasets.py` — downloads the external datasets (needs a Roboflow key
  in `.roboflow_key`, git-ignored; the script never prints it).
- `src/build_final.py` — normalises classes, drops non-robot labels, dedupes.
- `src/build_multival.py`, `src/prep_val.py`, `src/prep_val_full.py` — validation
  sets: this match's labels, and whole held-out clips from other events.
- `src/train.py`, `src/fullrun.py`, `src/cont640.py` — local training and the
  checkpointed full run; `src/ckpt_sweep.py`, `src/bench_models.py`,
  `src/imgsz_sweep.py` — pick a checkpoint on **each** validation set separately.
- `src/train_kaggle.py` — a self-contained cloud run (pinned versions; the
  Roboflow key comes from a secret named `ROBOFLOW_KEY`).

Select on the multi-event validation set, not the development match, and expect
best-checkpoint numbers to be optimistic.

---

## Known limitations to keep in mind

See `docs/COMPONENTS.md` for the measured version of each. In short: everything
assumes robots are on the carpet; a partially occluded box gives a wrong depth
with a normal-looking confidence; station cameras are precise across their view
axis and poor along it; calibration and layout constants are per event and per
camera and must be redone for every new stream.
