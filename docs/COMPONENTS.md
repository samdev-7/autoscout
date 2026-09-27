# Component reference — accuracy, reliability, limitations

One entry per independent component, so integration decisions can be made at a
glance. Every number here was **measured on `data/match1_qual.mp4`** (2026 ONT
Niagara, 190.7 s, 1920×1080@30 fps) against the hand trace `out/trace.json`
unless stated otherwise. One match is one match: treat every figure as a point
estimate, not a distribution, and expect the tracker figures in particular to be
optimistic because every tracker parameter was tuned on this same match.

Legend for the "status" column: ✅ verified and stable · ⚠️ works, with a known
sharp edge · ❌ known wrong / not yet usable as-is.

| # | Component | Status | Headline number | Biggest limitation for integration |
|---|---|---|---|---|
| 1 | Broadcast layout | ✅ | fixed 3-panel composite | red camera physically moved at t=103.3–104.3 s |
| 2 | Detector (YOLO) | ⚠️ | not the accuracy bottleneck (err flat ~0.125 m over 3× mAP) | partially-occluded boxes give systematically wrong depth and are not flagged |
| 3 | Wide calibration | ✅ | tags reproduced to 1.24 px | — |
| 4 | Station calibrations (×3) | ⚠️ | median residual 1.2–1.8 px | cameras at Z≈0.8 m → depth 1-σ up to 3.1 m |
| 5 | Image→field geometry | ✅ | corner correction 0.579±0.057 m; `mcam.py` does it per camera | cross-camera bias measured ≤ 0.06 m (on solver inputs, not blind) |
| 6 | Uncertainty model | ⚠️ | wide 14.5 cm → fused 6.3 cm (2.33×) | station pixel noise was 3–4× under-modelled; now set from measured residuals |
| 7 | Field-space tracklets | ✅ | **100 % pure** (0 of 796 traced samples on a wrong robot) | fragmented: ~230 tracklets per match; gaps p50 0.93 s, p90 2.57 s |
| 8 | Duplicate merge | ✅ | 36 alternating duplicate pairs, 35 invisible to the old test | fixed (span-interpolated 0.7 m); merges stay 100 % pure |
| 9 | Appearance / ReID | ⚠️ | within-alliance tracklet AUC 0.75 colour / 0.92 height (red) | tiebreaker only; colour fixes 1 switch, height neutral; cross-camera unmeasured |
| 10 | Identity solver | ✅ | **1 switch / 95.5 % purity** (exact K-path min-cost flow; was 13 / 84 % greedy) | remaining error is a true double re-emergence no camera saw |
| 11 | Smoother (RTS) | ✅ | err p50 0.09–0.17 m where observed | asserts a position during unobserved gaps (48–77 % observed) |
| 12 | Evaluation (trace) | ⚠️ | 1,156 samples, 182 occluded, 4 of 6 robots | per-frame Hungarian flips between coasting duplicates; only 77 pts on b1 |
| 13 | Station detections | ⚠️ | hit rate on visible traced robots only 19 % (red) / 32 % (blue) | low recall + 15–24 px box noise + ~0.15 m lateral bias of unknown origin |
| 16 | Fusion (`fuse_track.py`) | ✅ | +734 observed frames inside gaps, identity unchanged (1 switch) | station tracklets are link *evidence*, never merged — merging them made impure tracklets |
| 14 | Visualisation | ✅ | — | — |
| 15 | Calibration tooling (HTML) | ✅ | — | every save snapshots to `out/backup/` |

---

## 1. Broadcast layout

**What.** Match VODs are a 3-panel composite: a wide field camera across the top
(1920×490, the field occupies rows 98–442 = `BAND0..BAND1`) and two driver-station
cameras below (red, blue). The layout is fixed for the whole match; the play period
never shows a full-screen wide field.

**Reliable.** Panel boundaries are constant. All three panels are cut from the
same broadcast frame, so the three views are **time-synchronised by construction**
— no sync estimation is needed anywhere downstream.

**Limitations.**
- The red station camera was **physically bumped between t=103.33 s and 104.33 s**
  and stays in the new pose for the rest of the match. Intrinsics and crop did not
  change (same physical camera, same feed). A mean-RGB stability check was blind to
  this; the user caught it by eye. Any red-side quantity must be split at the cut:
  `red_station` before, `red_station_b` after.
- On-screen graphics obscure ground features in the station panels (this drove the
  cross-view pairing tool).

---

## 2. Detector

**What.** YOLO trained on online datasets (a pool of FRC robot datasets), with the
user's labels held out for validation only. Winning checkpoint for tracking:
**epoch60 at conf ≥ 0.30**, followed by field-space NMS (§5). Runs on a standard Mac.

**Measured.**
- Position error is **flat at ~0.125 m across a 3× range of mAP** — the detector
  is not what limits position accuracy.
- 18 points more recall bought 3 fewer ID switches with the greedy solver; with the
  exact solver the effect is far larger (1 vs 15 switches between the two detection
  sets) because recall is what makes elimination possible (§10).
- 100 % of accepted detections project inside the field.
- Box width matches the geometric prediction within 3–7 %.

**Reliable.** Fully visible robots on carpet.

**Limitations.**
- **Domain mismatch**: the training pool drifts away from our broadcast panel, and
  the combined validation set cannot choose a checkpoint — evaluate per validation
  set, never combined, or `ext`-style overfitting hides.
- Best-checkpoint mAP carries the winner's curse; do not compare it to a single
  fixed checkpoint.
- **Partially occluded / truncated boxes**: the box bottom is the *occluder's*
  edge, not a floor contact, so the derived position is wrong by metres in depth
  with a normal-looking confidence. Nothing flags this today. It is the "garbage
  in" case for every downstream stage.
- AP50 is the wrong headline metric for this problem: only the bottom edge is
  used, box height is both unreliable and irrelevant.

---

## 3. Wide calibration

**What.** `out/cam_params.npy` → `geom.py` (`K`, `DIST`, `RVEC`, `TVEC`). Focal from
vanishing-point orthogonality, distortion from plumb lines, pose by PnP with fixed
intrinsics. Camera centre ≈ (8.00, 23.18, 5.11) m.

**Measured.** Field→image→field round trip exact (0.000 mm). Detected AprilTag
corners reproduced to **1.24 px**.

**Reliable.** Locked-off camera; this is the reference frame everything else is
solved into.

**Limitations.** Reprojection RMS and plumb-line tests *both* misled during
calibration — a small RMS does not prove the model; the tag-square projection was
the only unambiguous test. See memory `autoscout-wide-camera-calibration`.

---

## 4. Station calibrations

**What.** `out/cams.json` holds `red_station`, `red_station_b`, `blue_station`,
`wide`, each with `f, cx, cy, k1, k2, R, t, W, H` plus residuals (`median`,
`carpet`). Solved by `solve_api.solve(view, pts, lines, lens_from)` from carpet
points (`points_multi.json`), cross-view pairs (`pairs_xview.json`), straight
lines (`lines_multi.json`) and detected AprilTags (`tags_views.json`).

**Measured.**
- Camera centres: red ≈ (15.8, 8.3, **0.79**) m, blue ≈ (0.57, 8.43, **0.80**) m —
  at driver-station height, looking across the field.
- Residuals after outlier removal: blue median 1.22 px (carpet 3.64); red median
  1.81 px. Lens is **pincushion**, k1 ≈ +0.37, k2 ≈ +0.76 (a first sweep hit its
  bound and wrongly concluded "no distortion").
- `red_station_b` shares intrinsics with `red_station` (`LENS_FROM`); only pose
  differs.
- Angle between wide and station view axes at field points: 21–86°, median ~54°.

**Reliable.** Lateral (perpendicular to the view ray) position; the tag-based
solve once bad hand-marks were removed.

**Limitations.**
- **Grazing geometry.** At Z≈0.8 m the depth 1-σ ranges 0.1–3.1 m across the
  field. Station positions are precise in one axis only.
- Hand-marked tags are unusable for the solve: the editor seeds corners in image
  order, the detector reports them in tag order — a corner-*ordering* mismatch that
  looked like a 21 px error. Use detections only.
- Two outliers had to be dropped (a blue point 0.6 m from the camera; red tags 7/9
  marked at predicted, not detected, positions). Residuals are only meaningful
  after that.
- The carpet points are outnumbered 8:1 by tag corners yet are the only thing that
  pins the ground plane. "7–8 carpet points is enough" was wrong for the 3-D solve
  (it came from the homography check, which ignores tags).

---

## 5. Image → field geometry (`src/geom.py`)

**What.** Flat-ground (Z=0) back-projection. A detection's `(x-centre, bottom
edge)` is the robot's **nearest ground corner**, not its footprint centre, so the
point is pushed away from the camera by the yaw-averaged corner distance
`CORNER_MEAN = 0.579 m` (`± 0.057`). The human tracer produces the same
convention. `field_nms(sep=0.70)` suppresses same-frame detections closer than a
footprint — depth-correct where IoU-NMS is not.

**Measured.** Corner correction validated: skipping it puts every robot ~0.6 m too
near the camera. Depth error is **2–4× lateral**, as the Jacobian predicts.
Same-alliance traced robots were **never closer than 0.99 m** (r1–r2, 151 common
samples), so the 0.70 m NMS radius did not merge real robots in this match.

**Reliable.** For the wide camera.

**Limitations.**
- `geom.py` itself is **wide-only** (`_away()` closes over the wide `CAM`). The
  per-camera version is **`src/mcam.py`** (2026-09-04): `Cam(name)` from
  `cams.json` with `project / img_to_ground / point_to_field / box_to_field / cov /
  height_of`, the red bump split (`RED_CUT0..1` = frames 3100–3130, `red_view(f)`),
  and `WideCam` delegating to `geom.py` because the `cams.json` wide entry is a
  different fit (disagrees by up to 25 px at the edges). Round trip exact for all
  four cameras. Before this existed, station detections got no corner correction
  and sat 0.579 m toward their own camera — fixing it moved "station detection
  within 0.5 m of an occluded robot" from 8 % to 27 %.
- Cross-camera **bias** against the hand-marked ground pairs: |mean| 0.02–0.06 m,
  rms 0.08–0.13 m for all three station poses — but those pairs were also solver
  inputs, so this is a residual, not a blind test.
- Height is not modelled at all (accepted design decision: height and depth are not
  separable from one view). Anything not touching the carpet (a climbing robot, a
  lifted robot) is placed wrong.
- The yaw term (`CORNER_STD`) is included in `geom.cov` but **not** in
  `uncertainty.cov`; the two covariance functions disagree by that much.

---

## 6. Uncertainty and fusion model (`src/uncertainty.py`)

**What.** Per-camera field-space covariance: pixel σ = detection noise ⊕
calibration residual (in quadrature), pushed through each camera's own
finite-difference Jacobian → anisotropic, position-dependent ellipse. `fuse()` is
the information-form (inverse-covariance) sum.

**Measured.** Wide alone 14.5 cm → fused 6.3 cm (**2.33×**). 97 % of the field is
seen by ≥2 views in both time segments. Occlusions in the wide view cluster at
**median Y = 1.7 m (far side)**, only 2 % in the near strip Y > 6 — the guardrail
band was *not* the main loss.

**Reliable.** The *shape* of each ellipse and the relative weighting between
cameras.

**Limitations.**
- Models zero-mean noise. A **systematic calibration offset** between views
  neither averages out nor appears in the ellipse; it makes every cross-camera
  comparison wrong in the same direction. Must be measured separately (against
  `pairs_xview.json`) before any cross-camera association is trusted.
- Omits the yaw/corner term (see §5).
- `SIG_DETECT` (4 px) was **3–4× too small for the stations**. Measured against
  traced robots projected into the panel: red du sd 14.7 px (+22.8 px bias), dv sd
  6.1; blue du sd 23.7 (−11.5 bias), dv sd 11.1 — robots span 30–300 px there, so
  box edges are sloppy. Field-space: lateral residual 3.4–3.9× the model, depth
  1.3–1.6×. `mcam.SIG_PX` now carries anisotropic (su, sv) per station with the
  bias folded in as noise. The lateral bias (opposite sign on the two stations)
  is **not understood** — candidates: yaw calibration error growing with range,
  distortion model at the panel edges, silhouette-centre vs footprint-centre
  for diagonally viewed robots. Under-modelling it made cross-view merges both
  miss and mis-fire, and let stations over-pull fused positions (r1 err 0.09 →
  0.19 m).

---

## 7. Field-space tracklets (`src/track.py`)

**What.** Constant-velocity Kalman in field metres. `VMAX=5.0 m/s`, `SIG_A=7.0
m/s²`, `GATE=13.8` (χ², 2 dof), `MISS_MAX` 20 frames (0.67 s) in the winning
config, `MIN_LEN=5`. A **hard reachability gate** (`VMAX·Δt`) sits in front of the
Gaussian gate — without it tracks teleported at 29 m/s. Each tracklet records its
source detection indices `di` (keying on filtered positions mapped only 8 % of
detections).

**Measured.** **0 internal identity switches; 0 of 796 traced samples inside a
tracklet lie on a minority robot.** Tracklets are pure. Every identity error in the
pipeline is made *after* this stage.

**Reliable.** Short-term continuity on a visible robot.

**Limitations.**
- **Fragmentation.** ~230 tracklets per match for 6 robots. Immediate same-robot
  successor gaps: **p50 0.93 s, p90 2.57 s, max 5.97 s; 13 of 29 exceed 1 s.**
  Over 1–2.5 s a robot at ≤5 m/s can be anywhere on the field, so *no* motion
  model can link across those gaps. This gap distribution is the single most
  important input to the identity problem, and shortening it is what more cameras
  are for.
- **Alternating duplicates.** When a second tracklet spawns on an already-tracked
  robot, the one detection per frame goes to one of them and the other coasts, so
  the two *alternate* frames and never share an observed frame (see §8).
- `MISS_MAX` counts frames without a *wide* detection. For multi-camera use it
  must count frames without a detection from *any* camera.

---

## 8. Duplicate merge (`assign.build`)

**What.** Before identity assignment, same-alliance tracklets that are one robot
are merged (union-find; positions averaged per frame).

**Old test.** ≥5 common observed frames and median Mahalanobis < `DUP=6.0`.
**Blind spot:** alternating duplicates have *no* common frames. In this match, 36
same-alliance pairs overlapped in span at < 0.7 m (the distance distribution is
bimodal: 34 pairs < 0.5 m, 3 in 0.5–1.0, 14 in 1.0–2.0, 355 beyond); **35 of the
36 had < 5 common frames** and were never merged. The concurrency rule then forced
each pair into *different* identities — 36 wrong hard constraints — producing the
hand-off flip-flops on b2 (frames 1980–2136) and r1 (2166–2346).

**New test (2026-09-03).** If < 5 common frames, compare positions interpolated over
the span overlap; merge if median distance < `DUP_M=0.7 m`. Result: 235 → 200
tracklets, merged tracklets still **100 % pure** (0 of 799 samples).

**History.** With the *greedy* solver, end-to-end got worse after this fix (13 → 16
switches) because that solver had adapted to the extra constraints. With the exact
solver (§10) the clean merge is strictly better. Lesson kept: judge a component at
its own level, not by a fragile consumer.

---

## 9. Appearance / ReID (`src/appear.py`, `assign.bhat_sets`)

**What.** HSV 2-D histogram (12×3) + V histogram (4) over `chas` (the box) or
`body` (above the box). Tracklets keep `NEX=24` exemplars; distance between
tracklets is the **10th percentile** of pairwise Bhattacharyya distances
(`PCT=10`), robust to outliers and much better than a mean descriptor (EER 1.8 %
vs 6.6 % in the earlier verification test).

**Measured (this review, on pure trace-labelled tracklets, within alliance).**
- red: AUC **0.818** (114 same / 117 different pairs); same p50 0.034, diff 0.045.
- blue: AUC **0.724** (31 / 35 pairs).
- **Physical height** (box top → `geom.height_of` given the ground position; per-tracklet
  median) is the strongest single within-alliance cue measured: red AUC **0.917**
  (r1 ≈ 0.33–0.52 m vs r2 ≈ 0.55–0.77 m), chas+height 0.936 vs chas 0.75; blue 0.745
  (b1 has only 5 labelled tracklets, heights 0.27–0.63). Colour-independent and
  perspective-corrected, but it *is* box-top based: truncation and raised mechanisms
  move it. The `body` region did not help (0.695 / 0.718).
- **How much reID is actually needed.** When a new tracklet starts, two same-alliance
  tracklets are already alive 52 % (red) / 56 % (blue) of the time — identity is then
  forced by elimination, no appearance required. One alive (binary choice) 35 % / 26 %;
  none alive (full 3-way) 11 % / 7 %. Elimination is only as good as the alive
  identities, so it must be solved jointly, not greedily. Blue shows 10 starts with
  ≥3 alive — residual duplicates beyond 0.7 m (truncated boxes), still unmerged.
- Bumpers dominate the chassis region and carry alliance colour, which is useless
  *within* an alliance — the earlier verification numbers were inflated by
  cross-alliance pairs and duplicate-derived positives.

**In the exact solver (ablation, wide-only):** motion-only 2 switches / 88 %;
+colour 1 / 95.5 %; +height no change at `W_H=1`, **3 switches at `W_H=3`**
(over-weighting a per-tracklet median that truncation and raised mechanisms move).
Colour and height are tiebreakers; the structure does the work.

**Reliable.** As a tiebreaker between motion-plausible candidates.

**Limitations.**
- Too weak to carry identity: at AUC 0.72–0.82 it is coin-flip-plus on the very
  pairs that matter (two same-alliance robots re-emerging after a gap).
- **Learned-appearance trap.** Any appearance model fitted from the *current*
  assignment reinforces whatever swap already happened (EM with appearance collapsed
  purity 59 → 17 %). `W_APP=0` in `fit6.py` is deliberate. Appearance may only be
  trained on labels that are certain by construction (concurrency).
- **Cross-camera transfer is unmeasured** and expected to be poor (different white
  balance, viewing angle Z 0.8 vs 5.1 m, scale). Do not assume the wide-view AUC
  holds across views.
- A spatial (grid) descriptor failed; the principled fix (yaw-conditioned
  appearance) needs OBB or keypoint labels.

---

## 10. Identity solver (`fuse_track.solve`; greedy `assign.assign` superseded)

**What (2026-09-04).** Per alliance, an **exact minimum-cost cover by K=3
node-disjoint, time-ordered paths** over the pure tracklets — min-cost flow
(`networkx.min_cost_flow`; DAG, so negative coverage rewards are fine). Edge
i→j exists only if j starts after i ends and `d ≤ VMAX·gap + 0.3 m` (hard
reachability). Cost = `W_M·(d/reach)² + W_G·unexplained-gap-seconds +
W_H·min((Δh/0.1)², 9) + W_C·colour`; each covered frame earns `W_R = 0.1`.
Concurrency exclusion is implicit in the time ordering, and elimination ("the
other two are alive, so this is the third") is what the global optimum does.

**Measured (same tracklets as the greedy solver's 13 switches / 84 %).**
- **1 ID switch, weighted purity 95.5 %, matched 98 %** (b1 100 %, b2 100 %, r1
  100 %, r2 76.7 %). The greedy solver on identical input: 13 / 84 %.
- Ablation: motion-only **2 / 88 %** → the algorithm, not the signals, was the
  problem. Colour adds one fix; `W_M`, `W_G` have no effect (the hard gate and the
  path structure carry it); `W_R` 0.03 strands tracklets (matched 85 %), 0.3 forces
  bad links (2 switches). Few weights matter, so there is little to overfit.
- The remaining switch: r2's last tracklet ends at frame 1934, its next starts at
  2021 (2.9 s), the untraced third red robot re-emerges at 1995, and **the red
  station lost r2 at 1937 too**. A true double re-emergence with no camera
  coverage, decided the wrong way by a 0.06 m height difference.

**Why the greedy solver failed (kept for the record).** Coordinate descent from a
seed; cost dominated by within-alliance colour (AUC 0.72–0.82) with `(d/reach)²`
against the nearest neighbour only — uninformative over the gaps that actually
occur (same-robot successor gaps p50 0.93 s, p90 2.57 s). Errors were block swaps
at the long-gap occlusion events; results moved 84 → 65 % on a cleaner input.

**Limitations.**
- Exactly K paths per alliance is assumed; a robot that never appears (disabled
  before the match) would force a phantom path.
- Path count is fixed but path *start/end* is free, so a robot unseen for the
  first 20 s is fine; a tracklet pool with a systematic false-positive source (a
  human, a game piece the detector likes) would be covered greedily by `W_R`.
- Alliance labels on wide tracklets were wrong 4/47 times (colour heuristic); a
  mislabelled tracklet enters the wrong pool. Not yet addressed.
- **Depends on detector recall through elimination.** On the older `fullB`
  detection set (2.76 dets/frame vs 3.84) the same solver gives **15 switches**.
  Gap lengths are similar; what differs is how often the other two robots are
  alive when a tracklet starts — red "forced" starts 29 % vs 52 %, fully open
  starts 27 % vs 11 %, frames with ≥2 same-alliance tracklets alive 54 % vs 80 %.
  Recall buys identity by feeding elimination, not by improving positions (which
  were flat across detectors, §2).
- Tuned and measured on one match (§12).

---

## 11. Smoother (`fit6.rts`)

**What.** Per identity, RTS smoothing of a constant-velocity model over the union
of its tracklets' observations, clipped to the field.

**Measured.** Where observed: err p50 0.09–0.17 m, p90 0.23–0.45 m; depth 0.07–0.16,
lateral 0.02–0.07 (depth 2–4× lateral, as predicted). Identities are *observed*
only 48–77 % of frames; the rest is interpolation.

**Reliable.** Position on observed frames; interpolation over short gaps on slow
robots.

**Limitations.** Asserts a position for every frame, including during long
occlusions — the evaluator cannot tell interpolation from measurement. Judge
unobserved stretches by the ellipse (`P`), never by the dot. Renderers cap the
ellipse (`LOST_M=1.5`, `CAP_M=3.0`) purely for legibility.

---

## 12. Evaluation (`src/eval_trace.py`, `out/trace.json`)

**What.** Hand trace at 5 Hz, one robot at a time, with explicit `occ` (occluded)
marks rather than guesses. Traced points are footprint centres (no corner
correction applied). Per frame, Hungarian-match traced robots to tracks of the same
alliance within `MATCH=1.0 m`; report matched %, err p50/p90, depth/lateral split,
ID switches, purity. This is the **only** evaluation that sees identity; the
box-label evaluation re-pairs swapped tracks and cannot.

**Coverage.** 1,156 annotations, 182 occluded (16 %); r1 393 pts, b2 316, r2 188,
**b1 only 77**; r3 and b3 untraced. Occluded stretches interpolate; 132 of 182 are
interpolatable.

**Limitations.**
- **Tuning leak.** Every tracker parameter was chosen on this match. 13 vs 11
  switches is within the noise of that choice. A second traced match is the only
  fix.
- **Coasting-duplicate flips.** Per-frame Hungarian happily swaps between an
  observed track and a coasting one within 1 m; the observed-only variant is the
  honest identity count.
- Tracer's own error is anisotropic the opposite way to the geometry: lateral
  (judging the left/right middle of a rotating robot) is the hard click, depth
  (bumper meets carpet) is crisp.
- One match, four robots. Statements like "same-alliance robots never < 0.99 m"
  are true *here* only.

---

## 13. Station detections (`src/detect_match.py`, `PANEL=`)

**What.** The same detector run on the red and blue station panels.
`out/dets_red_station.npz`: 10,718 detections (2.18/frame); `dets_blue_station.npz`:
15,584 (3.17/frame). 8,714 and 15,569 respectively project onto the field via a
floor ray.

**Measured (corner-corrected, 2026-09-04).** Of 178 wide-view occlusions with an
interpolated position, a station detection lies within 0.5 m **27 %** (was 8 %
uncorrected), 1.0 m 49 %, 2.0 m 66 %. But the baseline is low: on traced robots
the wide view *does* see, a station detection is within 1 m only **19 % (red) /
32 % (blue)** of the time, and inside its own 99 % Mahalanobis gate only 3 % / 25 %
(the red figure is the covariance under-model in §6). 100 % of traced points are
in-frame for both stations — so the ceiling is **station detector recall and
station-side occlusion**, not field of view.

**Limitations.**
- Station robot-on-robot occlusion at Z≈0.8 m is probably *worse* than in the wide
  view — different, not fewer. Whether station and wide occlusions are uncorrelated
  is unmeasured.
- No tracklets have been built from station detections yet; no cross-camera bias
  check has been done.
- The metric that matters for identity is not "within 1 m of truth" but "closer
  to the right tracklet than to any other", and ultimately **the same-robot gap
  distribution after adding station tracklets** (§7).

---

## 16. Fusion (`src/fuse_track.py`)

**What.** Tracklets from every view in one field-metre pool. Wide tracklets are
duplicate-merged (§8) and are the only nodes in the identity solve (§10). Station
tracklets are **link evidence**: a station tracklet that *vouches* for wide
tracklet A (≥ 8 frames overlap, 75th-percentile Mahalanobis < 3 under the summed
covariances) and for B, and was observed across the gap with no run > 20 frames,
**bridges** A→B — the link's gap penalty drops to the part the station missed.
A station tracklet that vouches for two wide tracklets alive at the same time is
ambiguous and dropped. After the solve, the bridging station measurements fill
the gap for position (information-weighted in the RTS).

**Measured.** Identity **identical to wide-only** (1 switch, same purities, same
position error); **+734 observed frames inside gaps** (per identity 15–241); red
gaps >1 s 53 → 47, blue 38 → 34; 10 bridges used per alliance, 2–4 ambiguous
station tracklets dropped.

**What did not work — and why it matters.**
- v1–v3 *merged* station tracklets into wide tracklets (union-find). A station
  ellipse is large enough to agree with two different wide tracklets, and
  union-find is transitive, so it produced a 310-frame tracklet spanning **r1 and
  r2**, and another spanning **b2 and r2** (cross-alliance). 3–11 switches. An
  impure tracklet is the one error the pipeline cannot recover from; station data
  must never be able to create one.
- Station **alliance labels are wrong 3/31** — a red robot's station tracklet in
  the blue pool is a phantom alive robot that breaks elimination. Station-only
  tracklets therefore never enter the solve.

**Limitations.** Gap-shortening is bounded by station recall (§13): most wide gaps
have no station tracklet to bridge them. Bridges are single-tracklet; two station
tracklets chained across one gap are not yet used. Station positions carry the
unexplained lateral bias (§6), so filled frames are less accurate than wide ones.

---

## 14. Visualisation (`src/render.py`, `render_cmp.py`)

Broadcast + top-down field; accuracy as Google-Maps-style ellipses; fading tails
that break at observation gaps (`MAXGAP=4`) drawn in `BANDS=5` alpha bands; two
independent columns colour-coded to the claiming track. Cosmetic only — nothing
downstream reads it.

---

## 15. Calibration tooling (`src/serve.py`, port 8765; `*.html`)

`calib.html` (four-view switcher, median/first-frame/enhanced plates, live re-solve
via `/solve`), `pair.html` (cross-view pairing, wide click → field → station),
`lines.html` (straight-line marking, live lens fit, bisection inverse — the
fixed-point inverse folded the grid), `trace.html` (5 Hz tracing, occluded mode).
Every save snapshots the previous file to `out/backup/`.

**Past bugs worth remembering when reading old data.**
- Field diagram was drawn rotated 180° while the mapping was not → every early
  click mirrored in X (`points_corrected.json` exists because of this).
- localStorage restore replaced `data` wholesale, hiding a newly added view.
- Hand-marked tag corners are ordered in image space, detector corners in tag
  space (§4).
- JS homography solver (power iteration) was wrong; replaced by a normalised 8×8
  direct solve, verified to 3e-14. `imgToGround` in JS matches Python to 4 dp.

---

## Integration hazards, in priority order

1. **Station covariance and bias** (§6): pixel noise is 15–24 px, not 4, and there
   is a ~0.15 m lateral bias of unknown origin. Any consumer of station positions
   must use `mcam.Cam.cov` (measured values), never the 4 px model.
2. **Never merge station tracklets into wide tracklets** (§16). Evidence, not
   measurements, until station recall and bias are fixed.
3. **Station alliance labels are unreliable** (3/31 wrong; wide 4/47) — never let
   a colour label alone place a tracklet in an alliance pool.
4. **Station recall is the ceiling** (§13): 19–32 % of visible traced robots.
   Improving fusion further means improving the detector on station panels.
5. **Appearance must never be learned from the assignment being optimised** (§9);
   height is a tiebreaker, not a signal (`W_H=3` → 3 switches).
6. **Red tracklets must not span the t=103.3 s cut** (§1; `mcam.red_view`).
7. **Partially occluded detections are unflagged garbage** (§2) — they are also
   the residual >0.7 m duplicates (blue: 10 tracklet starts with ≥3 alive).
8. **Everything is tuned and measured on one match** (§12) — though the exact
   solver has few weights that matter.
