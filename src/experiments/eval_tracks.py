"""Evaluate tracks against the 83 hand-labelled frames.

This is the first measurement of the thing that actually matters.  Everything
before it was per-frame detection on isolated frames; a scouting pipeline needs
a robot to have a position at every instant, which is a track-level question.

Ground-truth boxes are converted to field positions the same way detections are,
so this measures the tracker, not the projection.
"""
import numpy as np, json, sys, os
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

lab = json.load(open("out/labels.json"))
man = {m["file"]: m["t"] for m in json.load(open("out/frames.json"))["frames"]}
tr = list(np.load(os.environ.get("TRK","out/tracks_em.npy"), allow_pickle=True))
det = np.load("out/dets_fullB.npz", allow_pickle=True)["dets"]
dxy = np.load("out/dets_xy.npy")
nf = len(tr[0]["xy"])
AL = {"R": 1, "B": 2}

THR = [0.5, 1.0, 1.5, 2.0]
res = {t: dict(m=0, g=0, p=0) for t in THR}
errs, alok, altot = [], 0, 0
draw_g = draw_d = 0
for fn, boxes in lab["frames"].items():
    if fn not in man:
        continue
    f = int(round((man[fn] - 8.0) * 30))
    if not (0 <= f < nf):
        continue
    # labels are in full-panel (0..490) coords; detections are band-cropped.
    G = np.array([[b["x0"], b["y0"] - geom.BAND0, b["x1"], b["y1"] - geom.BAND0]
                  for b in boxes], float)
    gal = np.array([AL[b["c"]] for b in boxes])
    gxy = geom.box_to_field(G)
    live = [k for k in range(6) if tr[k]["obs"][max(f-60,0):f+61].any()]
    T = np.array([tr[k]["xy"][f] for k in live])
    tal = np.array([1 if tr[k]["alliance"] == "red" else 2 for k in live])
    D = np.linalg.norm(gxy[:, None, :] - T[None, :, :], axis=2)
    D[gal[:, None] != tal[None, :]] += 100.0          # alliance must agree
    ri, ci = linear_sum_assignment(D)
    for thr in THR:
        m = sum(1 for i, j in zip(ri, ci) if D[i, j] < thr)
        res[thr]["m"] += m; res[thr]["g"] += len(gxy); res[thr]["p"] += len(T)
    for i, j in zip(ri, ci):
        if D[i, j] < 2.0:
            errs.append(D[i, j]); altot += 1; alok += (gal[i] == tal[j])
    # per-frame detection recall on the same frames, for comparison
    dm = det[:, 0].astype(int) == f
    if dm.any():
        Dd = np.linalg.norm(gxy[:, None, :] - dxy[dm][None, :, :], axis=2)
        r2, c2 = linear_sum_assignment(Dd)
        draw_d += sum(1 for i, j in zip(r2, c2) if Dd[i, j] < 1.0)
    draw_g += len(gxy)

print(f"evaluated {len(lab['frames'])} labelled frames, {res[1.0]['g']} ground-truth robots\n")
print(f"{'match within':>13} | {'track recall':>12} {'precision':>10}")
for thr in THR:
    r = res[thr]
    print(f"{thr:11.1f} m | {100*r['m']/r['g']:11.1f}% {100*r['m']/r['p']:9.1f}%")
errs = np.array(errs)
print(f"\nposition error of matched tracks (m): p50 {np.median(errs):.2f}  "
      f"p90 {np.percentile(errs,90):.2f}  p95 {np.percentile(errs,95):.2f}")
print(f"alliance agreement on matches: {100*alok/max(altot,1):.1f}%")
print(f"\nfor comparison, raw per-frame DETECTION recall within 1.0 m: "
      f"{100*draw_d/draw_g:.1f}%")
print(f"                    tracker recall within 1.0 m: "
      f"{100*res[1.0]['m']/res[1.0]['g']:.1f}%")
