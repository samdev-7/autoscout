"""Full-match trajectory summary: six paths on the field + coverage timeline."""
import numpy as np, cv2, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom
from geom import LEN, WID

W = 1600
FB = cv2.rotate(cv2.imread("out/field_box.png"), cv2.ROTATE_180)
FH = int(round(FB.shape[0] * W / FB.shape[1]))
FB = cv2.resize(FB, (W, FH), interpolation=cv2.INTER_AREA)
COLS = [(50, 50, 240), (60, 150, 255), (120, 60, 200),
        (240, 120, 40), (225, 200, 60), (200, 90, 130)]
NAMES = ["red 1", "red 2", "red 3", "blue 1", "blue 2", "blue 3"]


def f2px(P):
    P = np.atleast_2d(np.asarray(P, float))
    return np.c_[W * (1 - P[:, 0] / LEN), FH * (P[:, 1] / WID)]


tr = list(np.load("out/tracks_em.npy", allow_pickle=True))
nf = len(tr[0]["xy"])
BH, PAD = 26, 10
H = FH + PAD + len(tr) * BH + 40
img = np.full((H, W, 3), 255, np.uint8)
img[:FH] = cv2.addWeighted(FB, 0.45, np.full_like(FB, 255), 0.55, 0)

for k, t in enumerate(tr):
    p = f2px(t["xy"]).astype(np.int32)
    obs = t["obs"]
    # draw only stretches we actually observed; interpolated gaps stay blank
    s = None
    for f in range(nf):
        if obs[f] and s is None:
            s = f
        elif not obs[f] and s is not None:
            if f - s > 2:
                cv2.polylines(img, [p[s:f]], False, COLS[k], 2, cv2.LINE_AA)
            s = None
    if s is not None:
        cv2.polylines(img, [p[s:]], False, COLS[k], 2, cv2.LINE_AA)

y0 = FH + PAD
for k, t in enumerate(tr):
    y = y0 + k * BH
    cv2.rectangle(img, (10, y + 4), (30, y + 18), COLS[k], -1)
    cv2.putText(img, NAMES[k], (38, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)
    x0, wpx = 120, W - 300
    cv2.rectangle(img, (x0, y + 4), (x0 + wpx, y + 18), (232, 232, 232), -1)
    obs = t["obs"]
    for f in range(0, nf, 4):
        if obs[f:f + 4].any():
            a = x0 + int(wpx * f / nf)
            cv2.rectangle(img, (a, y + 4), (a + max(1, wpx * 4 // nf), y + 18),
                          COLS[k], -1)
    d = np.linalg.norm(np.diff(t["xy"], axis=0), axis=1).sum()
    cv2.putText(img, f"{100*obs.mean():4.1f}% seen   {d:5.0f} m",
                (x0 + wpx + 12, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (60, 60, 60), 1, cv2.LINE_AA)

cv2.putText(img, "observed path only (interpolated gaps not drawn)  -  "
            "bars: when each robot was seen across the 164 s match",
            (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)
cv2.imwrite("out/summary.png", img)
print("wrote out/summary.png", img.shape)
for k, t in enumerate(tr):
    sp = np.hypot(t["v"][:, 0], t["v"][:, 1])
    print(f"  {NAMES[k]:7} seen {100*t['obs'].mean():5.1f}%  "
          f"path {np.linalg.norm(np.diff(t['xy'],axis=0),axis=1).sum():6.1f} m  "
          f"speed p50 {np.median(sp):.2f} p95 {np.percentile(sp,95):.2f} m/s")
