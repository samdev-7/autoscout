"""Side-by-side broadcast + top-down field visualisation.

The field diagram is used unrotated.  Verified against the calibrated camera
model by locating the red and blue structures in both: the video puts red at
field X ~12.2 m and blue at ~4.0 m, and the unrotated diagram agrees (12.1 /
4.5) while a 180 deg rotation mirrors them (4.5 / 12.1).  A detection therefore sits
at roughly the same horizontal position in both panels, which makes the two
views readable as one picture.

Accuracy is drawn as an ELLIPSE, not a circle.  Lateral error is ~3 cm and depth
error ~13-25 cm, so a circle would have to use the worst axis and would overstate
sideways precision by about 6x.  The ellipse also makes the anisotropy visible.
"""
import cv2, numpy as np, sys, os, av
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom
from geom import LEN, WID, BAND0, BAND1

W = 1920
FB = cv2.imread("out/field_box.png")
FH = int(round(FB.shape[0] * W / FB.shape[1]))
FB = cv2.resize(FB, (W, FH), interpolation=cv2.INTER_AREA)
PH = BAND1 - BAND0
SX, SY = W / LEN, FH / WID
COL = {0: (150, 150, 150), 1: (60, 60, 235), 2: (235, 120, 40)}   # BGR


def f2px(XY):
    P = np.atleast_2d(np.asarray(XY, float))
    return np.c_[W * (1 - P[:, 0] / LEN), FH * (P[:, 1] / WID)]


def draw(frame_bgr, dets, xy, trails=None, t=None, sigma=2.0):
    """dets: (N,9) rows for this frame; xy: (N,2) field positions."""
    top = frame_bgr.copy()
    bot = FB.copy()
    ov = bot.copy()

    if len(dets):
        S = geom.cov(xy)
        for k in range(len(dets)):
            c = COL[int(dets[k, 8])]
            x0, y0, x1, y1 = dets[k, 3:7].astype(int)
            cv2.rectangle(top, (x0, y0), (x1, y1), c, 2)
            cv2.circle(top, ((x0 + x1) // 2, y1), 4, c, -1)        # floor contact used
            px, py = f2px(xy[k])[0]
            w, V = np.linalg.eigh(S[k])
            ax = np.sqrt(np.maximum(w, 1e-9)) * sigma
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            cv2.ellipse(ov, (int(px), int(py)),
                        (max(int(ax[1] * SX), 2), max(int(ax[0] * SY), 2)),
                        ang, 0, 360, c, -1)
        cv2.addWeighted(ov, 0.30, bot, 0.70, 0, bot)
        for k in range(len(dets)):
            c = COL[int(dets[k, 8])]
            px, py = f2px(xy[k])[0]
            w, V = np.linalg.eigh(S[k])
            ax = np.sqrt(np.maximum(w, 1e-9)) * sigma
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            cv2.ellipse(bot, (int(px), int(py)),
                        (max(int(ax[1] * SX), 2), max(int(ax[0] * SY), 2)),
                        ang, 0, 360, c, 1, cv2.LINE_AA)
            cv2.circle(bot, (int(px), int(py)), 6, c, -1, cv2.LINE_AA)
            cv2.circle(bot, (int(px), int(py)), 6, (255, 255, 255), 1, cv2.LINE_AA)

    if trails:
        for c, pts in trails:
            if len(pts) > 1:
                cv2.polylines(bot, [f2px(pts).astype(np.int32)], False, c, 2, cv2.LINE_AA)

    out = np.vstack([top, bot])
    cv2.line(out, (0, PH), (W, PH), (40, 40, 40), 3)
    if t is not None:
        cv2.rectangle(out, (12, 12), (250, 52), (0, 0, 0), -1)
        cv2.putText(out, f"t = {t:6.2f} s", (22, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{sigma:.0f}-sigma accuracy", (W - 340, PH + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return out


def frame_at(t):
    c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
    for f in c.decode(s):
        if f.time is not None and f.time >= t:
            a = f.to_ndarray(format="rgb24")[BAND0:BAND1]
            c.close(); return cv2.cvtColor(a, cv2.COLOR_RGB2BGR), float(f.time)
    c.close(); raise SystemExit("time past end")


if __name__ == "__main__":
    d = np.load("out/dets_fullB.npz", allow_pickle=True)["dets"]
    xy = np.load("out/dets_xy.npy")
    for t in (float(sys.argv[1]) if len(sys.argv) > 1 else 60.0,):
        bgr, tt = frame_at(t)
        m = np.abs(d[:, 1] - tt) < 1e-3
        img = draw(bgr, d[m], xy[m], t=tt)
        p = f"out/viz_t{int(tt)}.png"
        cv2.imwrite(p, img)
        print(f"{p}  {img.shape[1]}x{img.shape[0]}  {m.sum()} detections at t={tt:.2f}")
