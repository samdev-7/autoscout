"""Render the tracked match: broadcast panel + top-down field, with tails.

Tails fade over TAIL frames and BREAK at observation gaps, so a stretch the
smoother interpolated never draws a tail that implies we saw it.  Fading is done
in a few discrete alpha bands rather than per segment -- 6 blends per frame
instead of 270, which is the difference between rendering in minutes and hours.
"""
import cv2, numpy as np, av, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom
from geom import LEN, WID, BAND0, BAND1

W = 1920
FB = cv2.imread("out/field_box.png")
FH = int(round(FB.shape[0] * W / FB.shape[1]))
FB = cv2.resize(FB, (W, FH), interpolation=cv2.INTER_AREA)
PH = BAND1 - BAND0
SX, SY = W / LEN, FH / WID
TAIL, BANDS, MAXGAP = 45, 6, 4
UNSURE, LOST = 22, 300     # never hide a track: a wide ellipse is more
                           # useful than nothing, so only drop it after 10 s
MAXAX = 4.0                # m; let the ellipse grow to show low confidence
COLS = [(50, 50, 240), (60, 150, 255), (120, 60, 200),
        (240, 120, 40), (225, 200, 60), (200, 90, 130)]


def f2px(P):
    P = np.atleast_2d(np.asarray(P, float))
    return np.c_[W * (1 - P[:, 0] / LEN), FH * (P[:, 1] / WID)]


def tail_pts(tr, f):
    """Positions over the current continuous observed run, newest last."""
    obs = tr["obs"]
    if not obs[max(f - MAXGAP, 0):f + 1].any():
        return None
    s, gap = f, 0
    while s > 0 and f - s < TAIL:
        s -= 1
        gap = 0 if obs[s] else gap + 1
        if gap > MAXGAP:
            s += gap; break
    return tr["xy"][s:f + 1] if f - s >= 2 else None


def fade(img, segs):
    """segs: list of (colour, points, age0..1). Blend in discrete alpha bands."""
    for b in range(BANDS):
        lo, hi = b / BANDS, (b + 1) / BANDS
        ov, any_ = img.copy(), False
        for c, p, a in segs:
            m = (a >= lo) & (a < hi)
            if m.sum() < 2:
                continue
            cv2.polylines(ov, [p[m].astype(np.int32)], False, c,
                          max(1, int(2 + 3 * (1 - (lo + hi) / 2))), cv2.LINE_AA)
            any_ = True
        if any_:
            cv2.addWeighted(ov, 0.15 + 0.75 * (1 - (lo + hi) / 2), img,
                            1 - (0.15 + 0.75 * (1 - (lo + hi) / 2)), 0, img)
    return img


def gapdist(obs):
    """Frames to the nearest observed frame, in either direction."""
    n = len(obs); big = n + 1
    d = np.where(obs, 0, big)
    for i in range(1, n):
        d[i] = min(d[i], d[i - 1] + 1)
    for i in range(n - 2, -1, -1):
        d[i] = min(d[i], d[i + 1] + 1)
    return d


def render(tracks, t0, t1, out, dets=None, sigma=2.0, scale=1.0):
    for tr in tracks:
        tr["gap"] = gapdist(tr["obs"])
    c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
    OW, OH = int(W * scale) // 2 * 2, int((PH + FH) * scale) // 2 * 2
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), 30, (OW, OH))
    if not vw.isOpened():
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (OW, OH))
    n = 0
    for fr in c.decode(s):
        if fr.time is None or fr.time < t0:
            continue
        if fr.time > t1:
            break
        f = int(round((fr.time - 8.0) * 30))
        if f < 0 or f >= len(tracks[0]["xy"]):
            continue
        top = cv2.cvtColor(fr.to_ndarray(format="rgb24")[BAND0:BAND1],
                           cv2.COLOR_RGB2BGR)
        bot = FB.copy()

        if dets is not None:
            m = dets[:, 0].astype(int) == f
            for b in dets[m]:
                x0, y0, x1, y1 = b[3:7].astype(int)
                cv2.rectangle(top, (x0, y0), (x1, y1), (200, 200, 200), 1)

        segs_b, segs_t = [], []
        for k, tr in enumerate(tracks):
            p = tail_pts(tr, f)
            if p is None or len(p) < 2:
                continue
            a = np.linspace(0, 1, len(p))            # 0 = oldest
            segs_b.append((COLS[k], f2px(p), a))
            q = geom.field_to_img(p); q[:, 1] -= BAND0
            segs_t.append((COLS[k], q, a))
        fade(bot, segs_b); fade(top, segs_t)

        for k, tr in enumerate(tracks):
            P = tr["xy"][f]
            if not np.isfinite(P).all():
                continue
            g = tr["gap"][f]
            if g > LOST:                     # not seen for 2 s -- claim nothing
                continue
            col = COLS[k]
            solid = g <= MAXGAP
            px, py = f2px(P)[0]
            S = tr["P"][f][:2, :2] if tr["P"] is not None else np.eye(2) * .01
            w, V = np.linalg.eigh(S)
            ax = np.minimum(np.sqrt(np.maximum(w, 1e-9)) * sigma, MAXAX)
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            cv2.ellipse(bot, (int(px), int(py)),
                        (max(int(ax[1] * SX), 3), max(int(ax[0] * SY), 3)),
                        ang, 0, 360, col, -1 if False else 1, cv2.LINE_AA)
            r = 8 if solid else 6
            cv2.circle(bot, (int(px), int(py)), r, col, -1 if solid else 2, cv2.LINE_AA)
            cv2.circle(bot, (int(px), int(py)), r, (255, 255, 255), 1, cv2.LINE_AA)
            u, v = geom.field_to_img(P[None])[0]; v -= BAND0
            cv2.circle(top, (int(u), int(v)), 6, col, -1 if solid else 2, cv2.LINE_AA)
            cv2.circle(top, (int(u), int(v)), 6, (255, 255, 255), 1, cv2.LINE_AA)

        img = np.vstack([top, bot])
        cv2.line(img, (0, PH), (W, PH), (40, 40, 40), 3)
        cv2.rectangle(img, (12, 12), (232, 50), (0, 0, 0), -1)
        cv2.putText(img, f"t = {fr.time:6.2f} s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(cv2.resize(img, (OW, OH), interpolation=cv2.INTER_AREA)
                 if scale != 1.0 else img); n += 1
        if n % 300 == 0:
            print(f"  {n} frames  t={fr.time:.1f}", flush=True)
    vw.release(); c.close()
    print(f"wrote {out}  ({n} frames)")


if __name__ == "__main__":
    tr = list(np.load("out/tracks_em.npy", allow_pickle=True))
    d = np.load("out/dets_fullB.npz", allow_pickle=True)["dets"]
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 55.0
    b = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
    render(tr, a, b, sys.argv[3] if len(sys.argv) > 3 else "out/track_clip.mp4",
           dets=d, scale=float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
