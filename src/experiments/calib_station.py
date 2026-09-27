"""Calibrate an alliance-station panel from AprilTags alone.

The wide view needed hand-marked carpet points because its three visible tags
all face the same way, so their corners are exactly coplanar and pose/intrinsics
are not separable.  The station views see tags on surfaces at two different
orientations, so the corner cloud has real 3D extent and PnP is well posed.

The camera is locked off, so corner observations are averaged over the whole
match before fitting -- that drives detection noise far below one pixel.
"""
import av, cv2, csv, numpy as np, sys, os
from scipy.optimize import least_squares

IN, SZ = 0.0254, 0.1651
PANELS = {"red_station": (540, 1080, 0, 960), "blue_station": (540, 1080, 960, 1920)}
T = {}
with open("data/field/2026-rebuilt-welded.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        T[int(r["ID"])] = r


def corners3d(i):
    r = T[i]
    c = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])]) * IN
    rz, rx = np.radians(float(r["Z-Rotation"])), np.radians(float(r["X-Rotation"]))
    right = np.array([-np.sin(rz), np.cos(rz), 0.0])
    up = np.array([np.sin(rx) * np.cos(rz), np.sin(rx) * np.sin(rz), np.cos(rx)])
    h = SZ / 2
    return np.array([c - right * h + up * h, c + right * h + up * h,
                     c + right * h - up * h, c - right * h - up * h])


def observe(panel, step=15):
    y0, y1, x0, x1 = PANELS[panel]
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    pa = cv2.aruco.DetectorParameters()
    pa.adaptiveThreshWinSizeMin = 3; pa.adaptiveThreshWinSizeMax = 45
    pa.adaptiveThreshWinSizeStep = 4; pa.minMarkerPerimeterRate = 0.008
    pa.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(d, pa)
    acc = {}
    c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
    n = 0
    for f in c.decode(s):
        if f.time is None or f.time < 10:
            continue
        if f.time > 170:
            break
        n += 1
        if n % step:
            continue
        g = cv2.cvtColor(f.to_ndarray(format="rgb24")[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        co, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        for cc, i in zip(co, ids.ravel()):
            acc.setdefault(int(i), []).append(cc.reshape(4, 2))
    c.close()
    return {i: (np.median(v, 0), len(v)) for i, v in acc.items() if len(v) >= 8}


def fit(P3, P2, w, h):
    def unpack(p):
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
        dd = np.zeros(5); dd[0] = p[9]
        return K, p[3:6], p[6:9], dd
    def res(p):
        K, rv, tv, dd = unpack(p)
        q, _ = cv2.projectPoints(P3.reshape(-1, 1, 3), rv, tv, K, dd)
        return (q.reshape(-1, 2) - P2).ravel()
    best = None
    for f0 in (600, 900, 1300, 1800, 2400):
        K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]])
        ok, rv, tv = cv2.solvePnP(P3.reshape(-1, 1, 3), P2.reshape(-1, 1, 2), K0,
                                  np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        p0 = np.r_[f0, w / 2, h / 2, rv.ravel(), tv.ravel(), 0.0]
        try:
            s = least_squares(res, p0, method="lm", max_nfev=40000).x
        except Exception:
            continue
        e = np.linalg.norm(res(s).reshape(-1, 2), axis=1)
        if best is None or e.mean() < best[1]:
            best = (s, e.mean(), e)
    return best


if __name__ == "__main__":
    for panel in ("red_station", "blue_station"):
        y0, y1, x0, x1 = PANELS[panel]
        w, h = x1 - x0, y1 - y0
        obs = observe(panel)
        ids = sorted(obs)
        print(f"\n=== {panel} ({w}x{h}) — {len(ids)} tags: {ids}")
        P3 = np.vstack([corners3d(i) for i in ids])
        # Corner ordering has EIGHT candidates, not four: 4 cyclic shifts x 2
        # windings.  And low reprojection error on a near-coplanar tag set does
        # not mean the pose extrapolates -- so each candidate is also required
        # to place the field where a floor-level camera would actually see it.
        best, bs = None, None
        for flip in (False, True):
            for shift in range(4):
                P2 = np.vstack([np.roll(obs[i][0][::-1] if flip else obs[i][0],
                                        shift, axis=0) for i in ids])
                r = fit(P3, P2, w, h)
                if not r:
                    continue
                sc = r[0]
                K = np.array([[sc[0], 0, sc[1]], [0, sc[0], sc[2]], [0, 0, 1]])
                dd = np.zeros(5); dd[0] = sc[9]
                g = np.array([[x, y, 0.0] for x in np.linspace(1, 15.5, 8)
                              for y in np.linspace(1, 7, 5)])
                q, _ = cv2.projectPoints(g.reshape(-1, 1, 3), sc[3:6], sc[6:9], K, dd)
                q = q.reshape(-1, 2)
                inside = ((q[:, 0] > 0) & (q[:, 0] < w) & (q[:, 1] > 0) & (q[:, 1] < h)).mean()
                Rm, _ = cv2.Rodrigues(sc[3:6]); C = (-Rm.T @ sc[6:9]).ravel()
                plausible = 0.0 < C[2] < 4.0 and -6 < C[0] < 23 and -6 < C[1] < 20
                if inside < 0.35 or not plausible:
                    continue
                if best is None or r[1] < best[1]:
                    best, bs = r, (flip, shift, inside)
        if best is None:
            print("  FAILED"); continue
        s, mean_e, e = best
        K = np.array([[s[0], 0, s[1]], [0, s[0], s[2]], [0, 0, 1]])
        R, _ = cv2.Rodrigues(s[3:6]); C = (-R.T @ s[6:9]).ravel()
        print(f"  corner order flip={bs[0]} shift={bs[1]};  {100*bs[2]:.0f}% of a field grid "
              f"projects in-frame;  reprojection mean {mean_e:.2f} px  max {e.max():.2f}")
        print(f"  focal {s[0]:.0f} px   principal ({s[1]:.0f},{s[2]:.0f})   k1 {s[9]:+.4f}")
        print(f"  camera centre  X {C[0]:6.2f}  Y {C[1]:6.2f}  Z {C[2]:5.2f} m")
        np.save(f"out/cam_{panel}.npy", s)
