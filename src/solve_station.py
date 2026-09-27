"""Full 3D calibration of a panel from carpet points + AprilTag corners.

The tags ARE measurements -- 4 corners each, with known 3D position and
orientation.  What they cannot do alone is separate focal length from distance,
because every tag sits within ~0.2 m of the same height and a near-planar target
makes those two trade off.  A handful of carpet points at Z=0 supplies the
vertical baseline that breaks it, after which the 24 tag corners carry most of
the geometric weight.

Corner ordering is not searched: it was pinned by reproducing the detected
corners in the WIDE view (validated camera) to 1.24 px.
"""
import numpy as np, cv2, csv, json, os, sys
from scipy.optimize import least_squares

IN, SZ = 0.0254, 0.1651
SIGN, SHIFT, FLIP = +1, 0, True          # verified against the wide view
PAN = {"red_station": (960, 540), "blue_station": (960, 540), "wide": (1920, 490)}
T = {}
with open("data/field/2026-rebuilt-welded.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        T[int(r["ID"])] = r


def corners3d(i):
    r = T[i]
    c = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])]) * IN
    rz, rx = np.radians(float(r["Z-Rotation"])), np.radians(float(r["X-Rotation"]))
    n = np.array([np.cos(rz), np.sin(rz), 0.0])
    up = np.array([0, 0, 1.0])
    right = SIGN * np.cross(n, up)
    h = SZ / 2
    return np.array([c - right*h + up*h, c + right*h + up*h,
                     c + right*h - up*h, c - right*h - up*h])


def build(view):
    pts = json.load(open("out/points_multi.json"))[view]
    tv = {g["id"]: g for g in json.load(open("out/tags_views.json"))[view]}
    P3, P2, kind = [], [], []
    for p in pts["pairs"]:
        P3.append([p["X"], p["Y"], 0.0]); P2.append([p["u"], p["v"]]); kind.append("carpet")
    for k, c in pts["tagMarks"].items():
        i = int(k)
        if i not in T:
            continue
        obs = np.array(c, float)
        if FLIP:
            obs = obs[::-1]
        obs = np.roll(obs, SHIFT, axis=0)
        for a, b in zip(corners3d(i), obs):
            P3.append(a); P2.append(b); kind.append(f"tag{i}")
    return np.array(P3), np.array(P2), kind


def fit(P3, P2, w, h, nk=1, wts=None):
    def unpack(p):
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
        d = np.zeros(5); d[0] = p[9] if nk else 0
        return K, p[3:6], p[6:9], d
    W = np.ones(len(P3)) if wts is None else np.asarray(wts, float)
    def res(p):
        K, rv, tv, d = unpack(p)
        q, _ = cv2.projectPoints(P3.reshape(-1, 1, 3), rv, tv, K, d)
        return ((q.reshape(-1, 2) - P2) * W[:, None]).ravel()
    best = None
    for f0 in (400, 550, 700, 900, 1200, 1600):
        K0 = np.array([[f0, 0, w/2], [0, f0, h/2], [0, 0, 1]], float)
        ok, rv, tv = cv2.solvePnP(P3.reshape(-1, 1, 3), P2.reshape(-1, 1, 2), K0,
                                  np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        p0 = np.r_[f0, w/2, h/2, rv.ravel(), tv.ravel(), 0.0]
        try:
            s = least_squares(res, p0, method="lm", max_nfev=60000).x
        except Exception:
            continue
        e = np.linalg.norm((res(s).reshape(-1, 2) / W[:, None]), axis=1)
        R, _ = cv2.Rodrigues(s[3:6]); C = (-R.T @ s[6:9]).ravel()
        if not (0.1 < C[2] < 4.0):        # a broadcast camera is not underground
            continue
        if best is None or e.mean() < best[1]:
            best = (s, e.mean(), e, C)
    return best


if __name__ == "__main__":
    for view in (sys.argv[1:] or ["red_station", "blue_station"]):
        w, h = PAN[view]
        P3, P2, kind = build(view)
        nc = sum(1 for k in kind if k == "carpet")
        print(f"\n=== {view}: {len(P3)} points ({nc} carpet + {len(P3)-nc} tag corners)")
        if len(P3) < 8:
            print("  too few"); continue
        # Carpet points are outnumbered 8:1 by tag corners, yet they are the ONLY
        # source of vertical baseline -- without weighting they get sacrificed
        # (carpet residual ran 3x the tag residual) and the degeneracy returns.
        WC = float(os.environ.get("WCARPET", "6"))
        wts = np.array([WC if k == "carpet" else 1.0 for k in kind])
        b = fit(P3, P2, w, h, wts=wts)
        if not b:
            print("  no plausible solution"); continue
        s, m, e, C = b
        print(f"  reprojection mean {m:.2f} px  max {e.max():.2f} px")
        print(f"  carpet residual {np.mean([e[i] for i,k in enumerate(kind) if k=='carpet']):.2f} px"
              f"   tag residual {np.mean([e[i] for i,k in enumerate(kind) if k!='carpet']):.2f} px")
        print(f"  focal {s[0]:.0f}  principal ({s[1]:.0f},{s[2]:.0f})  k1 {s[9]:+.4f}")
        print(f"  camera centre  X {C[0]:6.2f}  Y {C[1]:6.2f}  Z {C[2]:5.2f} m")
        np.save(f"out/cam_{view}.npy", s)
