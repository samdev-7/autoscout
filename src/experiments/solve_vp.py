"""Pose from vanishing points + position from ground points.

The failure mode all along was asking one least-squares problem to recover
intrinsics, rotation AND position from points that only constrain position.
Lines break that apart:
    focal + rotation  <- vanishing points of the X, Y and vertical families
    position          <- the ground correspondences, now 3 unknowns not 10
"""
import json, numpy as np, cv2, sys
from scipy.optimize import least_squares
sys.path.insert(0, "src")
import lines_solve as LS
import geom

W, H = 960, 540


def rotation_from_vps(vX, vY, K):
    Ki = np.linalg.inv(K)
    dx = Ki @ vX; dx /= np.linalg.norm(dx)
    dy = Ki @ vY; dy /= np.linalg.norm(dy)
    # force orthogonality (marking noise makes them not exactly perpendicular)
    dz = np.cross(dx, dy); dz /= np.linalg.norm(dz)
    dy = np.cross(dz, dx); dy /= np.linalg.norm(dy)
    R = np.c_[dx, dy, dz]                       # world axes expressed in camera frame
    if R[2, 2] < 0:                              # keep the field in front
        R = np.c_[-dx, -dy, dz]
    return R


def solve(view, k1=0.0):
    L = [l for l in json.load(open("out/lines_multi.json"))[view] if len(l["pts"]) >= 3]
    fam = {d: [l for l in L if l["dir"] == d] for d in ("X", "Y", "V")}
    cx, cy = W / 2, H / 2
    vX = LS.vanish(fam["X"], k1, cx, cy, 660.0)
    vY = LS.vanish(fam["Y"], k1, cx, cy, 660.0)
    ex = lambda h: np.array([h[0] / h[2], h[1] / h[2]])
    a = ex(vX) - [cx, cy]; b = ex(vY) - [cx, cy]
    f = float(np.sqrt(max(-(a @ b), 1.0)))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    R = rotation_from_vps(np.r_[ex(vX), 1.0], np.r_[ex(vY), 1.0], K)

    pts = json.load(open("out/points_multi.json"))[view]["pairs"]
    extra = []
    try:
        extra = json.load(open("out/pairs_xview.json")).get(view, [])
    except Exception:
        pass
    P = pts + extra
    P3 = np.array([[p["X"], p["Y"], 0.0] for p in P])
    P2 = np.array([[p["u"], p["v"]] for p in P])
    d = np.zeros(5); d[0] = k1
    rv, _ = cv2.Rodrigues(R)

    def res(t):
        q, _ = cv2.projectPoints(P3.reshape(-1, 1, 3), rv, t, K, d)
        return (q.reshape(-1, 2) - P2).ravel()

    best = None
    for t0 in ([0, 0, 6.], [0, 0, 10.], [2, 1, 8.], [-2, -1, 12.]):
        try:
            s = least_squares(res, np.array(t0, float), method="lm", max_nfev=20000).x
        except Exception:
            continue
        e = np.linalg.norm(res(s).reshape(-1, 2), axis=1)
        C = (-R.T @ s).ravel()
        if not (0.05 < C[2] < 5.0):
            continue
        if best is None or np.median(e) < best[0]:
            best = (float(np.median(e)), s, e, C)
    return dict(f=f, K=K, R=R, rv=rv, k1=k1, best=best, n=len(P), P=P)


if __name__ == "__main__":
    view = sys.argv[1] if len(sys.argv) > 1 else "red_station"
    r = solve(view)
    print(f"{view}: focal {r['f']:.1f} from vanishing points, rotation fixed by them")
    if not r["best"]:
        print("  no plausible position"); sys.exit()
    med, t, e, C = r["best"]
    print(f"  position solved from {r['n']} ground points: only 3 unknowns")
    print(f"  residual median {med:.2f} px  mean {e.mean():.2f}  max {e.max():.2f}")
    print(f"  camera centre  X {C[0]:6.2f}  Y {C[1]:6.2f}  Z {C[2]:5.2f} m")
    inl = e < 15
    print(f"  points within 15 px: {inl.sum()}/{len(e)}")
    for i, p in enumerate(r["P"]):
        print(f"    ({p['X']:6.2f},{p['Y']:5.2f})  resid {e[i]:7.1f} px"
              + ("" if inl[i] else "   <-- outlier"))
    np.save(f"out/cam_vp_{view}.npy",
            np.r_[r["f"], W / 2, H / 2, r["rv"].ravel(), t, r["k1"]])
