"""Full station calibration: lines -> lens, vanishing points -> focal, points -> pose.

Order matters.  Solving everything at once lets the optimiser hide model error in
whichever parameter is loosest (it put k1 at -0.3 when the truth was +0.4).  Each
stage here is constrained by data that does not depend on the later stages.
"""
import json, numpy as np, cv2, sys, itertools
from scipy.optimize import minimize, least_squares
sys.path.insert(0, "src")
import lines_solve as LS, solve_station as S, geom

W, H = 960, 540


def undist_pts(P, k1, k2, cx, cy, f):
    P = np.atleast_2d(np.asarray(P, float))
    x = (P[:, 0] - cx) / f; y = (P[:, 1] - cy) / f
    r2 = x * x + y * y; s = 1 + k1 * r2 + k2 * r2 * r2
    return np.c_[cx + x * s * f, cy + y * s * f]


def redist(q, k1, k2, cx, cy, f):
    X = (q[0] - cx) / f; Y = (q[1] - cy) / f; R = np.hypot(X, Y)
    if R < 1e-12:
        return (cx, cy)
    g = lambda r: r * (1 + k1 * r * r + k2 * r ** 4)
    lo, hi = 0.0, max(R, 1e-3)
    while g(hi) < R and hi < 50:
        hi *= 1.5
    if g(hi) < R:
        return (np.nan, np.nan)
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if g(m) < R else (lo, m)
    r = 0.5 * (lo + hi)
    return (cx + X * (r / R) * f, cy + Y * (r / R) * f)


def straight(p, L, f):
    return float(np.sqrt(np.mean([
        LS.resid(undist_pts(np.array(l["pts"]), p[0], p[1], p[2], p[3], f)) ** 2 for l in L])))


def fit_lens(L, f):
    best = None
    for c0 in itertools.product((0.35, 0.5, 0.65), (0.3, 0.5, 0.7)):
        r = minimize(lambda p: straight(p, L, f), [0.2, 0.2, W * c0[0], H * c0[1]],
                     method="Nelder-Mead",
                     bounds=[(-0.9, 1.2), (-0.9, 1.5), (0.15 * W, 0.85 * W), (0.15 * H, 0.85 * H)],
                     options=dict(maxiter=9000, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def vps(L, p, f):
    out = {}
    for d in ("X", "Y", "V"):
        ls = [l for l in L if l["dir"] == d]
        if len(ls) < 2:
            continue
        rows = []
        for l in ls:
            Q = undist_pts(np.array(l["pts"]), p[0], p[1], p[2], p[3], f)
            m = Q.mean(0); _, _, V = np.linalg.svd(Q - m); dd = V[0]
            n = np.array([-dd[1], dd[0]]); rows.append([n[0], n[1], -(n @ m)])
        _, _, V = np.linalg.svd(np.array(rows)); h = V[-1]
        out[d] = np.array([h[0] / h[2], h[1] / h[2]])
    return out


def run(view):
    L = [l for l in json.load(open("out/lines_multi.json"))[view] if len(l["pts"]) >= 3]
    print(f"=== {view}: {len(L)} lines "
          f"({sum(1 for l in L if l['dir']=='X')}X {sum(1 for l in L if l['dir']=='Y')}Y "
          f"{sum(1 for l in L if l['dir']=='V')}V)")
    raw = straight([0, 0, W / 2, H / 2], L, 675.0)
    f = 675.0
    for it in range(6):
        p, rms = fit_lens(L, f)
        v = vps(L, p, f)
        if "X" in v and "Y" in v:
            a = v["X"] - p[2:]; b = v["Y"] - p[2:]; f2 = -(a @ b)
            nf = float(np.sqrt(f2)) if f2 > 0 else f
        else:
            nf = f
        if abs(nf - f) < 0.5:
            f = nf; break
        f = 0.5 * f + 0.5 * nf
    k1, k2, cx, cy = p
    print(f"  lens: k1 {k1:+.3f} k2 {k2:+.3f} pp ({cx:.0f},{cy:.0f})  "
          f"straightness {raw:.2f} -> {rms:.2f} px  ({raw/max(rms,1e-9):.1f}x)")
    print(f"  focal from vanishing points: {f:.1f}")
    # leave-one-out on the lines
    loo = []
    for i in range(len(L)):
        pi, _ = fit_lens([L[j] for j in range(len(L)) if j != i], f)
        loo.append(straight(pi, [L[i]], f))
    print(f"  line LOO: median {np.median(loo):.3f} px  max {max(loo):.3f} px")

    P3, P2, kind = S.build(view)
    P2u = undist_pts(P2, k1, k2, cx, cy, f)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]); Z = np.zeros(5)
    ok, rv0, tv0 = cv2.solvePnP(P3.reshape(-1, 1, 3), P2u.reshape(-1, 1, 2), K, Z,
                                flags=cv2.SOLVEPNP_SQPNP)
    res = lambda q: (cv2.projectPoints(P3.reshape(-1, 1, 3), q[:3], q[3:], K, Z)[0]
                     .reshape(-1, 2) - P2u).ravel()
    s = least_squares(res, np.r_[rv0.ravel(), tv0.ravel()], method="lm", max_nfev=40000).x
    e = np.linalg.norm(res(s).reshape(-1, 2), axis=1)
    R, _ = cv2.Rodrigues(s[:3]); C = (-R.T @ s[3:]).ravel()
    ci = [i for i, t in enumerate(kind) if t == "carpet"]
    ti = [i for i, t in enumerate(kind) if t != "carpet"]
    nc = len(ci)
    print(f"  pose from {nc} carpet + {len(ti)} tag corners: median {np.median(e):.2f} px "
          f"(carpet {np.mean(e[ci]):.2f}, tags {np.mean(e[ti]):.2f})")
    print(f"  camera centre  X {C[0]:6.2f}  Y {C[1]:6.2f}  Z {C[2]:5.2f} m")
    np.save(f"out/cam_{view}_full.npy", np.r_[f, cx, cy, s[:3], s[3:], k1, k2])
    return dict(f=f, cx=cx, cy=cy, k1=k1, k2=k2, rv=s[:3], tv=s[3:], C=C, e=e, kind=kind)


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["blue_station"]):
        run(v); print()
