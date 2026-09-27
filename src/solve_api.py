"""One entry point that solves any view from whatever data currently exists.

Kept separate from the tools so the browser and the command line run the SAME
code -- the overlay disagreeing with the solver is exactly the confusion this
removes.  Stages are ordered so each is constrained by data the later ones do
not depend on: lines -> lens, vanishing points -> focal, points -> pose.
"""
import json, os, numpy as np, cv2, itertools
from scipy.optimize import minimize, least_squares

W, H = 960, 540
IN, SZ = 0.0254, 0.1651
SIGN, SHIFT, FLIP = +1, 0, True
_T = None


def tags():
    global _T
    if _T is None:
        import csv
        _T = {}
        with open("data/field/2026-rebuilt-welded.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                _T[int(r["ID"])] = r
    return _T


def corners3d(i):
    r = tags()[i]
    c = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])]) * IN
    rz, rx = np.radians(float(r["Z-Rotation"])), np.radians(float(r["X-Rotation"]))
    n = np.array([np.cos(rz), np.sin(rz), 0.0]); up = np.array([0, 0, 1.0])
    right = SIGN * np.cross(n, up); h = SZ / 2
    return np.array([c - right*h + up*h, c + right*h + up*h,
                     c + right*h - up*h, c - right*h - up*h])


def lresid(P):
    P = np.atleast_2d(P); m = P.mean(0); D = P - m
    _, _, V = np.linalg.svd(D)
    return float(np.sqrt(((D @ V[1]) ** 2).mean()))


def undist(P, k1, k2, cx, cy, f):
    P = np.atleast_2d(np.asarray(P, float))
    x = (P[:, 0] - cx) / f; y = (P[:, 1] - cy) / f
    r2 = x*x + y*y; s = 1 + k1*r2 + k2*r2*r2
    return np.c_[cx + x*s*f, cy + y*s*f]


def straight(p, L, f):
    return float(np.sqrt(np.mean([lresid(undist(np.array(l["pts"]), *p, f)) ** 2 for l in L])))


def fit_lens(L, f):
    best = None
    for c0 in itertools.product((0.4, 0.55), (0.35, 0.6)):
        r = minimize(lambda p: straight(p, L, f), [0.2, 0.2, W*c0[0], H*c0[1]],
                     method="Nelder-Mead",
                     bounds=[(-0.9, 1.2), (-0.9, 1.5), (0.15*W, 0.85*W), (0.15*H, 0.85*H)],
                     options=dict(maxiter=5000, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def vp(ls, p, f):
    rows = []
    for l in ls:
        Q = undist(np.array(l["pts"]), *p, f)
        m = Q.mean(0); _, _, V = np.linalg.svd(Q - m); d = V[0]
        n = np.array([-d[1], d[0]]); rows.append([n[0], n[1], -(n @ m)])
    _, _, V = np.linalg.svd(np.array(rows)); h = V[-1]
    return np.array([h[0]/h[2], h[1]/h[2]])


def solve(view, pts=None, lines=None, lens_from=None):
    """lens_from: reuse another view's lens (same camera, only bumped)."""
    if lines is None:
        lines = json.load(open("out/lines_multi.json")).get(view, [])
    lines = [l for l in lines if len(l.get("pts", [])) >= 3]
    if pts is None:
        pts = json.load(open("out/points_multi.json")).get(view, {}).get("pairs", [])
        try:
            pts = pts + json.load(open("out/pairs_xview.json")).get(view, [])
        except Exception:
            pass
    out = dict(view=view, n_lines=len(lines), n_points=len(pts))

    if lens_from:
        a = np.load(f"out/cam_{lens_from}_solved.npy")
        f, cx, cy, k1, k2 = a[0], a[1], a[2], a[9], a[10]
        out["lens_source"] = lens_from
    elif len(lines) >= 3:
        f = 675.0
        for _ in range(4):
            p, rms = fit_lens(lines, f)
            fx = [l for l in lines if l["dir"] == "X"]; fy = [l for l in lines if l["dir"] == "Y"]
            if len(fx) >= 2 and len(fy) >= 2:
                a_ = vp(fx, p, f) - p[2:]; b_ = vp(fy, p, f) - p[2:]
                f2 = -(a_ @ b_); nf = float(np.sqrt(f2)) if f2 > 0 else f
            else:
                nf = f
            if abs(nf - f) < 0.5:
                f = nf; break
            f = 0.5*f + 0.5*nf
        k1, k2, cx, cy = p
        out.update(straight_raw=straight([0, 0, W/2, H/2], lines, f), straight_fit=rms)
        out["lens_source"] = "lines"
    else:
        return dict(out, error="need 3+ lines (or reuse another view's lens)")

    P3, P2, kind = [], [], []
    for q in pts:
        P3.append([q["X"], q["Y"], 0.0]); P2.append([q["u"], q["v"]]); kind.append("carpet")
    tm = json.load(open("out/points_multi.json")).get(view, {}).get("tagMarks", {})
    for k, c in tm.items():
        i = int(k)
        if i not in tags():
            continue
        obs = np.array(c, float)
        if FLIP:
            obs = obs[::-1]
        obs = np.roll(obs, SHIFT, axis=0)
        for A, B in zip(corners3d(i), obs):
            P3.append(A); P2.append(B); kind.append("tag")
    if len(P3) < 6:
        return dict(out, error=f"only {len(P3)} constraints; need 6+")
    P3 = np.array(P3); P2u = undist(np.array(P2), k1, k2, cx, cy, f)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]); Z = np.zeros(5)
    ok, rv0, tv0 = cv2.solvePnP(P3.reshape(-1, 1, 3), P2u.reshape(-1, 1, 2), K, Z,
                                flags=cv2.SOLVEPNP_SQPNP)
    res = lambda q: (cv2.projectPoints(P3.reshape(-1, 1, 3), q[:3], q[3:], K, Z)[0]
                     .reshape(-1, 2) - P2u).ravel()
    s = least_squares(res, np.r_[rv0.ravel(), tv0.ravel()], method="lm", max_nfev=30000).x
    e = np.linalg.norm(res(s).reshape(-1, 2), axis=1)
    R, _ = cv2.Rodrigues(s[:3]); C = (-R.T @ s[3:]).ravel()
    ci = [i for i, t in enumerate(kind) if t == "carpet"]
    np.save(f"out/cam_{view}_solved.npy", np.r_[f, cx, cy, s[:3], s[3:], k1, k2])
    out.update(f=float(f), cx=float(cx), cy=float(cy), k1=float(k1), k2=float(k2),
               R=R.tolist(), t=s[3:].tolist(), C=C.tolist(),
               median=float(np.median(e)), maxerr=float(e.max()),
               carpet=float(np.mean(e[ci])) if ci else None,
               n_constraints=len(P3), per_point=[float(x) for x in e[:len(pts)]])
    return out
