"""Field-space position uncertainty per camera, and what fusing them buys.

Two error sources, and both belong in the ellipse:
  * detection noise -- how precisely a robot's floor contact can be located in
    that view, in pixels
  * CALIBRATION residual -- how well the camera model reproduces known ground
    points.  Leaving this out makes the fused ellipse optimistically small and
    the tracker over-trust it.
They add in quadrature in pixels, then propagate to metres through each camera's
own Jacobian, so the result is anisotropic and position-dependent by construction.
"""
import json, numpy as np, cv2

CAMS = json.load(open("out/cams.json"))
SIG_DETECT = {"wide": 3.0, "red_station": 4.0, "red_station_b": 4.0, "blue_station": 4.0}


def _redist(cam, q):
    k1, k2 = cam["k1"], cam.get("k2", 0.0)
    f, cx, cy = cam["f"], cam["cx"], cam["cy"]
    X, Y = (q[0] - cx) / f, (q[1] - cy) / f
    R = np.hypot(X, Y)
    if R < 1e-12:
        return np.array([cx, cy])
    g = lambda r: r * (1 + k1 * r * r + k2 * r ** 4)
    lo, hi = 0.0, max(R, 1e-3)
    while g(hi) < R and hi < 50:
        hi *= 1.5
    if g(hi) < R:
        return np.array([np.nan, np.nan])
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if g(m) < R else (lo, m)
    r = 0.5 * (lo + hi)
    return np.array([cx + X * (r / R) * f, cy + Y * (r / R) * f])


def project(cam, XY, Z=0.0):
    R = np.array(cam["R"]); t = np.array(cam["t"])
    P = np.array([XY[0], XY[1], Z], float)
    c = R @ P + t
    if c[2] < 0.5:
        return np.array([np.nan, np.nan])
    q = np.array([cam["cx"] + cam["f"] * c[0] / c[2], cam["cy"] + cam["f"] * c[1] / c[2]])
    return _redist(cam, q)


def visible(cam, XY):
    q = project(cam, XY)
    if not np.isfinite(q).all():
        return False
    return 0 <= q[0] < cam["W"] and 0 <= q[1] < cam["H"]


def sigma_px(name):
    """Detection noise and calibration residual, added in quadrature."""
    cal = CAMS[name].get("carpet") or CAMS[name].get("median") or 3.0
    det = SIG_DETECT.get(name, 4.0)
    return float(np.hypot(det, cal))


def cov(name, XY, h=2e-3):
    cam = CAMS[name]
    if not visible(cam, XY):
        return None
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        a = project(cam, np.asarray(XY, float) + e)
        b = project(cam, np.asarray(XY, float) - e)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return None
        J[:, j] = (a - b) / (2 * h)
    if abs(np.linalg.det(J)) < 1e-9:
        return None
    Ji = np.linalg.inv(J)
    s = sigma_px(name)
    return Ji @ (np.eye(2) * s ** 2) @ Ji.T


def fuse(names, XY):
    """Inverse-covariance (information) sum over the cameras that can see it."""
    I = np.zeros((2, 2)); used = []
    for n in names:
        C = cov(n, XY)
        if C is None:
            continue
        try:
            I = I + np.linalg.inv(C); used.append(n)
        except np.linalg.LinAlgError:
            pass
    if not used:
        return None, []
    return np.linalg.inv(I), used


def axes(C):
    """1-sigma semi-axes (major, minor) in metres."""
    w = np.linalg.eigvalsh(C)
    w = np.sqrt(np.maximum(w, 0))
    return float(w[1]), float(w[0])
