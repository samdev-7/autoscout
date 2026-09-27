"""Per-camera field geometry for ALL broadcast views (wide + three station poses).

`geom.py` is the validated wide-camera module, but everything in it closes over
the wide camera: `_away()` pushes the corner correction toward the WIDE camera,
so a station detection put through a raw floor ray lands 0.579 m too close to
its OWN camera (the two view axes differ by ~54 deg, so this is a rigid 0.6 m
cross-camera disagreement).  This module does the same job generically from
`out/cams.json`.

Lens convention is the one `lines.html` fitted and `uncertainty.py` uses:
    r_undistorted = r_distorted * (1 + k1 r_d^2 + k2 r_d^4)
so UNDISTORTING a pixel is closed-form and PROJECTING needs a 1-D root solve.
Image coordinates are PANEL pixels (960x540 for stations, full-frame for wide).
"""
import json, numpy as np
import geom

CAMS = json.load(open("out/cams.json"))
SIG_DETECT = {"wide": 3.0, "red_station": 4.0, "red_station_b": 4.0, "blue_station": 4.0}
# MEASURED station residuals (traced robot projected into the panel vs nearest
# detection, 2026-09-04): red du sd 14.7 px + 22.8 px bias, dv sd 6.1; blue du sd
# 23.7 + 11.5 bias, dv sd 11.1.  Robots span 30-300 px in a station panel, so box
# edges are far sloppier than the 4 px a wide-panel detection gets.  Under-modelling
# this by 3-4x made cross-view merges fail and let stations over-pull fused
# positions.  Bias is folded in as noise until its source is understood.
SIG_PX = {"red_station": (24.0, 7.0), "red_station_b": (24.0, 7.0), "blue_station": (26.0, 12.0)}
CORNER_MEAN, CORNER_STD = geom.CORNER_MEAN, geom.CORNER_STD

# red station camera was bumped between these two frames (t=103.33..104.33 s)
RED_CUT0, RED_CUT1 = 3100, 3130


def red_view(frame):
    """Which red calibration applies to this frame; None inside the bump."""
    if frame < RED_CUT0: return "red_station"
    if frame >= RED_CUT1: return "red_station_b"
    return None


class Cam:
    def __init__(self, name):
        c = CAMS[name]; self.name = name
        self.f, self.cx, self.cy = c["f"], c["cx"], c["cy"]
        self.k1, self.k2 = c["k1"], c.get("k2", 0.0)
        self.R, self.t = np.array(c["R"], float), np.array(c["t"], float).ravel()
        self.C = -self.R.T @ self.t
        self.W, self.H = c["W"], c["H"]
        cal = c.get("carpet") or c.get("median") or 3.0
        self.sig_px = float(np.hypot(SIG_DETECT.get(name, 4.0), cal))

    # ---- lens -----------------------------------------------------------
    def _g(self, r):
        return r * (1 + self.k1 * r * r + self.k2 * r ** 4)

    def _undist(self, xd, yd):
        r = np.hypot(xd, yd); s = np.where(r > 1e-12, self._g(r) / np.maximum(r, 1e-12), 1.0)
        return xd * s, yd * s

    def _dist(self, xu, yu):
        """Invert g by bisection on radius (vectorised); NaN where g never reaches R."""
        Ru = np.hypot(xu, yu)
        lo = np.zeros_like(Ru); hi = np.maximum(Ru, 1e-3)
        for _ in range(60):
            grow = self._g(hi) < Ru
            if not grow.any(): break
            hi = np.where(grow, hi * 1.5, hi)
            if (hi > 50).any(): break
        bad = self._g(hi) < Ru
        for _ in range(60):
            m = 0.5 * (lo + hi); up = self._g(m) < Ru
            lo, hi = np.where(up, m, lo), np.where(up, hi, m)
        rd = 0.5 * (lo + hi); s = np.where(Ru > 1e-12, rd / np.maximum(Ru, 1e-12), 1.0)
        xd, yd = xu * s, yu * s
        xd[bad] = np.nan; yd[bad] = np.nan
        return xd, yd

    # ---- projection ---------------------------------------------------------
    def project(self, P):
        """(...,2) ground or (...,3) field points -> (...,2) panel pixels."""
        P = np.atleast_2d(np.asarray(P, float))
        if P.shape[-1] == 2: P = np.c_[P, np.zeros(len(P))]
        c = P @ self.R.T + self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            xu, yu = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
        xd, yd = self._dist(xu, yu)
        q = np.c_[self.cx + self.f * xd, self.cy + self.f * yd]
        q[c[:, 2] < 0.5] = np.nan
        return q

    def img_to_plane(self, uv, Z=0.0):
        uv = np.atleast_2d(np.asarray(uv, float))
        xu, yu = self._undist((uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f)
        d = np.c_[xu, yu, np.ones(len(uv))] @ self.R          # ray dirs in field frame
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (Z - self.C[2]) / d[:, 2]
        G = self.C[None, :] + s[:, None] * d
        G[(s <= 0) | ~np.isfinite(s)] = np.nan
        return G[:, :2]

    def img_to_ground(self, uv):
        return self.img_to_plane(uv, 0.0)

    def away(self, XY):
        v = np.atleast_2d(np.asarray(XY, float)) - self.C[:2]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def point_to_field(self, uv, corner=CORNER_MEAN):
        """(x-centre, nearest-ground-corner) pixel -> footprint centre, pushed
        away from THIS camera."""
        G = self.img_to_ground(uv)
        return G + corner * self.away(G)

    def box_to_field(self, boxes, band=0, corner=CORNER_MEAN):
        b = np.atleast_2d(np.asarray(boxes, float))
        return self.point_to_field(np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3] + band], corner)

    def visible(self, XY, pad=0):
        q = self.project(XY)
        ok = np.isfinite(q).all(1)
        ok &= (q[:, 0] >= -pad) & (q[:, 0] < self.W + pad) & (q[:, 1] >= -pad) & (q[:, 1] < self.H + pad)
        return ok

    # ---- uncertainty ---------------------------------------------------------
    def jac(self, XY, h=2e-3):
        P = np.atleast_2d(np.asarray(XY, float)); J = np.empty((len(P), 2, 2))
        for j in range(2):
            e = np.zeros(2); e[j] = h
            J[:, :, j] = (self.project(P + e) - self.project(P - e)) / (2 * h)
        return J

    def cov(self, XY, sig_px=None, yaw=CORNER_STD):
        """Field covariance (N,2,2): pixel noise through the inverse Jacobian, plus
        the unknown-yaw corner term along the view ray.  `sig_px` is (su, sv) or a
        scalar; the default is the MEASURED residual for this camera (see SIG_PX)."""
        P = np.atleast_2d(np.asarray(XY, float))
        s = SIG_PX.get(self.name, (self.sig_px, self.sig_px)) if sig_px is None else sig_px
        su, sv = (s, s) if np.isscalar(s) else s
        J = self.jac(P); out = np.full((len(P), 2, 2), np.nan)
        ok = np.isfinite(J).all((1, 2)) & (np.abs(np.linalg.det(np.nan_to_num(J))) > 1e-9)
        Ji = np.linalg.inv(J[ok])
        u = self.away(P[ok])[:, :, None]
        out[ok] = Ji @ np.diag([su ** 2, sv ** 2]) @ Ji.transpose(0, 2, 1) + (yaw ** 2) * (u @ u.transpose(0, 2, 1))
        return out

    def height_of(self, uv_top, XY):
        """Robot height (m) that puts its top at pixel row v_top, given ground XY."""
        v = float(np.asarray(uv_top, float).reshape(-1)[1]); XY = np.asarray(XY, float).reshape(2)
        lo, hi = 0.05, 2.5
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            pv = self.project(np.r_[XY, mid])[0][1]
            if not np.isfinite(pv): return np.nan
            if pv > v: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)


class WideCam(Cam):
    """The wide view keeps geom.py's validated model (tags to 1.24 px); the
    cams.json 'wide' entry is a different fit that disagrees by up to 25 px at
    the field edges.  Only the lens/pose primitives are swapped."""
    def __init__(self):
        super().__init__("wide"); self.C = geom.CAM.copy()
    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        return geom.field_to_img(P)
    def img_to_plane(self, uv, Z=0.0):
        return geom.img_to_plane(uv, Z)


_cache = {}
def cam(name):
    if name not in _cache: _cache[name] = WideCam() if name == "wide" else Cam(name)
    return _cache[name]


if __name__ == "__main__":
    g = np.array([[x, y] for x in np.linspace(0.5, geom.LEN - .5, 17) for y in np.linspace(0.5, geom.WID - .5, 9)])
    w = cam("wide")
    qa, qb = w.project(g), geom.field_to_img(g)
    print(f"wide: cams.json vs geom.py projection of {len(g)} ground points: "
          f"max {np.nanmax(np.linalg.norm(qa-qb,axis=1)):.2f} px, median {np.nanmedian(np.linalg.norm(qa-qb,axis=1)):.2f} px")
    for n in CAMS:
        c = cam(n); q = c.project(g); vis = c.visible(g)
        rt = c.img_to_ground(q[vis]); err = np.linalg.norm(rt - g[vis], axis=1)
        print(f"{n:14s} centre ({c.C[0]:6.2f},{c.C[1]:6.2f},{c.C[2]:5.2f})  visible {vis.sum():3d}/{len(g)}  "
              f"round trip max {err.max()*1000:.2f} mm   sig_px {c.sig_px:.2f}")
    # corner correction direction differs per camera
    p = np.array([[8.27, 4.0]])
    print("away-from-camera unit vectors at field centre:", {n: cam(n).away(p)[0].round(2).tolist() for n in CAMS})
