"""Estimate distortion and focal from marked straight lines.

Two constraints that need no camera position at all:
  * a real straight edge must be straight after undistortion (plumb line) -> k1
  * families of parallel edges meet at a vanishing point, and two perpendicular
    families' vanishing points satisfy v1 . w . v2 = 0 -> focal length
Only once those are known do the ground points have to do anything, and then
only to fix position -- which is what near-camera points can actually do.
"""
import json, numpy as np, sys
from scipy.optimize import minimize_scalar, least_squares

W, H = 960, 540


def undist(P, k1, cx, cy, f):
    P = np.atleast_2d(P).astype(float)
    x = (P[:, 0] - cx) / f; y = (P[:, 1] - cy) / f
    s = 1 + k1 * (x * x + y * y)
    return np.c_[cx + x * s * f, cy + y * s * f]


def resid(P):
    """RMS perpendicular distance of points from their best-fit line."""
    P = np.atleast_2d(P); m = P.mean(0); D = P - m
    _, _, V = np.linalg.svd(D)
    n = V[1]
    return float(np.sqrt(((D @ n) ** 2).mean()))


def total(lines, k1, cx, cy, f):
    r = [resid(undist(np.array(L["pts"]), k1, cx, cy, f)) ** 2
         for L in lines if len(L["pts"]) >= 3]
    return float(np.sqrt(np.mean(r))) if r else None


def vanish(lines, k1, cx, cy, f):
    """Least-squares vanishing point of a family, in homogeneous coords."""
    rows = []
    for L in lines:
        P = undist(np.array(L["pts"]), k1, cx, cy, f)
        m = P.mean(0); D = P - m
        _, _, V = np.linalg.svd(D)
        d = V[0]                                  # line direction
        n = np.array([-d[1], d[0]])               # normal
        rows.append([n[0], n[1], -(n @ m)])       # line through the points
    A = np.array(rows)
    _, _, V = np.linalg.svd(A)
    return V[-1]


if __name__ == "__main__":
    view = sys.argv[1] if len(sys.argv) > 1 else "red_station"
    L = json.load(open("out/lines_multi.json"))[view]
    L = [l for l in L if len(l["pts"]) >= 3]
    print(f"{view}: {len(L)} usable lines "
          f"({sum(1 for l in L if l['dir']=='X')} X, {sum(1 for l in L if l['dir']=='Y')} Y, "
          f"{sum(1 for l in L if l['dir']=='V')} V)\n")
    cx, cy = W / 2, H / 2
    f0 = 660.0
    print(f"{'k1':>8}{'straightness RMS':>20}")
    for k in (0.0, -0.1, -0.2, -0.3, -0.4):
        print(f"{k:8.2f}{total(L,k,cx,cy,f0):20.3f}")
    r = minimize_scalar(lambda k: total(L, k, cx, cy, f0), bounds=(-0.6, 0.2), method="bounded")
    k1 = r.x
    print(f"\nbest k1 = {k1:+.4f}   straightness {total(L,k1,cx,cy,f0):.3f} px "
          f"(was {total(L,0,cx,cy,f0):.3f} px undistorted)")
    print(f"improvement factor {total(L,0,cx,cy,f0)/max(total(L,k1,cx,cy,f0),1e-9):.1f}x")

    # per-line residual after correction -- flags a mis-traced edge
    print(f"\n{'line':>6}{'dir':>5}{'pts':>5}{'span px':>9}{'before':>9}{'after':>8}")
    for i, l in enumerate(L):
        P = np.array(l["pts"]); span = np.hypot(*(P.max(0) - P.min(0)))
        print(f"{i+1:6d}{l['dir']:>5}{len(P):5d}{span:9.0f}"
              f"{resid(P):9.2f}{resid(undist(P,k1,cx,cy,f0)):8.2f}")

    fams = {d: [l for l in L if l["dir"] == d] for d in ("X", "Y", "V")}
    print()
    vps = {}
    for d, ls in fams.items():
        if len(ls) < 2:
            print(f"family {d}: {len(ls)} line(s) — need 2+ for a vanishing point"); continue
        v = vanish(ls, k1, cx, cy, f0)
        vps[d] = v
        if abs(v[2]) > 1e-9:
            print(f"family {d}: vanishing point ({v[0]/v[2]:9.1f},{v[1]/v[2]:9.1f}) px")
        else:
            print(f"family {d}: vanishing point at infinity (lines are parallel in image)")
