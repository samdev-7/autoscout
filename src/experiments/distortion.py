"""Measure lens distortion via the plumb-line constraint.

A straight 3D line images straight under an ideal pinhole camera regardless of
viewpoint, so a consistent BOW in a long world-straight edge is lens distortion.
Trace long horizontal edges on the clean median plate (gap-tolerant, since
spectators occlude parts of them), then fit one radial model that straightens
all of them simultaneously.
"""
import numpy as np, cv2
from scipy.optimize import least_squares

plate = np.load("out/top_plate.npy")
g = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
g = cv2.GaussianBlur(g, (0,0), 1.2)
sy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
H, W = g.shape

def subpix(yy, x):
    if 0 < yy < H-1:
        a,b,c = sy[yy-1,x], sy[yy,x], sy[yy+1,x]
        d = a - 2*b + c
        if abs(d) > 1e-6: return yy + 0.5*(a-c)/d
    return float(yy)

def trace(seed_y, x0=40, x1=1880, win=7, thr=15, iters=4):
    """Fit y(x) robustly, tolerating occluded columns. Returns (xs, ys) or None."""
    model = np.array([0.0, float(seed_y)])          # start: horizontal line
    xs = ys = None
    for it in range(iters):
        X, Y = [], []
        for x in range(x0, x1):
            yc = np.polyval(model, x)
            lo, hi = int(max(0, yc-win)), int(min(H, yc+win+1))
            if hi - lo < 3: continue
            col = sy[lo:hi, x]
            if col.max() < thr: continue            # occluded / no edge: skip
            X.append(x); Y.append(subpix(lo + int(col.argmax()), x))
        if len(X) < 500: return None
        X, Y = np.array(X, float), np.array(Y, float)
        m = np.polyfit(X, Y, 1)
        r = Y - np.polyval(m, X)
        keep = np.abs(r) < max(2.5, 2.5*r.std())    # trim outliers
        if keep.sum() < 400: return None
        X, Y = X[keep], Y[keep]
        model = np.polyfit(X, Y, 1)
        xs, ys = X, Y
    if np.ptp(xs) < 1000 or len(xs) < 600: return None
    return xs, ys

rowscore = (sy > 25).sum(axis=1)
seeds, merged = [y for y in range(15, H-15) if rowscore[y] > 700], []
for y in seeds:
    if not merged or y - merged[-1] > 14: merged.append(y)
print("candidate edge rows:", merged)

curves, seen = [], []
for s in merged:
    r = trace(s)
    if r is None: continue
    xs, ys = r
    yc = np.polyval(np.polyfit(xs, ys, 1), 960)
    if any(abs(yc - v) < 10 for v in seen): continue   # dedupe same edge
    seen.append(yc); curves.append((s, xs, ys))

print(f"\nkept {len(curves)} long straight edges")
print(f"{'seed':>5} {'span':>6} {'npts':>6} {'rms':>7} {'max|r|':>7} {'quad':>11}")
for s, xs, ys in curves:
    m = np.polyfit(xs, ys, 1); res = ys - np.polyval(m, xs)
    q = np.polyfit(xs, ys, 2)[0]
    print(f"{s:5d} {np.ptp(xs):6.0f} {len(xs):6d} {res.std():7.3f} {np.abs(res).max():7.2f} {q:+11.3e}")

if len(curves) < 3:
    raise SystemExit("\nnot enough straight edges to fit a distortion model")

F = 1600.0
def undistort(x, y, cx, cy, k1, k2):
    xn, yn = (x-cx)/F, (y-cy)/F
    r2 = xn*xn + yn*yn
    s = 1 + k1*r2 + k2*r2*r2
    return cx + xn*s*F, cy + yn*s*F

def resid(p):
    cx, cy, k1, k2 = p
    out = []
    for _, xs, ys in curves:
        xu, yu = undistort(xs, ys, cx, cy, k1, k2)
        out.append(yu - np.polyval(np.polyfit(xu, yu, 1), xu))
    return np.concatenate(out)

p0 = np.array([W/2, 300.0, 0.0, 0.0])
sol = least_squares(resid, p0, bounds=([0,-3000,-2,-2],[W,4000,2,2]))
cx, cy, k1, k2 = sol.x
b, a = resid(p0), resid(sol.x)
print(f"\nfitted  cx={cx:.1f}  cy={cy:.1f}  k1={k1:+.5f}  k2={k2:+.5f}   (f fixed {F:.0f})")
print(f"RMS straightness residual:  before={b.std():.3f}px   after={a.std():.3f}px"
      f"   ({(1-a.std()/b.std())*100:.1f}% reduction)")

print("\npixel displacement implied by the fitted model:")
for x, y in [(60,150),(1860,150),(60,430),(1860,430),(960,290),(1303,205)]:
    xu, yu = undistort(np.array([x],float), np.array([y],float), cx, cy, k1, k2)
    print(f"  ({x:4d},{y:3d}) shift = {np.hypot(xu[0]-x, yu[0]-y):6.2f} px")
