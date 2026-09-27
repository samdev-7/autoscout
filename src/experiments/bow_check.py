"""Bound the distortion using only edges verified to be single physical lines.

Radial distortion imposes structure: bow sign must flip about the distortion
centre and grow with distance from it. We measure bow per edge and test for
that structure, instead of trusting an unconstrained k1/k2 fit.
"""
import numpy as np, cv2

plate = np.load("out/top_plate.npy")
rgb = plate.copy()
g = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32), (0,0), 1.0)
sy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
H, W = g.shape

def trace_tight(seed_y, x0, x1, win=3, thr=30):
    """Tight-window trace: cannot jump to a different edge."""
    model = np.array([0.0, float(seed_y)])
    for _ in range(6):
        X, Y, S = [], [], []
        for x in range(x0, x1):
            yc = np.polyval(model, x)
            lo, hi = int(max(0, yc-win)), int(min(H, yc+win+1))
            if hi-lo < 3: continue
            col = sy[lo:hi, x]
            if col.max() < thr: continue
            yy = lo + int(col.argmax())
            if 0 < yy < H-1:
                a,b,c = sy[yy-1,x], sy[yy,x], sy[yy+1,x]
                d = a-2*b+c
                yy = yy + (0.5*(a-c)/d if abs(d) > 1e-6 else 0)
            X.append(x); Y.append(float(yy)); S.append(col.max())
        if len(X) < 300: return None
        X, Y = np.array(X,float), np.array(Y,float)
        m = np.polyfit(X, Y, 1); r = Y-np.polyval(m, X)
        k = np.abs(r) < 2.0
        if k.sum() < 250: return None
        X, Y = X[k], Y[k]; model = np.polyfit(X, Y, 1)
    return X, Y

# candidate long, high-contrast, mostly-unoccluded horizontal structures
CANDS = [(292,60,1870),(354,60,1870),(400,60,1870),(430,60,1870),(468,60,1870),
         (202,200,1700),(172,200,1700),(121,200,1700)]
print(f"{'seed':>5} {'span':>6} {'n':>5} {'rms':>7} {'max|r|':>7} {'bow_px':>8}  verdict")
rows=[]
for s,x0,x1 in CANDS:
    r = trace_tight(s,x0,x1)
    if r is None:
        print(f"{s:5d}      -     -       -       -        -  no clean trace"); continue
    X,Y = r
    lin = np.polyfit(X,Y,1); res = Y-np.polyval(lin,X)
    q = np.polyfit(X,Y,2)
    half = np.ptp(X)/2
    bow = q[0]*half**2          # sagitta: peak deviation from chord
    ok = res.std() < 0.8 and np.abs(res).max() < 3.0
    ymid = np.polyval(lin, (X.min()+X.max())/2)
    print(f"{s:5d} {np.ptp(X):6.0f} {len(X):5d} {res.std():7.3f} {np.abs(res).max():7.2f} {bow:+8.2f}  "
          f"{'CLEAN' if ok else 'contaminated'}")
    if ok: rows.append((ymid,bow,np.ptp(X)))
    for xx,yy in zip(X[::12],Y[::12]):
        cv2.circle(rgb,(int(xx),int(round(yy))),1,(0,255,0) if ok else (255,0,0),-1)

print("\nclean edges — bow vs image row (radial distortion => monotonic sign flip about centre):")
for ymid,bow,span in sorted(rows):
    print(f"  y={ymid:6.1f}  bow={bow:+6.2f}px over {span:.0f}px span")
if len(rows) >= 3:
    ys = np.array([r[0] for r in rows]); bw = np.array([r[1] for r in rows])
    c = np.corrcoef(ys, bw)[0,1]
    print(f"\n  correlation(bow, row) = {c:+.3f}   "
          f"({'consistent with radial distortion' if abs(c)>0.9 else 'NO radial structure'})")
    print(f"  |bow| max = {np.abs(bw).max():.2f}px  -> distortion is at most ~{np.abs(bw).max():.1f}px "
          f"over a {np.ptp([r[2] for r in rows]):.0f}px span")
cv2.imwrite("out/traced_edges.jpg", cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])
print("\nwrote out/traced_edges.jpg (green=clean, red=contaminated)")
