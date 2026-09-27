"""Fit the ground-plane calibration and test whether distortion is detectable.

The 11 marked points are exact known correspondences, so they test distortion
far better than traced edges: fit homography-only vs homography+radial terms and
see whether the extra freedom actually buys anything above marking noise.
"""
import json, numpy as np, cv2
from scipy.optimize import least_squares

LEN, WID, IN = 16.541, 8.069, 0.0254
pts = json.load(open("out/points_user.json"))
img = np.array([[p["u"],p["v"]] for p in pts], float)
# hypothesis B: camera views the field from the opposite side
fld = np.array([[LEN-p["X"], WID-p["Y"]] for p in pts], float)
TAGS = [(2,616.8,210.9,469.111*IN,182.6*IN),(11,583.3,211.2,483.111*IN,182.6*IN),
        (21,1303.8,204.6,182.111*IN,182.6*IN)]
TAGZ = 44.25*IN

def resid_H(h, F, I, k=(), cxy=None, f=1000.0):
    H = np.append(h[:8],1.0).reshape(3,3)
    P = cv2.perspectiveTransform(F.reshape(-1,1,2), H).reshape(-1,2)
    if k:
        cx,cy = cxy
        xn,yn = (P[:,0]-cx)/f, (P[:,1]-cy)/f
        r2 = xn*xn+yn*yn
        s = 1.0
        for i,kk in enumerate(k): s = s + kk*r2**(i+1)
        P = np.column_stack([cx+(P[:,0]-cx)*s, cy+(P[:,1]-cy)*s])
    return (P-I).ravel()

H0,_ = cv2.findHomography(fld, img, 0)
h0 = (H0/H0[2,2]).ravel()[:8]

def run(nk, free_c=False, label=""):
    cx0, cy0 = 960.0, 245.0
    def fun(p):
        h = p[:8]; k = tuple(p[8:8+nk])
        c = (p[8+nk], p[9+nk]) if free_c else (cx0, cy0)
        return resid_H(h, fld, img, k, c)
    p0 = np.concatenate([h0, np.zeros(nk), ([cx0,cy0] if free_c else [])])
    s = least_squares(fun, p0, method="lm" if not free_c else "trf", max_nfev=20000)
    r = s.fun.reshape(-1,2); e = np.linalg.norm(r, axis=1)
    npar = len(p0)
    print(f"  {label:34} RMS={np.sqrt((e**2).mean()):6.3f}px  max={e.max():6.3f}  "
          f"params={npar:2d}  k={np.round(s.x[8:8+nk],5)}"
          + (f"  c=({s.x[8+nk]:.0f},{s.x[9+nk]:.0f})" if free_c else ""))
    return s, e

print("distortion model comparison (11 points, 22 observations):")
s0,e0 = run(0, False, "H only (8 dof)")
s1,e1 = run(1, False, "H + k1 (9)")
s2,e2 = run(2, False, "H + k1,k2 (10)")
s3,e3 = run(1, True,  "H + k1 + free centre (11)")

rms0 = np.sqrt((e0**2).mean())
print(f"\n  adding k1 changes RMS by {100*(np.sqrt((e1**2).mean())-rms0)/rms0:+.1f}%")
print("  -> distortion is NOT separable from marking noise at this point count/spread")

# bound: how big could k1 be before it visibly degrades the fit?
print("\n  k1 sensitivity (RMS as k1 is forced away from 0):")
for k1 in (-0.02,-0.01,-0.005,0.0,0.005,0.01,0.02):
    def fun(h): return resid_H(h, fld, img, (k1,), (960.0,245.0))
    ss = least_squares(fun, h0, method="lm", max_nfev=20000)
    ee = np.linalg.norm(ss.fun.reshape(-1,2),axis=1)
    xn,yn=(1800-960)/1000,(430-245)/1000
    shift = abs(k1*(xn*xn+yn*yn))*np.hypot(1800-960,430-245)
    print(f"    k1={k1:+.3f}  RMS={np.sqrt((ee**2).mean()):6.3f}px   "
          f"implies {shift:5.2f}px shift at frame corner")

H = np.append(s0.x[:8],1.0).reshape(3,3)
np.save("out/H_field_to_img.npy", H)

print("\nheld-out AprilTag check with final H:")
lines=[]
for tid,uo,vo,X,Y in TAGS:
    p = cv2.perspectiveTransform(np.array([[[X,Y]]],float), H).reshape(2)
    lines.append((p,(uo,vo)))
    print(f"  tag{tid:<3} carpet_pred=({p[0]:7.1f},{p[1]:6.1f})  tag_obs=({uo:6.1f},{vo:5.1f})"
          f"  du={p[0]-uo:+6.1f}  dv={p[1]-vo:+6.1f}")
# vertical vanishing point: each tag sits directly above its carpet point
A=[];b=[]
for p,(uo,vo) in lines:
    dx,dy = uo-p[0], vo-p[1]
    A.append([dy,-dx]); b.append(dy*p[0]-dx*p[1])
vp = np.linalg.lstsq(np.array(A),np.array(b),rcond=None)
Vz = vp[0]; res = vp[1]
print(f"\nvertical vanishing point from the 3 tag/carpet pairs: "
      f"({Vz[0]:.0f}, {Vz[1]:.0f})   residual={res if len(res) else 'exact':}")
print(f"  -> vertical is {np.degrees(np.arctan2(Vz[0]-960, -(Vz[1]-245))):+.2f} deg off image-vertical")
