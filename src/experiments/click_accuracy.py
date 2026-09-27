"""Error budget for mapping a clicked image point to field coordinates."""
import json, numpy as np, cv2, csv
from scipy.optimize import least_squares
IN=0.0254; LEN,WID=16.541,8.069; rng=np.random.default_rng(1)

tags={}
with open("data/field/2026-rebuilt-welded.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        tags[int(r["ID"])]=(float(r["X"])*IN,float(r["Y"])*IN,float(r["Z"])*IN)
P3,P2=[],[]
for p in json.load(open("out/points_corrected.json")):
    P3.append((p["X"],p["Y"],0.0)); P2.append((p["u"],p["v"]))
for tid,(u,v) in {2:(616.8,210.9),11:(583.3,211.2),21:(1303.8,204.6)}.items():
    P3.append(tags[tid]); P2.append((u,v))
for tid,c in json.load(open("out/tag_marks.json")).items():
    P3.append(tags[int(tid)]); P2.append(tuple(np.array(c,float).mean(0)))
P3=np.array(P3); P2=np.array(P2)

def unpack(p):
    K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]); d=np.zeros(5); d[0]=p[9]
    return K,p[3:6],p[6:9],d
def proj(p,X):
    K,rv,tv,d=unpack(p); q,_=cv2.projectPoints(X.reshape(-1,1,3),rv,tv,K,d)
    return q.reshape(-1,2)
def fit(idx,p0):
    return least_squares(lambda p:(proj(p,P3[idx])-P2[idx]).ravel(),p0,
                         method='lm',max_nfev=60000).x
p=np.load("out/cam_params.npy")

def back(p,u,v,Z=0.0):
    """image point -> field point at height Z"""
    K,rv,tv,d=unpack(p); R,_=cv2.Rodrigues(rv); C=-R.T@tv
    n=cv2.undistortPoints(np.array([[[u,v]]],float),K,d).reshape(2)
    ray=R.T@np.array([n[0],n[1],1.0]); t=(Z-C[2])/ray[2]
    return C[:2]+t*ray[:2]
def fwd(p,X,Y,Z=0.0): return proj(p,np.array([[X,Y,Z]]))[0]

K,rv,tv,d=unpack(p); R,_=cv2.Rodrigues(rv); C=-R.T@tv
print(f"camera at X{C[0]:.2f} Y{C[1]:.2f} Z{C[2]:.2f} m")

# bootstrap ensemble -> calibration parameter uncertainty
ens=[]
N=len(P3); ALL=np.arange(N)
for _ in range(200):
    idx=rng.choice(ALL,N,replace=True)
    if len(set(idx))<12: continue
    try: ens.append(fit(idx,p))
    except Exception: pass
ens=np.array(ens); print(f"bootstrap ensemble: {len(ens)} models\n")

CLICK_PX=1.0        # a careful click with the 6x refine view
print("error budget for a clicked CARPET-LEVEL point (inches):")
print(f"{'field X,Y':>13} {'in/px X':>8} {'in/px Y':>8} | {'click 1px':>10} "
      f"{'calib':>7} {'total':>7} | {'per 1in height':>14}")
rows=[]
for X in (2.0,5.0,8.27,11.5,14.5):
    for Y in (1.5,4.0,6.5):
        u,v=fwd(p,X,Y)
        if not(0<u<1920 and 0<v<490): continue
        sx=np.linalg.norm(back(p,u+1,v)-back(p,u,v))/IN
        sy=np.linalg.norm(back(p,u,v+1)-back(p,u,v))/IN
        click=CLICK_PX*np.hypot(sx,sy)
        pts=np.array([back(e,u,v) for e in ens])/IN
        calib=np.sqrt(((pts-pts.mean(0))**2).sum(1).mean())
        tot=np.hypot(click,calib)
        h=np.linalg.norm(back(p,u,v,0.0254)-back(p,u,v))/IN   # 1 inch of height
        rows.append((X,Y,tot,h))
        print(f"{X:6.1f},{Y:5.1f} {sx:8.2f} {sy:8.2f} | {click:10.1f} "
              f"{calib:7.1f} {tot:7.1f} | {h:14.1f}")

t=np.array([r[2] for r in rows]); h=np.array([r[3] for r in rows])
print(f"\nclicked carpet point: {t.min():.1f}-{t.max():.1f} in  (median {np.median(t):.1f})")
print(f"height sensitivity  : {h.min():.1f}-{h.max():.1f} in of field error PER INCH of height")
el=np.degrees(np.arctan2(C[2], C[1]-4.0))
print(f"\ncamera elevation above the field plane ~{el:.1f} deg -> "
      f"1/tan = {1/np.tan(np.radians(el)):.1f}x height amplification")
for name,hh in (("bumper top ~5in",5),("robot deck ~12in",12),("mechanism ~30in",30)):
    print(f"  clicking {name:18} instead of floor contact -> "
          f"{np.median(h)*hh:6.0f} in ({np.median(h)*hh/12:.1f} ft) of field error")
