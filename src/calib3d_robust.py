"""Constrained camera model + bootstrap uncertainty.

Tag centres come from ~7x15 px targets, so the honest move is to REMOVE model
freedom (square pixels, zero skew -- both physically true of a broadcast camera)
rather than let 11 unconstrained DOF absorb marking noise into nonsense like
skew=33. Bootstrap then says how well each parameter is actually determined.
"""
import json, numpy as np, cv2, csv
from scipy.optimize import least_squares
IN=0.0254; rng=np.random.default_rng(0)

tags={}
with open("data/field/2026-rebuilt-welded.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        tags[int(r["ID"])]=(float(r["X"])*IN,float(r["Y"])*IN,float(r["Z"])*IN)
P3,P2,lab=[],[],[]
for p in json.load(open("out/points_corrected.json")):
    P3.append((p["X"],p["Y"],0.0)); P2.append((p["u"],p["v"])); lab.append("carpet")
for tid,(u,v) in {2:(616.8,210.9),11:(583.3,211.2),21:(1303.8,204.6)}.items():
    P3.append(tags[tid]); P2.append((u,v)); lab.append(f"tag{tid}d")
for tid,c in json.load(open("out/tag_marks.json")).items():
    c=np.array(c,float); P3.append(tags[int(tid)]); P2.append(tuple(c.mean(0)))
    lab.append(f"tag{tid}m")
P3=np.array(P3); P2=np.array(P2)

def pack(f,cx,cy,rv,tv,k): return np.r_[f,cx,cy,rv,tv,k]
def unpack(p,nk):
    f,cx,cy=p[0],p[1],p[2]; rv=p[3:6]; tv=p[6:9]; k=p[9:9+nk]
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1]])
    d=np.zeros(5); d[0]=k[0] if nk>0 else 0; d[1]=k[1] if nk>1 else 0
    return K,rv,tv,d
def proj(p,X,nk):
    K,rv,tv,d=unpack(p,nk)
    q,_=cv2.projectPoints(X.reshape(-1,1,3),rv,tv,K,d)
    return q.reshape(-1,2)

K0=np.array([[1836.,0,960.],[0,1836.,192.],[0,0,1]])
ok,rv0,tv0=cv2.solvePnP(P3.reshape(-1,1,3),P2.reshape(-1,1,2),K0,np.zeros(5),
                        flags=cv2.SOLVEPNP_ITERATIVE)
def fit(idx,nk,p0=None):
    p0=pack(1836.,960.,192.,rv0.ravel(),tv0.ravel(),np.zeros(nk)) if p0 is None else p0
    f=lambda p:(proj(p,P3[idx],nk)-P2[idx]).ravel()
    return least_squares(f,p0,method='lm',max_nfev=60000).x

ALL=np.arange(len(P3))
print("constrained model (square pixels, zero skew):")
res={}
for nk in (0,1,2):
    s=fit(ALL,nk); e=np.linalg.norm(proj(s,P3,nk)-P2,axis=1)
    loo=[]
    for i in ALL:
        tr=np.array([j for j in ALL if j!=i]); si=fit(tr,nk)
        loo.append(np.linalg.norm(proj(si,P3[i:i+1],nk)[0]-P2[i]))
    res[nk]=(s,np.sqrt((e**2).mean()),np.mean(loo))
    print(f"  k terms={nk}  params={9+nk:2d}  fit RMS {np.sqrt((e**2).mean()):6.3f} px   "
          f"LOO mean {np.mean(loo):6.3f} px")

best=min(res,key=lambda n:res[n][2])
print(f"\nbest by held-out error: k terms={best}")
s=res[best][0]; K,rv,tv,d=unpack(s,best)
R,_=cv2.Rodrigues(rv); C=-R.T@tv
print(f"  focal {K[0,0]:.1f} px   principal point ({K[0,2]:.1f}, {K[1,2]:.1f})   "
      f"k1={d[0]:+.5f}"+(f" k2={d[1]:+.5f}" if best>1 else ""))
print(f"  camera centre  X {C[0]:.2f}  Y {C[1]:.2f}  Z {C[2]:.2f} m")

# bootstrap: resample correspondences with replacement
print(f"\nbootstrap (300 resamples) — 16th/50th/84th percentile:")
B=[]
for _ in range(300):
    idx=rng.choice(ALL,len(ALL),replace=True)
    if len(set(idx))<12: continue
    try:
        sb=fit(idx,best); Kb,rvb,tvb,db=unpack(sb,best)
        Rb,_=cv2.Rodrigues(rvb); Cb=-Rb.T@tvb
        B.append([Kb[0,0],Kb[0,2],Kb[1,2],db[0],Cb[0],Cb[1],Cb[2]])
    except Exception: pass
B=np.array(B)
names=["focal px","ppx","ppy","k1","cam X m","cam Y m","cam Z m"]
for i,n in enumerate(names):
    lo,md,hi=np.percentile(B[:,i],[16,50,84])
    print(f"  {n:9} {md:9.3f}  [{lo:9.3f}, {hi:9.3f}]   +/-{(hi-lo)/2:8.3f}")

# what does the fitted distortion mean in pixels?
K,rv,tv,d=unpack(res[best][0],best)
for u,v in [(80,340),(960,245),(1750,240),(960,470)]:
    n=np.array([[(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1]]])
    r2=(n**2).sum(); sc=1+d[0]*r2+d[1]*r2*r2
    du=(n[0,0]*sc*K[0,0]+K[0,2])-u; dv=(n[0,1]*sc*K[1,1]+K[1,2])-v
    print(f"  distortion shift at ({u:4d},{v:3d}) = {np.hypot(du,dv):5.2f} px")
np.save("out/cam_params.npy",res[best][0]); print(f"\nsaved out/cam_params.npy (nk={best})")
