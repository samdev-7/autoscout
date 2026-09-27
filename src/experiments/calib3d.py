"""Full 3D camera calibration from carpet points + marked tag centres.

Coplanarity is broken now (Z at 0, 0.5525, 1.124 m), so radial distortion is no
longer degenerate with the projective terms -- the distortion question can be
answered instead of guessed.
"""
import json, numpy as np, cv2, csv
from scipy.linalg import rq
from scipy.optimize import least_squares
IN=0.0254

tags={}
with open("data/field/2026-rebuilt-welded.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        tags[int(r["ID"])]=(float(r["X"])*IN,float(r["Y"])*IN,float(r["Z"])*IN)

P3,P2,lab=[],[],[]
for p in json.load(open("out/points_corrected.json")):
    P3.append((p["X"],p["Y"],0.0)); P2.append((p["u"],p["v"])); lab.append("carpet")
for tid,(u,v) in {2:(616.8,210.9),11:(583.3,211.2),21:(1303.8,204.6)}.items():
    P3.append(tags[tid]); P2.append((u,v)); lab.append(f"tag{tid}(det)")
for tid,c in json.load(open("out/tag_marks.json")).items():
    c=np.array(c,float); P3.append(tags[int(tid)])
    P2.append(tuple(c.mean(axis=0))); lab.append(f"tag{tid}(marked)")
P3=np.array(P3,float); P2=np.array(P2,float)
print(f"{len(P3)} correspondences   X {P3[:,0].min():.2f}..{P3[:,0].max():.2f}  "
      f"Y {P3[:,1].min():.2f}..{P3[:,1].max():.2f}  Z {sorted(set(np.round(P3[:,2],4)))}")

def norm3(X):
    c=X.mean(0); d=np.linalg.norm(X-c,axis=1).mean(); s=np.sqrt(3)/d
    T=np.eye(4); T[:3,:3]*=s; T[:3,3]=-s*c; return T
def norm2(x):
    c=x.mean(0); d=np.linalg.norm(x-c,axis=1).mean(); s=np.sqrt(2)/d
    T=np.eye(3); T[:2,:2]*=s; T[:2,2]=-s*c; return T

def dlt(X3,x2):
    T,U=norm3(X3),norm2(x2)
    Xh=(T@np.c_[X3,np.ones(len(X3))].T).T
    xh=(U@np.c_[x2,np.ones(len(x2))].T).T
    A=[]
    for X,x in zip(Xh,xh):
        A.append(np.r_[np.zeros(4),-x[2]*X, x[1]*X])
        A.append(np.r_[x[2]*X, np.zeros(4),-x[0]*X])
    _,_,Vt=np.linalg.svd(np.array(A))
    P=Vt[-1].reshape(3,4)
    return np.linalg.inv(U)@P@T

def rp(P,X3):
    h=(P@np.c_[X3,np.ones(len(X3))].T).T
    return h[:,:2]/h[:,2:3]

P=dlt(P3,P2); e=np.linalg.norm(rp(P,P3)-P2,axis=1)
print(f"\nlinear DLT (no distortion): RMS {np.sqrt((e**2).mean()):.3f} px  max {e.max():.3f}")
print(f"{'point':>16} {'err px':>8}")
for l,x in sorted(zip(lab,e),key=lambda t:-t[1])[:6]: print(f"{l:>16} {x:8.2f}")

M=P[:,:3]; K,R=rq(M)
S=np.diag(np.sign(np.diag(K))); K,R=K@S,S@R
K/=K[2,2]; C=-np.linalg.inv(M)@P[:,3]
print(f"\ndecomposed intrinsics:\n  fx={K[0,0]:8.1f}  fy={K[1,1]:8.1f}  "
      f"(aspect {K[0,0]/K[1,1]:.4f})\n  principal point=({K[0,2]:.1f}, {K[1,2]:.1f})  "
      f"skew={K[0,1]:.2f}")
print(f"  camera centre = X {C[0]:.2f}  Y {C[1]:.2f}  Z {C[2]:.2f} m")
print(f"  (homography-only estimate earlier: X 7.93  Y 23.62, ground)")

# does radial distortion actually help, judged by LEAVE-ONE-OUT generalisation?
def fit_k(idx,nk):
    def fun(p):
        Pm=p[:12].reshape(3,4); k=p[12:12+nk]
        pr=rp(Pm,P3[idx]); r2=((pr[:,0]-K[0,2])**2+(pr[:,1]-K[1,2])**2)/1e6
        s=np.ones(len(pr))
        for i,kk in enumerate(k): s=s+kk*r2**(i+1)
        pr=np.c_[K[0,2]+(pr[:,0]-K[0,2])*s, K[1,2]+(pr[:,1]-K[1,2])*s]
        return (pr-P2[idx]).ravel()
    p0=np.r_[P.ravel(),np.zeros(nk)]
    return least_squares(fun,p0,method='lm',max_nfev=40000).x

print("\nleave-one-out generalisation (held-out reprojection error):")
for nk in (0,1,2):
    errs=[]
    for i in range(len(P3)):
        tr=[j for j in range(len(P3)) if j!=i]
        s=fit_k(tr,nk); Pm=s[:12].reshape(3,4); k=s[12:12+nk]
        pr=rp(Pm,P3[i:i+1]); r2=((pr[:,0]-K[0,2])**2+(pr[:,1]-K[1,2])**2)/1e6
        sc=1.0
        for j,kk in enumerate(k): sc=sc+kk*r2**(j+1)
        pr=np.c_[K[0,2]+(pr[:,0]-K[0,2])*sc, K[1,2]+(pr[:,1]-K[1,2])*sc]
        errs.append(np.linalg.norm(pr[0]-P2[i]))
    tag=" <-- best" if nk==0 else ""
    print(f"  k terms={nk}:  LOO mean {np.mean(errs):6.3f} px   median {np.median(errs):6.3f}")
np.save("out/P_cam.npy",P); np.save("out/K_cam.npy",K)
print("\nsaved out/P_cam.npy, out/K_cam.npy")
