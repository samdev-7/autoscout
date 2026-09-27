"""How much bias does an axis-aligned box introduce, and can we correct it?

Simulate a robot of known footprint at known field pose, project its silhouette,
take the AABB the way a labeller would, then back-project the bottom-centre and
compare with truth. Then test a footprint-radius correction that needs no
orientation knowledge.
"""
import numpy as np, cv2
IN=0.0254; LEN,WID=16.541,8.069
p=np.load("out/cam_params.npy")
K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]); d=np.zeros(5); d[0]=p[9]
rv,tv=p[3:6],p[6:9]; R,_=cv2.Rodrigues(rv); C=-R.T@tv

FOOT=36*IN      # bumper-to-bumper footprint (in)
BODY=30*IN      # body height
def silhouette(X,Y,th):
    h=FOOT/2; c,s=np.cos(th),np.sin(th)
    base=[(X+c*dx-s*dy, Y+s*dx+c*dy) for dx,dy in
          [(-h,-h),(h,-h),(h,h),(-h,h)]]
    pts=[(x,y,0.0) for x,y in base]+[(x,y,BODY) for x,y in base]
    q,_=cv2.projectPoints(np.array(pts,float).reshape(-1,1,3),rv,tv,K,d)
    return q.reshape(-1,2)
def back(u,v,Z=0.0):
    n=cv2.undistortPoints(np.array([[[u,v]]],float),K,d).reshape(2)
    ray=R.T@np.array([n[0],n[1],1.0]); t=(Z-C[2])/ray[2]
    return C[:2]+t*ray[:2]

raw,cor=[],[]
for X in (2.5,5.5,8.3,11.5,14.5):
    for Y in (1.5,4.0,6.5):
        for th in np.radians([0,15,30,45,60,75]):
            q=silhouette(X,Y,th)
            u0,v0=q[:,0].min(),q[:,1].min(); u1,v1=q[:,0].max(),q[:,1].max()
            bc=np.array([(u0+u1)/2, v1])            # AABB bottom-centre
            est=back(*bc)
            raw.append(np.linalg.norm(est-np.array([X,Y]))/IN)
            # correction: push away from camera by the footprint radius
            dirv=est-C[:2]; dirv/=np.linalg.norm(dirv)
            est2=est+dirv*(FOOT/2)
            cor.append(np.linalg.norm(est2-np.array([X,Y]))/IN)
raw,cor=np.array(raw),np.array(cor)
print(f"samples: {len(raw)}  (5 field X x 3 field Y x 6 orientations)")
print(f"\nnaive AABB bottom-centre -> ground:")
print(f"  bias  mean {raw.mean():6.1f} in   median {np.median(raw):6.1f}   "
      f"range {raw.min():.1f}-{raw.max():.1f}")
print(f"\nafter footprint-radius correction (no orientation needed):")
print(f"  error mean {cor.mean():6.1f} in   median {np.median(cor):6.1f}   "
      f"range {cor.min():.1f}-{cor.max():.1f}")
print(f"\n  -> correction removes {100*(1-cor.mean()/raw.mean()):.0f}% of the bias")
print(f"  residual is orientation-dependent (square footprint vs disc model):")
for th in (0,15,30,45):
    m=[]
    for X in (2.5,8.3,14.5):
        for Y in (1.5,4.0,6.5):
            q=silhouette(X,Y,np.radians(th))
            bc=np.array([(q[:,0].min()+q[:,0].max())/2, q[:,1].max()])
            e=back(*bc); dv=e-C[:2]; dv/=np.linalg.norm(dv)
            m.append(np.linalg.norm(e+dv*(FOOT/2)-np.array([X,Y]))/IN)
    print(f"    yaw {th:2d} deg : {np.mean(m):5.1f} in")
print(f"\nfor scale: robot footprint {FOOT/IN:.0f} in, calibration accuracy ~2 in")
