"""Leave-one-out outlier analysis.

Trap: dropping ANY point lowers reprojection RMS, because RMS is measured on the
same points that were fitted. The honest test is the held-out AprilTag error,
which never enters the fit. Same lesson as the orientation check.
"""
import json, numpy as np, cv2
LEN,WID,IN=16.541,8.069,0.0254
pts=json.load(open("out/points_corrected.json"))
img=np.array([[p["u"],p["v"]] for p in pts],float)
fld=np.array([[p["X"],p["Y"]] for p in pts],float)
TAGS=[(2,616.8,469.111*IN,182.6*IN),(11,583.3,483.111*IN,182.6*IN),(21,1303.8,182.111*IN,182.6*IN)]

def ev(F,I):
    H,_=cv2.findHomography(F,I,0)
    pr=cv2.perspectiveTransform(F.reshape(-1,1,2),H).reshape(-1,2)
    e=np.linalg.norm(pr-I,axis=1)
    du=[abs(cv2.perspectiveTransform(np.array([[[X,Y]]],float),H).reshape(2)[0]-u)
        for _,u,X,Y in TAGS]
    return H,np.sqrt((e**2).mean()),e,np.mean(du)

H,rms,res,tagdu=ev(fld,img)
print(f"ALL 11 points:  reproj RMS {rms:.3f} px   held-out tag |du| {tagdu:.2f} px\n")
med=np.median(res); mad=np.median(np.abs(res-med))
print(f"{'pt':>3} {'u':>8} {'v':>7} {'resid':>7} {'robust z':>9} | drop it ->"
      f" {'RMS':>7} {'tag|du|':>8}")
for i in range(len(pts)):
    z=0.6745*(res[i]-med)/mad if mad>0 else 0
    k=[j for j in range(len(pts)) if j!=i]
    _,r2,_,t2=ev(fld[k],img[k])
    flag=" <-- helps" if t2<tagdu-0.15 else (" (hurts)" if t2>tagdu+0.15 else "")
    print(f"{i+1:3d} {pts[i]['u']:8.1f} {pts[i]['v']:7.1f} {res[i]:7.2f} {z:9.2f} |"
          f"           {r2:7.3f} {t2:8.2f}{flag}")

print("\nnote: every 'drop' lowers RMS — that is guaranteed, not evidence.")
for name,meth in (("RANSAC(3px)",cv2.RANSAC),("LMEDS",cv2.LMEDS)):
    Hr,mask=cv2.findHomography(fld,img,meth,3.0)
    keep=mask.ravel().astype(bool)
    pr=cv2.perspectiveTransform(fld.reshape(-1,1,2),Hr).reshape(-1,2)
    e=np.linalg.norm(pr-img,axis=1)
    du=[abs(cv2.perspectiveTransform(np.array([[[X,Y]]],float),Hr).reshape(2)[0]-u)
        for _,u,X,Y in TAGS]
    print(f"  {name:12} keeps {keep.sum()}/11  RMS {np.sqrt((e[keep]**2).mean()):.3f}  "
          f"held-out tag |du| {np.mean(du):.2f} px  rejected={[i+1 for i in range(11) if not keep[i]]}")
