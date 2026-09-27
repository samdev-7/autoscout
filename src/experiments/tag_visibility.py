"""Locate the camera, then recompute which tags actually face it."""
import numpy as np, cv2, json, csv
IN=0.0254; LEN,WID=16.541,8.069
H=np.load("out/H_field_to_img.npy"); Hi=np.linalg.inv(H)
rows=[]
with open("data/field/2026-rebuilt-welded.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rows.append(dict(id=int(r["ID"]),X=float(r["X"])*IN,Y=float(r["Y"])*IN,
                         Z=float(r["Z"])*IN,rz=float(r["Z-Rotation"])))

# vertical vanishing point (nadir) from the 3 tag/carpet pairs, then camera ground pos
TAG=[(2,616.8,210.9),(11,583.3,211.2),(21,1303.8,204.6)]
A=[];b=[]
for tid,uo,vo in TAG:
    t=next(r for r in rows if r["id"]==tid)
    p=cv2.perspectiveTransform(np.array([[[t["X"],t["Y"]]]],float),H).reshape(2)
    dx,dy=uo-p[0],vo-p[1]                      # image direction of "up" at that spot
    A.append([dy,-dx]); b.append(dy*p[0]-dx*p[1])
V,_,_,_=np.linalg.lstsq(np.array(A,float),np.array(b,float),rcond=None)
g=cv2.perspectiveTransform(np.array([[V]],float),Hi).reshape(2)
print(f"nadir (vertical VP) in image : ({V[0]:.0f}, {V[1]:.0f})")
print(f"camera ground position        : X={g[0]:.2f} m  Y={g[1]:.2f} m "
      f"(field is 0..{LEN:.2f} x 0..{WID:.2f})")

# facing test: angle between each tag's outward normal and the direction to camera
NRM={0:(1,0),90:(0,1),180:(-1,0),270:(0,-1)}
print(f"\n{'id':>4} {'rz':>5} {'X m':>6} {'Y m':>6} {'angle':>7}  {'u_carpet':>9} {'v_carpet':>8}  facing")
vis=[]
for r in sorted(rows,key=lambda r:r["id"]):
    n=np.array(NRM[r["rz"]],float)
    d=np.array([g[0]-r["X"], g[1]-r["Y"]]); d/=np.linalg.norm(d)
    ang=np.degrees(np.arccos(np.clip(n@d,-1,1)))
    p=cv2.perspectiveTransform(np.array([[[r["X"],r["Y"]]]],float),H).reshape(2)
    onscreen = -60<p[0]<1980 and 0<p[1]<560
    if ang<75 and onscreen:
        vis.append(r["id"])
        print(f"{r['id']:4d} {r['rz']:5.0f} {r['X']:6.2f} {r['Y']:6.2f} {ang:6.1f}d "
              f"{p[0]:9.1f} {p[1]:8.1f}  FACES CAMERA")
print(f"\ntags facing the camera and on-screen: {vis}")
print("previously detected: [2, 11, 21]")
print(f"NEW candidates to mark: {[i for i in vis if i not in (2,11,21)]}")
