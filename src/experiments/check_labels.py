"""Validate labels: counts, field projection, and apparent-size plausibility."""
import json, numpy as np, cv2
IN=0.0254; LEN,WID=16.541,8.069; FOOT=36*IN
p=np.load("out/cam_params.npy")
K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]); d=np.zeros(5); d[0]=p[9]
rv,tv=p[3:6],p[6:9]; R,_=cv2.Rodrigues(rv); C=-R.T@tv

def back(u,v,Z=0.0):
    n=cv2.undistortPoints(np.array([[[u,v]]],float),K,d).reshape(2)
    ray=R.T@np.array([n[0],n[1],1.0]); t=(Z-C[2])/ray[2]
    return C[:2]+t*ray[:2]
def fwd(X,Y,Z=0.0):
    q,_=cv2.projectPoints(np.array([[[X,Y,Z]]]),rv,tv,K,d); return q.reshape(2)

L=json.load(open("out/labels.json")); F=L["frames"]
print(f"{len(F)} frames, {sum(len(v) for v in F.values())} boxes\n")

# --- 1. per-alliance counts: only 3 robots exist per alliance ---
bad=[]
for f,bs in sorted(F.items()):
    r=sum(1 for b in bs if b["c"]=="R"); bl=len(bs)-r
    if r>3 or bl>3: bad.append((f,r,bl))
print(f"frames exceeding 3 robots on an alliance: {len(bad)}")
for f,r,bl in bad[:10]: print(f"   {f}: {r} red, {bl} blue  <-- impossible")

cnt=np.array([len(v) for v in F.values()])
print(f"boxes/frame: mean {cnt.mean():.2f}  median {np.median(cnt):.0f}  "
      f"min {cnt.min()}  max {cnt.max()}")
print(f"  frames with 6 boxes: {(cnt==6).sum()}   <6: {(cnt<6).sum()}   >6: {(cnt>6).sum()}")

# --- 2. project bottom-centre to the field, with footprint-radius correction ---
pos=[]; oob=0
for f,bs in F.items():
    for b in bs:
        u=(b["x0"]+b["x1"])/2; v=b["y1"]
        e=back(u,v)
        dirv=e-C[:2]; dirv/=np.linalg.norm(dirv)
        g=e+dirv*(FOOT/2)
        inside = -0.5<=g[0]<=LEN+0.5 and -0.5<=g[1]<=WID+0.5
        if not inside: oob+=1
        pos.append((f,b["c"],g[0],g[1],b["x1"]-b["x0"],b["y1"]-b["y0"],u,v))
P=np.array([[x,y] for _,_,x,y,_,_,_,_ in pos])
print(f"\nfield projection of {len(P)} boxes:")
print(f"  X range {P[:,0].min():6.2f} .. {P[:,0].max():6.2f}  (field 0..{LEN:.2f})")
print(f"  Y range {P[:,1].min():6.2f} .. {P[:,1].max():6.2f}  (field 0..{WID:.2f})")
print(f"  outside field (>0.5m margin): {oob}  ({100*oob/len(P):.1f}%)")

# --- 3. apparent size: does box width match a 36in robot at that distance? ---
print(f"\napparent-size check (expected width of a {FOOT/IN:.0f}in robot):")
rows=[]
for f,c,X,Y,w,h,u,v in pos:
    if not(0<=X<=LEN and 0<=Y<=WID): continue
    dirv=np.array([X,Y])-C[:2]; dirv/=np.linalg.norm(dirv)
    perp=np.array([-dirv[1],dirv[0]])*(FOOT/2)
    a=fwd(X+perp[0],Y+perp[1]); bb=fwd(X-perp[0],Y-perp[1])
    exp=np.linalg.norm(a-bb)
    rows.append((w,exp,w/exp))
rows=np.array(rows)
print(f"  labelled width  mean {rows[:,0].mean():5.1f} px")
print(f"  expected width  mean {rows[:,1].mean():5.1f} px")
print(f"  ratio labelled/expected: median {np.median(rows[:,2]):.2f}  "
      f"16-84pct {np.percentile(rows[:,2],16):.2f}-{np.percentile(rows[:,2],84):.2f}")

# --- 4. duplicate / overlapping boxes within a frame ---
def iou(a,b):
    x0=max(a["x0"],b["x0"]); x1=min(a["x1"],b["x1"])
    y0=max(a["y0"],b["y0"]); y1=min(a["y1"],b["y1"])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0)
    A=(a["x1"]-a["x0"])*(a["y1"]-a["y0"]); B=(b["x1"]-b["x0"])*(b["y1"]-b["y0"])
    return i/(A+B-i)
dup=[(f,round(iou(bs[i],bs[j]),2)) for f,bs in F.items()
     for i in range(len(bs)) for j in range(i+1,len(bs)) if iou(bs[i],bs[j])>0.55]
print(f"\nbox pairs with IoU>0.55 (possible double-labels): {len(dup)}")
for f,v in dup[:8]: print(f"   {f}: IoU {v}")
np.save("out/label_positions.npy",np.array([[X,Y] for _,_,X,Y,_,_,_,_ in pos]))
json.dump([{"f":f,"c":c,"X":X,"Y":Y} for f,c,X,Y,_,_,_,_ in pos],
          open("out/label_field_pos.json","w"))
print("\nwrote out/label_field_pos.json")
