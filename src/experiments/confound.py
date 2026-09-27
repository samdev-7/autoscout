"""Separate model bias from validation-data properties.

Prior analysis used robot-on-robot IoU as the occlusion proxy, which misses
occlusion by guardrail/structures/people. Here: stratify alliance by depth, and
use image sharpness inside each box as a motion-blur / visibility proxy.
"""
import json,glob,os,numpy as np,cv2
IN=0.0254;LEN,WID=16.541,8.069;FOOT=36*IN
p=np.load("out/cam_params.npy")
K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]);d=np.zeros(5);d[0]=p[9]
rv,tv=p[3:6],p[6:9];R,_=cv2.Rodrigues(rv);C=-R.T@tv
L=json.load(open("out/labels.json"));B0,B1=L["band"]
man={m["file"][:-4]:m["t"] for m in json.load(open("out/frames.json"))["frames"]}
def back(u,v):
    n=cv2.undistortPoints(np.array([[[u,v]]],float),K,d).reshape(2)
    ray=R.T@np.array([n[0],n[1],1.0]);t=-C[2]/ray[2];return C[:2]+t*ray[:2]
def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
W=max(glob.glob("**/runs/ext/weights/best.pt",recursive=True),key=os.path.getmtime)
from ultralytics import YOLO,settings; settings.update({"sync":False})
m=YOLO(W); GT=json.load(open("out/yolo_val_full/alliance_gt.json"))
rows=[]
for stem,gts in GT.items():
    path=f"out/yolo_val_full/images/val/{stem}.jpg"
    img=cv2.imread(path); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dets=[list(map(float,b.xyxy[0])) for b in
          m.predict(path,imgsz=640,conf=0.15,iou=0.6,device="cpu",verbose=False)[0].boxes]
    used=set()
    for gi,g in enumerate(gts):
        gb=g["box"]; best=0.0;bj=-1
        for j,db in enumerate(dets):
            if j in used: continue
            v=iou(db,gb)
            if v>best: best=v;bj=j
        found=best>=0.5
        if found: used.add(bj)
        x0,y0,x1,y1=[int(v) for v in gb]
        x0,y0=max(0,x0),max(0,y0);x1,y1=min(gray.shape[1],x1),min(gray.shape[0],y1)
        crop=gray[y0:y1,x0:x1]
        sharp=cv2.Laplacian(crop,cv2.CV_64F).var() if crop.size>40 else np.nan
        fx,fy=back((gb[0]+gb[2])/2,gb[3]+B0)
        dv=np.array([fx,fy])-C[:2];dv/=np.linalg.norm(dv)
        fx,fy=np.array([fx,fy])+dv*(FOOT/2)
        rows.append(dict(found=found,c=g["c"],Y=fy,X=fx,sharp=sharp,
                         w=gb[2]-gb[0],stem=stem,t=man.get(stem,0)))
def rec(s): return (sum(x["found"] for x in s)/len(s), len(s)) if s else (float('nan'),0)
print("ALLIANCE x DEPTH  (does the red gap survive controlling for position?)")
print(f"{'Y band':>14} {'red':>16} {'blue':>16}")
for lo,hi,lab in [(0,2,"0-2m far"),(2,4,"2-4m"),(4,6,"4-6m"),(6,9,"6-8m near")]:
    a,na=rec([r for r in rows if r["c"]=="R" and lo<=r["Y"]<hi])
    b,nb=rec([r for r in rows if r["c"]=="B" and lo<=r["Y"]<hi])
    print(f"{lab:>14}  {a:.3f} (n={na:3d})  {b:.3f} (n={nb:3d})")
ra,_=rec([r for r in rows if r["c"]=="R"]); rb,_=rec([r for r in rows if r["c"]=="B"])
print(f"{'overall':>14}  {ra:.3f}            {rb:.3f}")
print("\ndepth distribution by alliance (were reds in worse positions?)")
for c in "RB":
    ys=np.array([r["Y"] for r in rows if r["c"]==c])
    print(f"  {c}: mean Y {ys.mean():.2f}  frac in near strip (Y>6): {(ys>6).mean():.3f}  n={len(ys)}")
print("\nSHARPNESS inside the box (motion blur / visibility proxy)")
for c in ("all","R","B"):
    s=[r for r in rows if c=="all" or r["c"]==c]
    f=[r["sharp"] for r in s if r["found"] and not np.isnan(r["sharp"])]
    mm=[r["sharp"] for r in s if not r["found"] and not np.isnan(r["sharp"])]
    print(f"  {c:4}: found median {np.median(f):7.1f} (n={len(f):3d})   "
          f"missed median {np.median(mm):7.1f} (n={len(mm):3d})   ratio {np.median(f)/max(np.median(mm),1e-9):.2f}x")
print("\nnear strip only (Y>6), split by sharpness tercile")
ns=[r for r in rows if r["Y"]>6 and not np.isnan(r["sharp"])]
q=np.percentile([r["sharp"] for r in ns],[33,66])
for lo,hi,lab in [(-1,q[0],"blurriest 3rd"),(q[0],q[1],"middle"),(q[1],1e18,"sharpest 3rd")]:
    s=[r for r in ns if lo<=r["sharp"]<hi]; v,n=rec(s)
    print(f"  {lab:>15}: recall {v:.3f} (n={n})")
print("\nare misses concentrated in particular frames?")
per={}
for r in rows: per.setdefault(r["stem"],[]).append(r["found"])
fr=sorted((sum(v)/len(v),k) for k,v in per.items())
print(f"  frames with 0% recall: {sum(1 for v,_ in fr if v==0)}/{len(fr)}")
print(f"  worst 5: {[(k,round(v,2)) for v,k in fr[:5]]}")
print(f"  best  5: {[(k,round(v,2)) for v,k in fr[-5:]]}")
