"""Are misses random (tracking can bridge them) or systematic (it can't)?"""
import json,glob,os,numpy as np,cv2
IN=0.0254;LEN,WID=16.541,8.069;FOOT=36*IN
p=np.load("out/cam_params.npy")
K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]);d=np.zeros(5);d[0]=p[9]
rv,tv=p[3:6],p[6:9];R,_=cv2.Rodrigues(rv);C=-R.T@tv
L=json.load(open("out/labels.json"));B0,B1=L["band"]
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
m=YOLO(W)
GT=json.load(open("out/yolo_val_full/alliance_gt.json"))
CONF=0.15
rows=[]
for stem,gts in GT.items():
    img=f"out/yolo_val_full/images/val/{stem}.jpg"
    r=m.predict(img,imgsz=640,conf=CONF,iou=0.6,device="cpu",verbose=False)[0]
    dets=[list(map(float,b.xyxy[0])) for b in r.boxes]
    used=set()
    for gi,g in enumerate(gts):
        gb=g["box"]; best=0.0; bj=-1
        for j,db in enumerate(dets):
            if j in used: continue
            v=iou(db,gb)
            if v>best: best=v; bj=j
        found = best>=0.5
        if found: used.add(bj)
        # context features
        w=gb[2]-gb[0]; h=gb[3]-gb[1]
        fx,fy=back((gb[0]+gb[2])/2, gb[3]+B0)
        dirv=np.array([fx,fy])-C[:2]; dirv/=np.linalg.norm(dirv)
        fx,fy=np.array([fx,fy])+dirv*(FOOT/2)
        ov=max([iou(gb,o["box"]) for k,o in enumerate(gts) if k!=gi]+[0])
        rows.append(dict(found=found,w=w,h=h,X=fx,Y=fy,c=g["c"],ovl=ov,n=len(gts)))
import collections
R_=[r for r in rows]
tot=len(R_); f=sum(r["found"] for r in R_)
print(f"conf={CONF}  overall recall {f}/{tot} = {f/tot:.3f}\n")
def grp(name,key,bins,labels):
    print(f"{name}:")
    for lo,hi,lab in zip(bins[:-1],bins[1:],labels):
        s=[r for r in R_ if lo<=key(r)<hi]
        if len(s)<8: continue
        print(f"   {lab:>16}  n={len(s):3d}  recall {sum(x['found'] for x in s)/len(s):.3f}")
grp("by field Y (depth: 0=far side, 8.07=near camera)",lambda r:r["Y"],
    [0,2,4,6,9],["0-2m (far)","2-4m","4-6m","6-8m (near)"])
grp("by box width (px)",lambda r:r["w"],[0,80,100,120,400],
    ["<80","80-100","100-120",">120"])
grp("by overlap with another robot",lambda r:r["ovl"],[0,0.01,0.1,1.0],
    ["none","slight","overlapping"])
grp("by robots labelled in frame",lambda r:r["n"],[0,5,6,7],["<5","5","6"])
for c in "RB":
    s=[r for r in R_ if r["c"]==c]
    print(f"\nalliance {c}: recall {sum(x['found'] for x in s)/len(s):.3f} (n={len(s)})")
