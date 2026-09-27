"""Post-training evaluation: cross-event transfer + bumper-hue alliance accuracy."""
import json, glob, os, sys, numpy as np, cv2

def find_best():
    c=glob.glob("**/runs/ext/weights/best.pt",recursive=True)
    return max(c,key=os.path.getmtime) if c else None
W=find_best()
out=open("out/eval_summary.txt","w")
def say(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); out.write(s+"\n")

if not W: say("NO best.pt FOUND — training did not produce weights"); out.close(); sys.exit(1)
say(f"weights: {W}  ({os.path.getsize(W)/2**20:.1f} MiB)")

from ultralytics import YOLO, settings
settings.update({"sync": False})
m=YOLO(W)

GT=json.load(open("out/yolo_val_full/alliance_gt.json"))
imgs=sorted(glob.glob("out/yolo_val_full/images/val/*.jpg"))
say(f"val: {len(imgs)} frames, {sum(len(v) for v in GT.values())} boxes\n")

def iou(a,b):
    x0=max(a[0],b[0]); y0=max(a[1],b[1]); x1=min(a[2],b[2]); y1=min(a[3],b[3])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)

# cache predictions once at low conf, then sweep thresholds
preds={}
for p in imgs:
    r=m.predict(p,imgsz=640,conf=0.05,iou=0.6,device="mps",verbose=False)[0]
    preds[os.path.basename(p)[:-4]]=[(list(map(float,b.xyxy[0])),float(b.conf[0]))
                                     for b in r.boxes]
say(f"{'conf':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'precision':>10} {'recall':>8} {'F1':>7}")
best=None
for c in (0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60):
    TP=FP=FN=0
    for stem,gts in GT.items():
        g=[x["box"] for x in gts]
        d=[b for b,cf in preds.get(stem,[]) if cf>=c]
        used=set()
        for db in d:
            hit=-1; bi=0.5
            for i,gb in enumerate(g):
                if i in used: continue
                v=iou(db,gb)
                if v>=bi: bi=v; hit=i
            if hit>=0: used.add(hit); TP+=1
            else: FP+=1
        FN+=len(g)-len(used)
    pr=TP/max(TP+FP,1); rc=TP/max(TP+FN,1); f1=2*pr*rc/max(pr+rc,1e-9)
    say(f"{c:6.2f} {TP:5d} {FP:5d} {FN:5d} {pr:10.3f} {rc:8.3f} {f1:7.3f}")
    if best is None or f1>best[1]: best=(c,f1,pr,rc)
say(f"\nbest F1 {best[1]:.3f} at conf {best[0]:.2f}  (precision {best[2]:.3f}, recall {best[3]:.3f})")

# ---- bumper-hue alliance classifier, measured on GROUND-TRUTH boxes ----
def alliance(img,box):
    x0,y0,x1,y1=[int(v) for v in box]
    x0,y0=max(0,x0),max(0,y0); x1,y1=min(img.shape[1],x1),min(img.shape[0],y1)
    if x1<=x0 or y1<=y0: return None
    crop=img[y0+int(0.45*(y1-y0)):y1, x0:x1]          # lower half = bumpers
    if crop.size==0: return None
    h=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    H,S,V=h[...,0].astype(int),h[...,1].astype(int),h[...,2].astype(int)
    sat=(S>95)&(V>45)
    red=(((H<=9)|(H>=168))&sat).sum(); blue=((H>=100)&(H<=132)&sat).sum()
    if red+blue < 12: return None
    return "R" if red>blue else "B"
ok=tot=abst=0; conf=np.zeros((2,2),int); idx={"R":0,"B":1}
for stem,gts in GT.items():
    img=cv2.imread(f"out/yolo_val_full/images/val/{stem}.jpg")
    for x in gts:
        a=alliance(img,x["box"]); tot+=1
        if a is None: abst+=1; continue
        conf[idx[x["c"]],idx[a]]+=1
        ok+= (a==x["c"])
say(f"\nbumper-hue alliance on ground-truth boxes:")
say(f"  accuracy {ok}/{tot-abst} = {100*ok/max(tot-abst,1):.1f}%   abstained {abst} ({100*abst/tot:.1f}%)")
say(f"  confusion  [true R -> R {conf[0,0]}, B {conf[0,1]}] [true B -> R {conf[1,0]}, B {conf[1,1]}]")

# ---- qualitative montage ----
tiles=[]
for stem in list(GT)[::22][:4]:
    img=cv2.imread(f"out/yolo_val_full/images/val/{stem}.jpg")
    for x in GT[stem]:
        b=[int(v) for v in x["box"]]
        cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(200,200,200),1)
    for b,cf in preds.get(stem,[]):
        if cf<best[0]: continue
        b=[int(v) for v in b]; a=alliance(img,b) or "?"
        col=(0,0,255) if a=="R" else ((255,150,0) if a=="B" else (0,255,255))
        cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),col,2)
        cv2.putText(img,f"{a}{cf:.2f}",(b[0],b[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1,cv2.LINE_AA)
    cv2.putText(img,f"{stem}  grey=ground truth  colour=detection",(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
    tiles.append(img)
if tiles: cv2.imwrite("out/eval_detections.jpg",np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,88])
say("\nwrote out/eval_detections.jpg")
out.close()
