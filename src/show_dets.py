"""Render detections from the winning model vs ground truth. CPU, so training is undisturbed."""
import glob,os,json,cv2,numpy as np
from ultralytics import YOLO, settings
settings.update({"sync":False})
W=max(glob.glob("**/runs/fullB/weights/best.pt",recursive=True),key=os.path.getmtime)
print("weights:",W)
m=YOLO(W)
CONF=0.25
def alliance(img,b):
    x0,y0,x1,y1=[int(v) for v in b]
    x0,y0=max(0,x0),max(0,y0); x1,y1=min(img.shape[1],x1),min(img.shape[0],y1)
    if x1<=x0 or y1<=y0: return "?"
    c=img[y0+int(.45*(y1-y0)):y1, x0:x1]
    if c.size==0: return "?"
    h=cv2.cvtColor(c,cv2.COLOR_BGR2HSV); H,S,V=h[...,0].astype(int),h[...,1].astype(int),h[...,2].astype(int)
    sat=(S>95)&(V>45)
    r=(((H<=9)|(H>=168))&sat).sum(); bl=((H>=100)&(H<=132)&sat).sum()
    return "?" if r+bl<12 else ("R" if r>bl else "B")
def draw(path,gtbox,title):
    im=cv2.imread(path)
    for g in gtbox:
        cv2.rectangle(im,(int(g[0]),int(g[1])),(int(g[2]),int(g[3])),(170,170,170),1)
    r=m.predict(path,imgsz=640,conf=CONF,iou=0.6,device="cpu",verbose=False)[0]
    n=0
    for b in r.boxes:
        x=list(map(float,b.xyxy[0])); cf=float(b.conf[0]); n+=1
        a=alliance(im,x)
        col=(60,60,255) if a=="R" else ((255,160,60) if a=="B" else (0,255,255))
        cv2.rectangle(im,(int(x[0]),int(x[1])),(int(x[2]),int(x[3])),col,2)
        cv2.circle(im,(int((x[0]+x[2])/2),int(x[3])),3,(255,255,255),-1)
        cv2.putText(im,f"{a} {cf:.2f}",(int(x[0]),int(x[1])-4),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1,cv2.LINE_AA)
    cv2.putText(im,f"{title}   {n} detected / {len(gtbox)} labelled   grey=ground truth",
                (8,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2,cv2.LINE_AA)
    return im
GT=json.load(open("out/yolo_val_full/alliance_gt.json"))
tiles=[]
for stem in ["f020","f045","f070"]:
    if stem in GT:
        tiles.append(draw(f"out/yolo_val_full/images/val/{stem}.jpg",
                          [g["box"] for g in GT[stem]], f"OUR MATCH {stem}"))
for f in sorted(glob.glob("out/val_multi/images/val/*.jpg"))[:3]:
    st=os.path.splitext(os.path.basename(f))[0]
    lb=f"out/val_multi/labels/val/{st}.txt"
    im0=cv2.imread(f); H,W_=im0.shape[:2]
    gb=[]
    for l in open(lb):
        p=l.split()
        if len(p)>=5:
            x,y,bw,bh=[float(v) for v in p[1:5]]
            gb.append([(x-bw/2)*W_,(y-bh/2)*H,(x+bw/2)*W_,(y+bh/2)*H])
    tiles.append(draw(f,gb,f"UNSEEN EVENT {st}"))
out=[]
for t in tiles:
    s=900/t.shape[1]; out.append(cv2.resize(t,(900,int(t.shape[0]*s))))
Hm=sum(t.shape[0] for t in out)
cv2.imwrite("out/detections.jpg",np.vstack(out),[cv2.IMWRITE_JPEG_QUALITY,86])
print("wrote out/detections.jpg")
