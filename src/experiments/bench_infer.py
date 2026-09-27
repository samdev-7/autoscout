"""Inference speed on this Mac: model size x resolution x backend."""
import time,glob,os,numpy as np,torch
from ultralytics import YOLO, settings
settings.update({"sync": False})
IMGS=sorted(glob.glob("out/yolo_val_full/images/val/*.jpg"))[:24]
def bench(m,imgsz,dev,n=3):
    m.predict(IMGS[0],imgsz=imgsz,device=dev,verbose=False)      # warmup
    t=[]
    for _ in range(n):
        t0=time.time()
        for p in IMGS: m.predict(p,imgsz=imgsz,device=dev,verbose=False)
        t.append((time.time()-t0)/len(IMGS))
    return min(t)*1000
OURS=max(glob.glob("**/runs/fullB/weights/best.pt",recursive=True),key=os.path.getmtime)
print(f"{'model':22}{'imgsz':>7}{'device':>8}{'ms/frame':>11}{'fps':>8}{'165s match':>12}")
rows=[]
for tag,w in (("ours (yolo11s)",OURS),("yolo11m",  "yolo11m.pt"),("yolo11l","yolo11l.pt")):
    for imgsz in (640,960):
        for dev in ("mps","cpu"):
            if dev=="cpu" and imgsz==960 and tag!="ours (yolo11s)": continue
            try:
                m=YOLO(w); ms=bench(m,imgsz,dev)
                fps=1000/ms; mins=4966/fps/60
                print(f"{tag:22}{imgsz:>7}{dev:>8}{ms:>11.1f}{fps:>8.1f}{mins:>10.1f}m")
                rows.append((tag,imgsz,dev,ms))
            except Exception as e:
                print(f"{tag:22}{imgsz:>7}{dev:>8}   FAILED {type(e).__name__}")
print("\n('165s match' = 4966 frames at 30fps, single stream, no batching)")
