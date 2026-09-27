"""Warm-start from the 960 checkpoint, continue at 640 where the curve was still rising."""
import os,glob,time,json
def log(m):
    s=f"[{time.strftime('%H:%M:%S')}] {m}"; print(s,flush=True); open("out/full.log","a").write(s+"\n")
from ultralytics import YOLO, settings
settings.update({"sync": False})
w=None
for c in ("best.pt","last.pt"):
    p=f"runs/detect/out/runs/fullB_960/weights/{c}"
    if os.path.exists(p): w=p; break
if not w: w=max(glob.glob("**/runs/fullB/weights/best.pt",recursive=True),key=os.path.getmtime)
log(f"640 continuation warm-starting from {w}")
YOLO(w).train(data="out/armB/data.yaml",epochs=60,imgsz=640,batch=16,device="mps",
    workers=8,seed=0,deterministic=False,cache="ram",amp=True,patience=20,save_period=5,
    close_mosaic=15,scale=0.5,mosaic=1.0,degrees=0.0,fliplr=0.5,flipud=0.0,
    hsv_h=0.015,hsv_s=0.7,hsv_v=0.4,erasing=0.4,
    project="out/runs",name="fullB_640cont",exist_ok=True,val=True,plots=False,verbose=True)
log("640 continuation finished")
bw=max(glob.glob("**/runs/fullB_640cont/weights/best.pt",recursive=True),key=os.path.getmtime)
res={"name":"fullB_640cont","weights":bw,"imgsz":640}
for tag,y in (("ours","out/yolo_val_full/data.yaml"),("multi","out/val_multi/data.yaml")):
    b=YOLO(bw).val(data=y,imgsz=640,device="mps",verbose=False,plots=False,split="val").box
    res[tag]=dict(P=round(float(b.mp),4),R=round(float(b.mr),4),
                  mAP50=round(float(b.map50),4),mAP=round(float(b.map),4))
log(f"640cont  OURS mAP50 {res['ours']['mAP50']:.3f} R {res['ours']['R']:.3f} | "
    f"MULTI mAP50 {res['multi']['mAP50']:.3f} R {res['multi']['R']:.3f}")
r=json.load(open("out/results.json")) if os.path.exists("out/results.json") else []
r.append(res); json.dump(r,open("out/results.json","w"),indent=1)
