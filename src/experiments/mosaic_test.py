"""Controlled test: same pool, same 2 epochs, mosaic OFF (close_mosaic>=epochs)."""
import json,os,glob,time
from ultralytics import YOLO, settings
settings.update({"sync": False})
t0=time.time()
YOLO("yolo11s.pt").train(data="out/armA/data.yaml",epochs=2,imgsz=640,batch=16,
    device="mps",workers=8,seed=0,deterministic=False,
    cache=False,                      # no RAM cache: arm B is holding ~5.9GB
    amp=True,patience=10000,
    close_mosaic=10,                  # >= epochs  ->  mosaic disabled throughout
    scale=0.5,mosaic=1.0,degrees=0.0,fliplr=0.5,flipud=0.0,erasing=0.4,
    project="out/runs",name="mosaicoff",exist_ok=True,val=True,plots=False,verbose=True)
w=max(glob.glob("**/runs/mosaicoff/weights/best.pt",recursive=True),key=os.path.getmtime)
res={}
for tag,y in (("ours","out/yolo_val_full/data.yaml"),("multi","out/val_multi/data.yaml")):
    b=YOLO(w).val(data=y,imgsz=640,device="mps",verbose=False,plots=False,split="val").box
    res[tag]=dict(P=round(float(b.mp),4),R=round(float(b.mr),4),
                  mAP50=round(float(b.map50),4),mAP=round(float(b.map),4))
print("RESULT mosaicoff "+json.dumps(res),flush=True)
print(f"took {time.time()-t0:.0f}s",flush=True)
