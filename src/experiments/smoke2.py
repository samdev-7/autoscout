import sys,time
from ultralytics import YOLO, settings
settings.update({"sync": False})
name,data=sys.argv[1],sys.argv[2]
t0=time.time()
YOLO("yolo11s.pt").train(data=data,epochs=2,imgsz=640,batch=16,device="mps",workers=8,
    seed=0,deterministic=False,cache="ram",amp=True,patience=10000,close_mosaic=0,
    scale=0.5,mosaic=1.0,degrees=0.0,fliplr=0.5,flipud=0.0,erasing=0.4,
    project="out/runs",name=name,exist_ok=True,val=True,plots=False,verbose=True)
print(f"SMOKE {name} {time.time()-t0:.0f}s",flush=True)
