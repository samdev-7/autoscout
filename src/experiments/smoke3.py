import sys,time,json,os,glob
from ultralytics import YOLO, settings
settings.update({"sync": False})
name,data=sys.argv[1],sys.argv[2]
t0=time.time()
m=YOLO("yolo11s.pt")
m.train(data=data,epochs=2,imgsz=640,batch=16,device="mps",workers=8,seed=0,
        deterministic=False,cache="ram",amp=True,patience=10000,close_mosaic=0,
        scale=0.5,mosaic=1.0,degrees=0.0,fliplr=0.5,flipud=0.0,erasing=0.4,
        project="out/runs",name=name,exist_ok=True,val=True,plots=False,verbose=True)
w=max(glob.glob(f"**/runs/{name}/weights/best.pt",recursive=True),key=os.path.getmtime)
res={}
for tag,y in (("ours","out/yolo_val_full/data.yaml"),("multi","out/val_multi/data.yaml")):
    r=YOLO(w).val(data=y,imgsz=640,device="mps",verbose=False,plots=False,split="val")
    b=r.box
    res[tag]=dict(P=round(float(b.mp),4),R=round(float(b.mr),4),
                  mAP50=round(float(b.map50),4),mAP=round(float(b.map),4))
print("RESULT "+name+" "+json.dumps(res),flush=True)
json.dump(res,open(f"out/smoke_{name}.json","w"))
print(f"SMOKE {name} {time.time()-t0:.0f}s",flush=True)
