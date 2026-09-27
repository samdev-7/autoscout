import os, time, torch
os.environ["YOLO_VERBOSE"]="true"
from ultralytics import YOLO, settings
settings.update({"sync": False})          # no telemetry
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
t0=time.time()
m=YOLO("yolo11s.pt")
r=m.train(data="out/yolo/data.yaml", epochs=140, imgsz=640, batch=16,
          device="mps", workers=4, patience=35, seed=0,
          project="out/runs", name="baseline", exist_ok=True,
          scale=0.6, mosaic=0.8, close_mosaic=15, degrees=0.0,
          fliplr=0.5, flipud=0.0, hsv_h=0.012, hsv_s=0.6, hsv_v=0.35,
          plots=True, val=True, verbose=False)
print(f"\nTRAIN WALL TIME {time.time()-t0:.0f}s")
