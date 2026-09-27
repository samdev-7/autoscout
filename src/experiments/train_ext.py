"""Train single-class robot detector on EXTERNAL data only. MPS + AMP + RAM cache."""
import time, torch
from ultralytics import YOLO, settings
settings.update({"sync": False})
print("torch",torch.__version__,"| mps",torch.backends.mps.is_available(),flush=True)
t0=time.time()
YOLO("yolo11s.pt").train(
    data="out/yolo_val_full/data.yaml", epochs=60, imgsz=640, batch=16,
    device="mps", workers=8, patience=15, seed=0,
    deterministic=False,        # deterministic MPS kernels fall back to slow paths
    cache="ram",                # workers are forced to 0 on MPS: kill JPEG decode cost
    amp=True,
    project="out/runs", name="ext", exist_ok=True,
    scale=0.5, mosaic=1.0, close_mosaic=10, degrees=0.0,
    fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    plots=True, val=True, verbose=True)
print(f"WALL {time.time()-t0:.0f}s",flush=True)
