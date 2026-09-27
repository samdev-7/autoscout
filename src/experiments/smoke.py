import torch
from ultralytics import YOLO, settings
settings.update({"sync": False})
print("mps:",torch.backends.mps.is_available(),flush=True)
YOLO("yolo11s.pt").train(data="out/arm_A/data.yaml", epochs=2, imgsz=640, batch=16,
    device="mps", workers=8, seed=0, deterministic=False, cache="ram", amp=True,
    project="out/runs", name="smoke", exist_ok=True, val=True, plots=False, verbose=True)
