"""Compare REAL training throughput across settings. s/it is the metric."""
import sys, time, re, subprocess, os
CFGS=[("baseline",       dict(deterministic=True,  cache=False, batch=16)),
      ("nondet",         dict(deterministic=False, cache=False, batch=16)),
      ("nondet+ramcache",dict(deterministic=False, cache="ram", batch=16)),
      ("nondet+cache+b32",dict(deterministic=False,cache="ram", batch=32))]
name=sys.argv[1]
cfg=dict(CFGS)[name]
from ultralytics import YOLO, settings
settings.update({"sync": False})
m=YOLO("yolo11s.pt")
t0=time.time()
m.train(data="out/yolo_val_full/data.yaml", epochs=2, imgsz=640, device="mps",
        workers=8, seed=0, fraction=0.25, val=False, plots=False, verbose=True,
        project="out/runs", name=f"speed_{name}", exist_ok=True, **cfg)
print(f"RESULT {name} wall={time.time()-t0:.1f}s cfg={cfg}", flush=True)
