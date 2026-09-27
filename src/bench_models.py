"""Evaluate every local detector on every validation set.

Locked-down settings so a later cloud model can be dropped into the same table:
each model is measured at its own training imgsz AND at 960, on all three val
sets, with identical conf/iou.  Results are appended to a JSON so comparisons
survive across sessions.
"""
import os, json, time, sys
import numpy as np
from ultralytics import YOLO

SETS = [("ours", "out/yolo_val_full/data.yaml"),
        ("multi", "out/val_multi/data.yaml"),
        ("combined", "out/val_combined/data.yaml")]
MODELS = [("fullA", "runs/detect/out/runs/fullA/weights/best.pt", 640),
          ("fullB", "runs/detect/out/runs/fullB/weights/best.pt", 640),
          ("fullB_960", "runs/detect/out/runs/fullB_960/weights/best.pt", 960),
          ("ext", "runs/detect/out/runs/ext/weights/best.pt", 640)]
CONF, IOU, DEV = 0.001, 0.6, "mps"

rows = []
for name, w, native in MODELS:
    if not os.path.exists(w):
        continue
    for imgsz in sorted({native, 960}):
        m = YOLO(w)
        for sn, y in SETS:
            t0 = time.time()
            r = m.val(data=y, imgsz=imgsz, conf=CONF, iou=IOU, device=DEV,
                      verbose=False, plots=False, save_json=False)
            b = r.box
            rows.append(dict(model=name, weights=w, imgsz=imgsz, valset=sn,
                             map50=float(b.map50), map=float(b.map),
                             precision=float(b.mp), recall=float(b.mr),
                             secs=round(time.time() - t0, 1)))
            print(f"{name:10s} @{imgsz:4d} {sn:9s}  mAP50 {b.map50:.3f}  "
                  f"mAP50-95 {b.map:.3f}  P {b.mp:.3f}  R {b.mr:.3f}", flush=True)
json.dump(rows, open("out/bench_models.json", "w"), indent=1)
print(f"\nsaved out/bench_models.json ({len(rows)} rows)")
