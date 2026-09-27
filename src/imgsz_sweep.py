"""Does inference resolution matter? The model trained at 640 but our panel is
1920x344, so letterboxing to 640 downscales robots 3x before they reach the net.
Higher imgsz may recover small-object recall -- or may hurt, because objects then
appear at a scale the model never trained on.  Measure, don't guess.
"""
import numpy as np, glob, os, sys
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VAL = "out/yolo_val_full/images/val"
GT = {}
for p in sorted(glob.glob(f"{VAL}/*.jpg")):
    s = os.path.basename(p)[:-4]
    lp = p.replace("images", "labels").replace(".jpg", ".txt")
    b = []
    for ln in open(lp):
        _, cx, cy, w, h = map(float, ln.split())
        b.append([(cx - w / 2) * 1920, (cy - h / 2) * 344,
                  (cx + w / 2) * 1920, (cy + h / 2) * 344])
    GT[s] = np.array(b)


def iou(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    it = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return it / (ar(a)[:, None] + ar(b)[None, :] - it + 1e-9)


W = os.environ.get("W", "runs/detect/out/runs/fullB/weights/best.pt")
m = YOLO(W)
print(f"{os.path.basename(os.path.dirname(os.path.dirname(W)))}  "
      f"{len(GT)} val frames, {sum(len(v) for v in GT.values())} boxes\n")
print(f"{'imgsz':>12} {'conf':>5} | {'TP':>5} {'FP':>5} {'FN':>5} | "
      f"{'prec':>6} {'recall':>7} {'F1':>6}")
for imgsz in (640, 960, 1280, (384, 1920)):
    ps = {}
    for p in sorted(glob.glob(f"{VAL}/*.jpg")):
        r = m.predict(p, imgsz=imgsz, conf=0.05, iou=0.6, device="mps", verbose=False)[0]
        ps[os.path.basename(p)[:-4]] = (r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.conf.cpu().numpy())
    best = None
    for conf in np.arange(0.10, 0.66, 0.05):
        TP = FP = FN = 0
        for s, g in GT.items():
            d = ps[s][0][ps[s][1] >= conf]
            if len(g) == 0:
                FP += len(d); continue
            if len(d) == 0:
                FN += len(g); continue
            M = iou(d, g); used = set()
            for i in np.argsort(-ps[s][1][ps[s][1] >= conf]):
                j = int(np.argmax(M[i]))
                if M[i, j] >= 0.5 and j not in used:
                    used.add(j); TP += 1
                else:
                    FP += 1
            FN += len(g) - len(used)
        pr = TP / max(TP + FP, 1); rc = TP / max(TP + FN, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        if best is None or f1 > best[-1]:
            best = (conf, TP, FP, FN, pr, rc, f1)
    c, TP, FP, FN, pr, rc, f1 = best
    print(f"{str(imgsz):>12} {c:5.2f} | {TP:5d} {FP:5d} {FN:5d} | "
          f"{pr:6.3f} {rc:7.3f} {f1:6.3f}")
