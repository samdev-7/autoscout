"""Rank detectors by FIELD POSITION error, not IoU.

mAP scores box overlap, which weights the top edge as heavily as the bottom.
This pipeline uses only the bottom edge (and the horizontal centre): the top of
a robot is ambiguous and contributes nothing to a field position.  So a detector
can lose mAP for sloppy top edges while being perfect for our purpose, or win it
while placing robots badly.

Here predictions and labels are both projected to the carpet and matched by
metric distance, so the number means "how far off is the robot, in metres".
"""
import numpy as np, glob, os, sys
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

IMG = "out/yolo_val_full/images/val"
LAB = "out/yolo_val_full/labels/val"
W, H = 1920, 344


def boxes_to_xy(b):
    return geom.box_to_field(np.atleast_2d(b))


def gt_for(stem):
    p = f"{LAB}/{stem}.txt"
    out = []
    if not os.path.exists(p):
        return np.zeros((0, 4))
    for l in open(p):
        _, x, y, w, h = [float(v) for v in l.split()]
        out.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    return np.array(out) if out else np.zeros((0, 4))


def evaluate(weights, imgsz=960, confs=(0.25, 0.40, 0.55), dev="mps"):
    m = YOLO(weights)
    files = sorted(glob.glob(f"{IMG}/*.jpg"))
    cache = {}
    for f in files:
        r = m.predict(f, imgsz=imgsz, conf=0.05, iou=0.6, device=dev,
                      max_det=40, verbose=False)[0]
        b = r.boxes
        cache[f] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy()) if b is not None and len(b) \
            else (np.zeros((0, 4)), np.zeros(0))
    rows = []
    for conf in confs:
        TP = FP = FN = 0; errs = []
        for f in files:
            stem = os.path.basename(f)[:-4]
            g = gt_for(stem)
            xy_g = boxes_to_xy(g) if len(g) else np.zeros((0, 2))
            pb, pc = cache[f]
            keep = pc >= conf
            pb = pb[keep]
            xy_p = boxes_to_xy(pb) if len(pb) else np.zeros((0, 2))
            ok_g = np.isfinite(xy_g).all(1) if len(xy_g) else np.zeros(0, bool)
            ok_p = np.isfinite(xy_p).all(1) if len(xy_p) else np.zeros(0, bool)
            xy_g, xy_p = xy_g[ok_g], xy_p[ok_p]
            if len(xy_g) and len(xy_p):
                D = np.linalg.norm(xy_g[:, None] - xy_p[None], axis=2)
                ri, ci = linear_sum_assignment(D)
                hit = D[ri, ci] < 1.0                 # matched within 1 m on the carpet
                TP += hit.sum(); FN += len(xy_g) - hit.sum(); FP += len(xy_p) - hit.sum()
                errs += list(D[ri, ci][hit])
            else:
                FN += len(xy_g); FP += len(xy_p)
        e = np.array(errs) if errs else np.array([np.nan])
        rows.append(dict(conf=conf, recall=TP / max(TP + FN, 1),
                         precision=TP / max(TP + FP, 1),
                         p50=np.nanmedian(e), p90=np.nanpercentile(e, 90),
                         n=int(TP)))
    return rows


if __name__ == "__main__":
    cands = sys.argv[1:] or ["runs/detect/out/runs/fullB/weights/best.pt"]
    print(f"{'model':<34}{'conf':>6}{'recall':>8}{'prec':>7}{'err p50':>9}{'err p90':>9}")
    for w in cands:
        for r in evaluate(w):
            print(f"{os.path.basename(os.path.dirname(os.path.dirname(w)))+'/'+os.path.basename(w):<34}"
                  f"{r['conf']:6.2f}{100*r['recall']:7.1f}%{100*r['precision']:6.1f}%"
                  f"{r['p50']:9.3f}{r['p90']:9.3f}")
