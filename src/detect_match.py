"""Run the detector + ByteTrack baseline across the whole match.

Decodes once with VideoToolbox, crops the wide panel's field band, and runs
inference at exactly the settings validation used (imgsz=640 on the 1920x344
crop) so detection quality matches the measured 0.775 mAP50.  Saves raw
per-frame detections; projection and tracking happen downstream so this
expensive pass runs once.
"""
import av, cv2, numpy as np, time, sys, os
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import BAND0, BAND1
# PANEL selects which broadcast panel to run on: the wide field band, or one of
# the alliance-station panels (which are 960x540 crops of the composite).
PANEL = os.environ.get("PANEL", "wide")
CROP = {"wide": (BAND0, BAND1, 0, 1920),
        "red_station": (540, 1080, 0, 960),
        "blue_station": (540, 1080, 960, 1920)}[PANEL]

VIDEO = "data/match1_qual.mp4"
WEIGHTS = os.environ.get("W", "runs/detect/out/runs/fullB/weights/best.pt")
OUT = os.environ.get("O", "out/dets_fullB.npz")
T0, T1 = 8.0, 172.0
CONF = float(os.environ.get("CONF", "0.10"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))


def alliance(img, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = img[y0 + int(0.45 * (y1 - y0)):y1, x0:x1]
    if crop.size == 0:
        return 0
    h = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = h[..., 0].astype(int), h[..., 1].astype(int), h[..., 2].astype(int)
    sat = (S > 95) & (V > 45)
    red = (((H <= 9) | (H >= 168)) & sat).sum()
    blue = ((H >= 100) & (H <= 132) & sat).sum()
    if red + blue < 12:
        return 0
    return 1 if red > blue else 2          # 0 abstain, 1 red, 2 blue


m = YOLO(WEIGHTS)
c = av.open(VIDEO); s = c.streams.video[0]
s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"

rows = []          # frame_idx, t, tid, x0,y0,x1,y1, conf, alliance
t0 = time.time(); n = 0; kept = 0
for f in c.decode(s):
    if f.time is None or f.time < T0:
        continue
    if f.time > T1:
        break
    y0, y1, x0, x1 = CROP
    bgr = cv2.cvtColor(f.to_ndarray(format="rgb24")[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
    # predict, not track: ByteTrack re-filters detections with its own
    # thresholds (track_high_thresh 0.25), which silently overrode CONF and
    # starved the downstream tracker of the weak boxes it can actually use.
    r = m.predict(bgr, imgsz=IMGSZ, conf=CONF, iou=0.6, max_det=40,
                  device="mps", verbose=False)[0]
    b = r.boxes
    if b is not None and len(b):
        xy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy()
        tid = np.full(len(b), -1.0)
        for k in range(len(b)):
            rows.append([n, float(f.time), float(tid[k]), *xy[k],
                         float(cf[k]), alliance(bgr, xy[k])])
        kept += len(b)
    n += 1
    if n % 600 == 0:
        el = time.time() - t0
        print(f"  {n:5d} frames  t={f.time:6.1f}s  {kept:6d} dets  "
              f"{n/el:5.1f} fps  eta {(4920-n)/(n/el):5.0f}s", flush=True)
c.close()

a = np.array(rows, float)
np.savez_compressed(OUT, dets=a, band=[BAND0, BAND1], weights=WEIGHTS,
                    conf=CONF, imgsz=IMGSZ)
el = time.time() - t0
print(f"\n{n} frames, {len(a)} detections in {el:.0f}s ({n/el:.1f} fps)")
print(f"mean {len(a)/max(n,1):.2f} det/frame   distinct track ids "
      f"{len(np.unique(a[:,2][a[:,2]>0]))}")
print(f"saved {OUT}")
