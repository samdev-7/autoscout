"""Background plates and a reference frame for each of the three panels.

Two things matter here:
  * The panels do not show their match views until t=7.17 s -- before that the
    broadcast is a different composite entirely.  Anything earlier poisons the
    median and would put the wrong scene in the calibration tool.
  * The median runs over the WHOLE play window so that a person standing in one
    part of the match cannot dominate; the cameras are locked off, so every
    extra frame is more evidence for the same static scene.

Memory is bounded by taking the median in horizontal strips, and by using
np.partition (which keeps uint8) rather than np.median (which upcasts to float).
"""
import av, cv2, numpy as np, os, sys

PAN = {"wide": (0, 490, 0, 1920), "red_station": (540, 1080, 0, 960),
       "blue_station": (540, 1080, 960, 1920)}
T0 = float(os.environ.get("T0", "7.25"))     # first frame after the panel switch
T1 = float(os.environ.get("T1", "172"))
STEP = int(os.environ.get("STEP", "8"))
FRAME_T = float(os.environ.get("FRAME_T", "7.33"))
STRIP = 64

buf = {k: [] for k in PAN}
ref = None
c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
n = 0
for f in c.decode(s):
    if f.time is None or f.time < T0:
        continue
    if f.time > T1:
        break
    im = None
    if ref is None and f.time >= FRAME_T:
        im = cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
        ref = (float(f.time), im)
    if n % STEP == 0:
        if im is None:
            im = cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
        for k, (y0, y1, x0, x1) in PAN.items():
            buf[k].append(im[y0:y1, x0:x1].copy())
    n += 1
c.close()

for k, v in buf.items():
    h, w = v[0].shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    for y in range(0, h, STRIP):
        st = np.stack([a[y:y + STRIP] for a in v])          # (N, strip, w, 3) uint8
        mid = len(st) // 2
        out[y:y + STRIP] = np.partition(st, mid, axis=0)[mid]
    cv2.imwrite(f"out/plate_{k}.png", out)
    y0, y1, x0, x1 = PAN[k]
    cv2.imwrite(f"out/frame_{k}.png", ref[1][y0:y1, x0:x1])
    print(f"{k:14s} plate from {len(v)} frames spanning t={T0}..{T1}s   "
          f"frame at t={ref[0]:.2f}s")
