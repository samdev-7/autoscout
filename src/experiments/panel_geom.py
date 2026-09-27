"""Find exact panel rectangles at full res, and test whether each camera is static."""
import av, numpy as np, cv2

PATH = "data/match1_qual.mp4"

def grab(times):
    out = []
    c = av.open(PATH); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}
    for t in times:
        c.seek(int(t / s.time_base), stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time >= t - 0.05:
                out.append(f.to_ndarray(format="rgb24")); break
    c.close(); return out

# --- exact geometry from a mid-match 3-panel frame ---
ref = grab([90.0])[0]
g = ref.mean(axis=2)
rows = g.mean(axis=1)
dark_rows = [y for y in range(400, 620) if rows[y] < 30]
print("dark separator rows in 400-620:", f"{min(dark_rows)}-{max(dark_rows)}" if dark_rows else "none")

# panel border colours: red band on left panel, blue on right
mid = 800
r_row = ref[mid, :, 0].astype(int); b_row = ref[mid, :, 2].astype(int)
redcols  = [x for x in range(0, 1920) if r_row[x] > 110 and b_row[x] < 70]
bluecols = [x for x in range(0, 1920) if b_row[x] > 110 and r_row[x] < 90]
print("red border cols  :", redcols[:4], "...", redcols[-4:] if redcols else "")
print("blue border cols :", bluecols[:4], "...", bluecols[-4:] if bluecols else "")

TOP = (0, 0, 1920, min(dark_rows) if dark_rows else 490)
print(f"\nTOP panel  = x0,y0,x1,y1 {TOP}  -> {TOP[2]-TOP[0]}x{TOP[3]-TOP[1]}")

# --- static-camera test across the whole 3PANEL play window ---
times = np.arange(10, 171, 4.0)
frames = grab(list(times))
print(f"\nstability across {len(frames)} frames, t=10..170s")

regions = {
    "TOP  (wide field)": (0,   0,  1920, 470),
    "BL   (red end)"   : (5,   545, 955, 1060),
    "BR   (blue end)"  : (965, 545, 1915, 1060),
}
for name, (x0, y0, x1, y1) in regions.items():
    crops = [cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY) for f in frames]
    base = crops[0]
    shifts, ncc = [], []
    for cr in crops[1:]:
        # phase correlation gives sub-pixel global translation
        d, _ = cv2.phaseCorrelate(np.float32(base), np.float32(cr))
        shifts.append(np.hypot(*d))
        ncc.append(cv2.matchTemplate(cr, base, cv2.TM_CCOEFF_NORMED)[0, 0])
    print(f"  {name}: shift px  med={np.median(shifts):5.2f}  p95={np.percentile(shifts,95):5.2f}"
          f"  max={max(shifts):5.2f}   NCC med={np.median(ncc):.3f}")
