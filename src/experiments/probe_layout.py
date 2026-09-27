"""Probe broadcast layout: panel boundaries + whether the top camera is static."""
import av, numpy as np

PATH = "data/match1_qual.mp4"

def sample(times):
    """Decode one frame at each timestamp using VideoToolbox hw decode."""
    out = {}
    c = av.open(PATH)
    s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}
    for t in times:
        c.seek(int(t / s.time_base), stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time >= t - 0.05:
                out[t] = f.to_ndarray(format="rgb24")
                break
    c.close()
    return out

times = list(range(4, 189, 6))
fr = sample(times)
print(f"decoded {len(fr)} frames, shape {next(iter(fr.values())).shape}\n")

# --- panel separators: rows that are near-black across the full width ---
stack = np.stack([f for f in fr.values()])          # (N,H,W,3)
rowmean = stack.mean(axis=(0, 2, 3))                # mean brightness per row
dark = np.where(rowmean < 30)[0]
runs, start = [], None
for i, y in enumerate(dark):
    if start is None: start = y
    elif y != dark[i-1] + 1:
        runs.append((start, dark[i-1])); start = y
if start is not None: runs.append((start, dark[-1]))
print("dark horizontal bands (panel separators):")
for a, b in runs:
    if b - a >= 2: print(f"  rows {a:4d}-{b:4d}  (h={b-a+1})")

# --- vertical split in the bottom half ---
bot = stack[:, 560:1080].mean(axis=(0, 1, 3))
cand = np.where(bot < 40)[0]
print(f"\nbottom-half dark columns near centre: {[c for c in cand if 900 < c < 1020]}")

# --- is the top panel camera static? ---
TOP = slice(0, 480)
g = [f[TOP].astype(np.float32).mean(axis=2) for f in fr.values()]
ref = g[0]
print("\ntop-panel camera stability (mean abs diff vs first frame):")
diffs = [np.abs(x - ref).mean() for x in g]
print(f"  min={min(diffs):.1f}  max={max(diffs):.1f}  mean={np.mean(diffs):.1f}")

# static-structure test: compare edge maps of upper region (field perimeter/truss)
import cv2
e = [cv2.Canny(x.astype(np.uint8), 80, 160) for x in g]
agree = np.mean([(e[0] == ei).mean() for ei in e[1:]])
print(f"  edge-map agreement vs frame 0: {agree*100:.1f}%  "
      f"({'STATIC camera' if agree > 0.90 else 'camera MOVES'})")
