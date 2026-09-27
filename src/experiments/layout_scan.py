"""Classify broadcast layout for every frame; report contiguous segments.

Decodes the whole match with VideoToolbox, downscaling in the decoder's
swscale pass so we never materialise 1080p arrays we don't need.
"""
import av, numpy as np, time

PATH, W, H = "data/match1_qual.mp4", 480, 270

c = av.open(PATH)
s = c.streams.video[0]
s.codec_context.options = {"hwaccel": "videotoolbox"}
s.thread_type = "AUTO"

feats, times = [], []
t0 = time.time()
for f in c.decode(s):
    a = f.to_ndarray(format="rgb24", width=W, height=H)
    g = a.mean(axis=2)
    feats.append((
        g[122:134, :].mean(),        # separator band between top/bottom panels
        g[230:266, 114:368].mean(),  # score-bug region (bright when in-match)
        a[136:264, 0:3, 0].mean(),   # left panel border: red channel
        a[136:264, 0:3, 2].mean(),   # left panel border: blue channel
        g.mean(),                    # overall brightness
    ))
    times.append(float(f.time))
n = len(feats)
dt = time.time() - t0
c.close()

F = np.array(feats)
print(f"decoded {n} frames in {dt:.1f}s  ->  {n/dt:.0f} fps  ({n/dt/30:.1f}x realtime)")
print(f"duration {times[-1]:.1f}s\n")

sep, bug = F[:, 0], F[:, 1]
lab = np.where((sep < 45) & (bug > 90), "3PANEL",
      np.where(bug > 90, "FULLWIDE", "GRAPHIC"))

# collapse to segments, ignoring runs shorter than 0.5s
segs, start = [], 0
for i in range(1, n + 1):
    if i == n or lab[i] != lab[start]:
        segs.append((times[start], times[i-1] if i < n else times[-1], lab[start], i - start))
        start = i
print(f"{'start':>7} {'end':>7} {'dur':>6}  layout")
for a, b, l, cnt in segs:
    if b - a >= 0.5:
        print(f"{a:7.1f} {b:7.1f} {b-a:6.1f}  {l}")

print()
for l in ("FULLWIDE", "3PANEL", "GRAPHIC"):
    tot = sum(b - a for a, b, ll, _ in segs if ll == l)
    print(f"  {l:9} {tot:6.1f}s  ({tot/times[-1]*100:4.1f}%)")
