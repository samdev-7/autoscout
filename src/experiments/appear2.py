"""Spatially-structured appearance descriptor.

The existing descriptor is one global colour histogram over the whole box, which
discards layout entirely: a robot that is dark on the left and bright on the
right is indistinguishable from its mirror image.  Humans identify objects by
parts and their arrangement, so give the descriptor the same information -- split
the box into a grid and describe each cell separately.
"""
import av, cv2, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import BAND0, BAND1

RC, CC = 2, 3        # cell rows, cell cols
HB, SB = 8, 2
CELL = HB * SB + 3


def desc(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    out = np.zeros(RC * CC * CELL, np.float32)
    if x1 - x0 < RC * 2 or y1 - y0 < CC * 2:
        return out
    c = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    H, W = c.shape[:2]
    for r in range(RC):
        for k in range(CC):
            b = c[r * H // RC:(r + 1) * H // RC, k * W // CC:(k + 1) * W // CC]
            if b.size == 0:
                continue
            hh, ss, vv = b[..., 0], b[..., 1], b[..., 2]
            m = (vv > 35) & (vv < 250)
            if m.sum() < 6:
                continue
            a = np.histogram2d(hh[m], ss[m], bins=[HB, SB],
                               range=[[0, 180], [0, 256]])[0].ravel()
            v = np.histogram(vv[m], bins=3, range=(0, 256))[0]
            z = np.r_[a, v].astype(np.float32)
            out[(r * CC + k) * CELL:(r * CC + k + 1) * CELL] = z / max(z.sum(), 1)
    return out / max(out.sum(), 1e-9)


if __name__ == "__main__":
    d = np.load(os.environ.get("DETS", "out/dets_fullB.npz"),
                allow_pickle=True)["dets"]
    fr = d[:, 0].astype(int); order = np.argsort(fr, kind="stable")
    b = np.searchsorted(fr[order], np.arange(fr.max() + 2))
    G = np.zeros((len(d), RC * CC * CELL), np.float32)
    c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
    n = 0
    for f in c.decode(s):
        if f.time is None or f.time < 8.0:
            continue
        if f.time > 172.0:
            break
        if n <= fr.max():
            im = cv2.cvtColor(f.to_ndarray(format="rgb24")[BAND0:BAND1],
                              cv2.COLOR_RGB2BGR)
            for k in order[b[n]:b[n + 1]]:
                G[k] = desc(im, *d[k, 3:7])
        n += 1
    c.close()
    np.savez_compressed(os.environ.get("APP2", "out/appear_grid.npz"), grid=G)
    print(f"{RC}x{CC} grid descriptors ({G.shape[1]} dims) for {len(d)} detections")
