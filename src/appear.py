"""Extract per-detection appearance descriptors and test if they separate robots.

A crossing is information-starved: two same-alliance robots at the same place
look identical to a position-only tracker.  Motion continuity is the only
tiebreak, and it fails precisely when robots stop and jostle -- which is when
they overlap.  So the question is whether appearance carries enough signal to
tell two robots of the SAME alliance apart.  Measure before building on it.

Two regions are tested separately, because the detector boxes the chassis
(~39 px tall) while the whole robot is ~69 px: the bumper region is dominated by
alliance colour and should be useless within an alliance, whereas the
superstructure above the box is where teams differ.
"""
import av, cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import BAND0, BAND1

HB, SB = 12, 3


def desc(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros(HB * SB + 4, np.float32)
    c = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    H, S, V = c[..., 0], c[..., 1], c[..., 2]
    m = (V > 35) & (V < 250)
    if m.sum() < 20:
        return np.zeros(HB * SB + 4, np.float32)
    hh = np.histogram2d(H[m], S[m], bins=[HB, SB],
                        range=[[0, 180], [0, 256]])[0].ravel()
    vv = np.histogram(V[m], bins=4, range=(0, 256))[0]
    d = np.r_[hh, vv].astype(np.float32)
    return d / max(d.sum(), 1)


if __name__ == "__main__":
    d = np.load(os.environ.get("DETS","out/dets_fullB.npz"), allow_pickle=True)["dets"]
    fr = d[:, 0].astype(int)
    order = np.argsort(fr, kind="stable")
    b = np.searchsorted(fr[order], np.arange(fr.max() + 2))
    body = np.zeros((len(d), HB * SB + 4), np.float32)
    chas = np.zeros_like(body)
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
                x0, y0, x1, y1 = d[k, 3:7]
                hgt = y1 - y0
                body[k] = desc(im, x0, y0 - 0.85 * hgt, x1, y0 + 0.15 * hgt)
                chas[k] = desc(im, x0, y0, x1, y1)
        n += 1
    c.close()
    np.savez_compressed(os.environ.get("APP", "out/appear.npz"), body=body, chas=chas)
    print(f"descriptors for {len(d)} detections over {n} frames")
