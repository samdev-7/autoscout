"""Extract crossing events and crops so identity can be labelled as yes/no.

A crossing is a binary question -- did these two tracks swap? -- so labelling it
does not need continuous tracing.  For each event we cut one crop per track from
the clean stretch before, and one from the clean stretch after.  The judgement is
then "which after-crop is the same robot as this before-crop", which is a couple
of seconds of work and produces exactly the ground truth the repair step needs.
"""
import numpy as np, cv2, av, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, repair
from geom import BAND0, BAND1

PAD, CW, CH = 8, 150, 120


def main():
    tr = list(np.load("out/tracks_em.npy", allow_pickle=True))
    d = np.load("out/dets_fullB.npz", allow_pickle=True)["dets"]
    nf = len(tr[0]["xy"])
    amap = [dict((f, i) for f, i in t["aidx"]) for t in tr]

    ev = []
    for a in (0, 3):
        for i in range(a, a + 3):
            for j in range(i + 1, a + 3):
                for w0, w1 in repair.windows(tr[i]["xy"], tr[j]["xy"], nf):
                    pick = {}
                    ok = True
                    for k in (i, j):
                        bf = [f for f in amap[k] if w0 - repair.WIN <= f < w0]
                        af = [f for f in amap[k] if w1 < f <= w1 + repair.WIN]
                        if len(bf) < 6 or len(af) < 6:
                            ok = False; break
                        pick[(k, "b")] = int(np.median(bf))
                        pick[(k, "a")] = int(np.median(af))
                    if ok:
                        ev.append(dict(i=i, j=j, w0=int(w0), w1=int(w1),
                                       t0=round(8 + w0 / 30, 2),
                                       t1=round(8 + w1 / 30, 2), pick=pick))
    print(f"{len(ev)} labellable crossing events")

    need = {}
    for n, e in enumerate(ev):
        for (k, s), f in e["pick"].items():
            need.setdefault(f, []).append((n, k, s))
    os.makedirs("out/cross", exist_ok=True)
    for f in os.listdir("out/cross"):
        os.remove(f"out/cross/{f}")

    c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
    n = 0
    for fr in c.decode(s):
        if fr.time is None or fr.time < 8.0:
            continue
        if fr.time > 172.0:
            break
        if n in need:
            im = cv2.cvtColor(fr.to_ndarray(format="rgb24")[BAND0:BAND1],
                              cv2.COLOR_RGB2BGR)
            for (evn, k, side) in need[n]:
                di = amap[k][n]
                x0, y0, x1, y1 = d[di, 3:7]
                h = y1 - y0
                y0 -= 0.85 * h                       # include the superstructure
                x0, x1 = int(x0) - PAD, int(x1) + PAD
                y0, y1 = int(y0) - PAD, int(y1) + PAD
                x0, y0 = max(0, x0), max(0, y0)
                x1 = min(im.shape[1], x1); y1 = min(im.shape[0], y1)
                cr = im[y0:y1, x0:x1]
                if cr.size:
                    cv2.imwrite(f"out/cross/e{evn:03d}_{k}_{side}.jpg",
                                cv2.resize(cr, (CW, CH)),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    c.close()
    for e in ev:
        e["pick"] = {f"{k}_{s}": v for (k, s), v in e["pick"].items()}
    json.dump(ev, open("out/crossings.json", "w"))
    print(f"wrote out/crossings.json and {len(os.listdir('out/cross'))} crops")


if __name__ == "__main__":
    main()
