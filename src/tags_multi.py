"""Detect AprilTags in all three panels; save corners and a verification image.

Corners are the median over many frames -- the cameras are locked off, so this
drives detection noise well below a pixel and rejects one-frame false IDs.
"""
import av, cv2, csv, numpy as np, json

PAN = {"wide": (0, 490, 0, 1920), "red_station": (540, 1080, 0, 960),
       "blue_station": (540, 1080, 960, 1920)}
IN = 0.0254
T = {}
with open("data/field/2026-rebuilt-welded.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        T[int(r["ID"])] = r

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
pa = cv2.aruco.DetectorParameters()
pa.adaptiveThreshWinSizeMin = 3; pa.adaptiveThreshWinSizeMax = 45
pa.adaptiveThreshWinSizeStep = 4; pa.minMarkerPerimeterRate = 0.008
pa.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
det = cv2.aruco.ArucoDetector(d, pa)

acc = {k: {} for k in PAN}
c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
n = 0
for f in c.decode(s):
    if f.time is None or f.time < 8:
        continue
    if f.time > 170:
        break
    n += 1
    if n % 10:
        continue
    im = cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_BGR2GRAY) \
        if False else cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
    for k, (y0, y1, x0, x1) in PAN.items():
        g = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        co, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        for cc, i in zip(co, ids.ravel()):
            acc[k].setdefault(int(i), []).append(cc.reshape(4, 2))

out = {}
for k, a in acc.items():
    out[k] = {}
    for i, v in a.items():
        if len(v) < 5:
            continue                      # a handful of frames is a likely false ID
        out[k][str(i)] = dict(corners=np.median(v, 0).round(2).tolist(),
                              seen=len(v), known=(i in T))
json.dump(out, open("out/tags_multi.json", "w"), indent=1)

# verification image on the still pre-match frame
c.seek(0)
for f in c.decode(s):
    if f.time is not None and f.time >= 7.05:
        im = cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR); break
c.close()
tiles = []
for k, (y0, y1, x0, x1) in PAN.items():
    p = im[y0:y1, x0:x1].copy()
    for i, rec in sorted(out[k].items(), key=lambda z: -z[1]["seen"]):
        q = np.array(rec["corners"])
        col = (0, 255, 255) if rec["known"] else (0, 0, 255)
        cv2.polylines(p, [q.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        cv2.circle(p, tuple(q[0].astype(int)), 4, (0, 255, 0), -1)   # corner 0
        cen = q.mean(0)
        cv2.putText(p, f"{i}", (int(cen[0]) - 12, int(cen[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(p, f"{i}", (int(cen[0]) - 12, int(cen[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    cv2.putText(p, f"{k}  ({len(out[k])} tags)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if p.shape[1] != 1920:
        p = cv2.copyMakeBorder(p, 0, 0, 0, 1920 - p.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
    tiles.append(p)
cv2.imwrite("out/tags_check.png", np.vstack(tiles))
for k in PAN:
    ids = sorted(int(i) for i in out[k])
    print(f"{k:14s}: {ids}")
    for i in ids:
        r = out[k][str(i)]
        cen = np.mean(r["corners"], 0)
        print(f"    tag {i:3d} seen {r['seen']:4d}x  centre ({cen[0]:6.1f},{cen[1]:6.1f})"
              + ("" if r["known"] else "   <-- NOT IN FIELD CSV"))
print("\nwrote out/tags_multi.json and out/tags_check.png "
      "(green dot = corner 0, yellow = known tag)")
