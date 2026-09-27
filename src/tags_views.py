"""Detected AprilTags per panel, in the guide format calib.html already uses.

Detection runs on the median PLATE and upscaled: the plate has no motion blur
and nobody standing in front of it, and upscaling recovers tags whose printed
squares are only a few pixels across.  That takes each station view from 4 to 6
usable tags.
"""
import cv2, csv, json, numpy as np

PAN = ["wide", "red_station", "blue_station"]
IN = 0.0254
T = {}
with open("data/field/2026-rebuilt-welded.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        T[int(r["ID"])] = r


def detector(minper, win_max):
    pa = cv2.aruco.DetectorParameters()
    pa.adaptiveThreshWinSizeMin = 3; pa.adaptiveThreshWinSizeMax = win_max
    pa.adaptiveThreshWinSizeStep = 2; pa.minMarkerPerimeterRate = minper
    pa.maxMarkerPerimeterRate = 4.0; pa.polygonalApproxAccuracyRate = 0.05
    pa.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    pa.errorCorrectionRate = 0.8
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), pa)


out = {}
for p in PAN:
    im = cv2.imread(f"out/plate_{p}.png")
    g0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    found = {}
    for sc in (1, 2, 3):
        g = cv2.resize(g0, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc > 1 else g0
        for mp, wm in ((0.008, 45), (0.003, 71), (0.02, 31)):
            co, ids, _ = detector(mp, wm).detectMarkers(g)
            if ids is None:
                continue
            for cc, i in zip(co, ids.ravel()):
                i = int(i)
                if i not in T:               # reject IDs the field does not have
                    continue
                found.setdefault(i, cc.reshape(4, 2) / sc)
    guides = []
    for i, c in sorted(found.items()):
        cen = c.mean(0)
        guides.append(dict(id=i, gu=round(float(cen[0]), 2), gv=round(float(cen[1]), 2),
                           Z=round(float(T[i]["Z"]) * IN, 4), det=True,
                           corners=np.round(c, 2).tolist()))
    out[p] = guides
    print(f"{p:14s}: {len(guides)} tags  {[g['id'] for g in guides]}")
json.dump(out, open("out/tags_views.json", "w"), indent=1)
print("wrote out/tags_views.json")
