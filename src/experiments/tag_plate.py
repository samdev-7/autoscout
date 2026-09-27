"""Detect AprilTags on the noise-free median plate with aggressive upscaling.

The plate has no motion blur, no occluders and averaged-out compression noise,
so tags too marginal in single frames may resolve here. Breaking the collinear
degeneracy needs tags at differing field Y (e.g. 3/4/9/10 on the red structure).
"""
import numpy as np, cv2, csv

plate = np.load("out/top_plate.npy")
g = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)

layout = {}
with open("data/field/2026-rebuilt-welded.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        layout[int(r["ID"])] = (float(r["X"]), float(r["Y"]), float(r["Z"]))

p = cv2.aruco.DetectorParameters()
p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 43, 4
p.minMarkerPerimeterRate = 0.003
p.maxErroneousBitsInBorderRate = 0.4
p.polygonalApproxAccuracyRate = 0.06
p.perspectiveRemovePixelPerCell = 8
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)

found = {}
for up in (2, 3, 4, 6, 8):
    im = cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    for variant, img in (("plain", im), ("clahe", cv2.createCLAHE(3.0, (8,8)).apply(im))):
        cor, ids, _ = det.detectMarkers(img)
        if ids is None: continue
        for i, cid in enumerate(ids.flatten()):
            c = cor[i][0] / up
            found.setdefault(int(cid), []).append((up, variant, c))

print(f"detected {len(found)} distinct tags on the plate\n")
print(f"{'id':>4} {'img_x':>8} {'img_y':>7} {'field_X':>9} {'field_Y':>8} {'Z':>6}  scales")
pts_i, pts_f = [], []
for cid in sorted(found):
    cs = np.stack([c for _,_,c in found[cid]])
    ctr = cs.mean(axis=1).mean(axis=0)
    fx, fy, fz = layout.get(cid, (float('nan'),)*3)
    scales = ",".join(sorted({f"{u}{v[0]}" for u,v,_ in found[cid]}))
    print(f"{cid:4d} {ctr[0]:8.1f} {ctr[1]:7.1f} {fx:9.3f} {fy:8.3f} {fz:6.2f}  {scales}")
    pts_i.append(ctr); pts_f.append((fx, fy))

pts_f = np.array(pts_f)
print(f"\nfield-Y spread of detected tags: {np.ptp(pts_f[:,1]):.1f} in   "
      f"field-X spread: {np.ptp(pts_f[:,0]):.1f} in")
print("distinct field-Y values:", sorted(set(np.round(pts_f[:,1],1))))
if np.ptp(pts_f[:,1]) < 5:
    print("=> still collinear: homography/PnP from tags remains DEGENERATE")
else:
    print("=> 2D spread present: tag-based calibration is viable")
