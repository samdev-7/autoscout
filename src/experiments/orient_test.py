"""Decide the field-frame orientation using the held-out AprilTags.

Reprojection error CANNOT settle this: a homography absorbs any affine relabel
of the source plane, so a flipped frame fits the marked points equally well.
The tags were never used in the fit, so they arbitrate.

Hypotheses (only these two are physically realisable by moving a camera):
  A  (X, Y)                     as marked
  B  (LEN-X, WID-Y)             camera on the opposite side = 180 deg about vertical
"""
import json, numpy as np, cv2

LEN, WID = 16.541, 8.069
IN = 0.0254
pts = json.load(open("out/points_user.json"))
img = np.array([[p["u"], p["v"]] for p in pts], np.float64)
fld = np.array([[p["X"], p["Y"]] for p in pts], np.float64)

# tags: observed image position, true field pose (from WPILib CSV)
TAGS = [(2, 616.8, 210.9, 469.111*IN, 182.6*IN, 44.25*IN),
        (11, 583.3, 211.2, 483.111*IN, 182.6*IN, 44.25*IN),
        (21, 1303.8, 204.6, 182.111*IN, 182.6*IN, 44.25*IN)]

def fit(F, I):
    H, _ = cv2.findHomography(F, I, 0)
    proj = cv2.perspectiveTransform(F.reshape(-1,1,2), H).reshape(-1,2)
    return H, np.linalg.norm(proj-I, axis=1)

for name, F in (("A  as marked", fld),
                ("B  180deg (LEN-X, WID-Y)", np.column_stack([LEN-fld[:,0], WID-fld[:,1]]))):
    H, res = fit(F, img)
    print(f"\n=== hypothesis {name} ===")
    print(f"  reprojection RMS {res.std():.2f} px   mean {res.mean():.2f}   max {res.max():.2f}")
    du = []
    print(f"  {'tag':>4} {'u_obs':>8} {'u_carpet_pred':>14} {'du':>8} {'v_obs':>7} {'v_pred':>8} {'dv':>7}")
    for tid, uo, vo, X, Y, Z in TAGS:
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
        du.append(abs(p[0]-uo))
        print(f"  {tid:4d} {uo:8.1f} {p[0]:14.1f} {p[0]-uo:8.1f} {vo:7.1f} {p[1]:8.1f} {p[1]-vo:7.1f}")
    print(f"  --> mean |du| = {np.mean(du):8.1f} px    "
          f"(tag is {Z/IN:.1f}in above carpet: v_pred should be BELOW v_obs, i.e. dv>0)")
