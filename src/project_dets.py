"""Raw detections -> field positions + appearance descriptors for the tracker.

    DETS=out/dets_wide.npz CONF=0.30 OUT=out/dets_wide_c30 python src/project_dets.py

Writes OUT.npz (filtered detections), OUT_xy.npy (footprint centres, metres) and
OUT_app.npz (HSV descriptors, via appear.py).  Filtering is: confidence >= CONF,
projects inside the field, and field-space NMS (two robots cannot be closer than
a footprint).  This is the wide-panel path; station panels are projected inside
fuse_track.py through mcam because they need a per-camera corner correction.
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

if __name__ == "__main__":
    DETS = os.environ.get("DETS", "out/dets_wide.npz")
    CONF = float(os.environ.get("CONF", "0.30"))
    OUT = os.environ.get("OUT", os.path.splitext(DETS)[0] + f"_c{int(CONF*100):02d}")

    d = np.load(DETS, allow_pickle=True)["dets"]
    xy = geom.box_to_field(d[:, 3:7])
    keep = (d[:, 7] >= CONF) & np.isfinite(xy).all(1) & geom.in_field(xy)
    keep &= geom.field_nms(d[:, 0], xy, d[:, 7])
    print(f"{len(d)} detections -> {keep.sum()} after conf>={CONF}, in-field, field NMS")
    np.savez(OUT + ".npz", dets=d[keep]); np.save(OUT + "_xy.npy", xy[keep])
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "appear.py")],
                   env=dict(os.environ, DETS=OUT + ".npz", APP=OUT + "_app.npz"), check=True)
    print(f"wrote {OUT}.npz  {OUT}_xy.npy  {OUT}_app.npz")
