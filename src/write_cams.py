"""Assemble out/cams.json (every camera in one field frame) from the calibrations.

    python src/write_cams.py                      # all station views + wide
    LENS_FROM=red_station_b:red_station ...       # reuse a lens for a bumped camera

Station views are solved by solve_api.solve from the points/lines/pairs saved by
calib.html, lines.html and pair.html.  The wide entry is taken from geom.py
(out/cam_params.npy) so that mcam and geom agree exactly.  mcam.py reads the
result; the tracker and every renderer read mcam.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, solve_api

if __name__ == "__main__":
    VIEWS = os.environ.get("VIEWS", "red_station,red_station_b,blue_station").split(",")
    LENS = dict(p.split(":") for p in os.environ.get("LENS_FROM", "red_station_b:red_station").split(",") if p)

    cams = {}
    if os.path.exists("out/cams.json"):
        cams = json.load(open("out/cams.json"))
    for v in VIEWS:
        r = solve_api.solve(v, lens_from=LENS.get(v))
        if "error" in r:
            print(f"{v}: NOT solved -- {r['error']}"); continue
        cams[v] = r
        print(f"{v}: f {r['f']:.1f}  median {r['median']:.2f} px  carpet {r['carpet']:.2f} px  "
              f"({r['n_constraints']} constraints, lens from {r['lens_source']})")
        cams[v].update(W=solve_api.W, H=solve_api.H)
    # wide: out/wide_cam.json is the fit calib.html saved; fall back to geom.py's cam_params
    if os.path.exists("out/wide_cam.json"):
        w = json.load(open("out/wide_cam.json")); R = np.array(w["R"]); C = np.array(w["C"])
        wide = dict(f=w["f"], cx=w["cx"], cy=w["cy"], k1=w.get("k1", 0.0), k2=w.get("k2", 0.0),
                    R=R.tolist(), C=C.tolist())
    else:
        R, C = geom.R, geom.CAM
        wide = dict(f=float(geom.K[0, 0]), cx=float(geom.K[0, 2]), cy=float(geom.K[1, 2]),
                    k1=float(geom.DIST[0]), k2=0.0, R=R.tolist(), C=C.tolist())
    wide.update(t=(-R @ C).tolist(), W=1920, H=490,
                median=cams.get("wide", {}).get("median", 1.41),
                carpet=cams.get("wide", {}).get("carpet", 1.41))
    cams["wide"] = wide
    json.dump(cams, open("out/cams.json", "w"), indent=1)
    print("wrote out/cams.json:", ", ".join(cams))
