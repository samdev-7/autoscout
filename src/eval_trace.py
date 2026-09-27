"""Evaluate tracks against hand-traced robot paths -- including ID switches.

Errors are reported split along the camera view direction and across it, because
the two are not comparable: lateral resolution is ~0.4 in/px while depth is
~1-2 in/px, and the tracer's own uncertainty is anisotropic in the opposite
sense -- judging a rotating robot's left/right middle is the hard part, but that
is the axis the geometry resolves best, whereas depth is read off a crisp
feature (bumper meeting carpet) on the axis that resolves worst.

The box-label evaluation cannot see identity: it Hungarian-matches per frame, so
two tracks that swap cleanly are simply re-paired and recall does not move.  A
trace follows ONE robot through time, which is exactly the information needed to
ask whether a track stayed with that robot.

A traced point is the centre of the robot's contact patch, so it is NOT put
through the footprint correction -- that correction exists to turn a detection
box's bottom edge (the corner nearest the camera) into a centre, and a human
clicking the centre has already done that job.
"""
import numpy as np, json, os, sys
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

MATCH = float(os.environ.get("MATCH", "1.0"))     # m, counts as "on the robot"
ALLI = {"r": "red", "b": "blue"}


def _norm(v):
    """Accept the tracer's three annotation modes plus the legacy bare [x,y]."""
    if v == "occ":
        return ("occ", None)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return ("floor", [float(v[0]), float(v[1])])
    if isinstance(v, dict):
        if v.get("m") == "box":
            return ("box", [float(x) for x in v["b"]])
        return (v.get("m", "floor"), [float(x) for x in v["p"]])
    return (None, None)


def load_trace(path="out/trace.json"):
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    B = geom.BAND0

    # pass 1: boxes give this robot's height, which is what makes a TOP click
    # usable -- the top sits directly above the base, so intersecting the ray
    # with the plane at that height yields ground XY without seeing the floor.
    H = {}
    for rk, fr in raw.items():
        hs = []
        for v in fr.values():
            m, a = _norm(v)
            if m != "box":
                continue
            g = geom.box_to_field([[a[0], a[1], a[2], a[3]]])[0]
            if np.isfinite(g).all():
                hs.append(geom.height_of([(a[0] + a[2]) / 2, a[1] + B], g))
        H[rk] = float(np.median(hs)) if len(hs) >= 3 else None

    out = {}
    for rk, fr in raw.items():
        pts, occ, nt, warn = {}, set(), 0, False
        for f, v in fr.items():
            f = int(f); m, a = _norm(v)
            if m == "occ":
                occ.add(f); continue
            if m is None:
                continue
            if m == "box":
                g = geom.box_to_field([[a[0], a[1], a[2], a[3]]])[0]
            elif m == "floor":
                # the tracer marks (horizontal centre, nearest bumper-ground
                # point) -- the same convention as a detection box, so it needs
                # the same footprint correction, not a raw ground intersection.
                g = geom.point_to_field([[a[0], a[1] + B]])[0]
            else:                                   # top of robot
                h = H.get(rk)
                if h is None:
                    h = 0.8; warn = True
                g = geom.img_to_plane([[a[0], a[1] + B]], h)[0]
                nt += 1
            if np.isfinite(g).all():
                pts[f] = g
        if pts or occ:
            out[rk] = dict(pts=pts, occ=occ, alliance=ALLI[rk[0]],
                           height=H.get(rk), ntop=nt, warn=warn)
    return out


def main(trk=os.environ.get("TRK", "out/tracks_em.npy")):
    tr = list(np.load(trk, allow_pickle=True))
    T = load_trace()
    if not T:
        print("no trace data yet"); return
    nf = len(tr[0]["xy"])
    print(f"tracks: {trk}")
    for rk, d in sorted(T.items()):
        h = d.get("height")
        print(f"  traced {rk} ({d['alliance']}): {len(d['pts'])} points, "
              f"{len(d['occ'])} occluded, {d.get('ntop',0)} top-clicks"
              + (f", height {h:.2f} m from boxes" if h else "")
              + ("  [WARNING: top clicks used an assumed 0.80 m height -- "
                 "draw >=3 boxes for this robot to measure it]" if d.get("warn") else ""))

    frames = sorted({f for d in T.values() for f in d["pts"]})
    keys = sorted(T)
    assign = {k: {} for k in keys}
    for f in frames:
        act = [k for k in keys if f in T[k]["pts"]]
        cand = [i for i in range(len(tr))]
        C = np.full((len(act), len(cand)), 1e6)
        for a, k in enumerate(act):
            for b, i in enumerate(cand):
                if tr[i]["alliance"] != T[k]["alliance"]:
                    continue
                C[a, b] = np.linalg.norm(T[k]["pts"][f] - tr[i]["xy"][f])
        ri, ci = linear_sum_assignment(C)
        for a, b in zip(ri, ci):
            if C[a, b] < MATCH:
                assign[act[a]][f] = (cand[b], C[a, b])

    print(f"\n{'robot':>7} | {'pts':>5} {'matched':>8} {'err p50':>8} {'err p90':>8} "
          f"{'depth':>7} {'lat':>6} | {'ID sw':>6} {'purity':>7}")
    tot_sw = tot_pts = 0
    for k in keys:
        pts = sorted(T[k]["pts"])
        seq = [(f, assign[k][f][0]) for f in pts if f in assign[k]]
        err = np.array([assign[k][f][1] for f in pts if f in assign[k]])
        dep, lat = [], []
        for f in pts:
            if f not in assign[k]:
                continue
            p0 = T[k]["pts"][f]
            d = tr[assign[k][f][0]]["xy"][f] - p0
            u = geom._away([p0])[0]                     # unit vector away from camera
            dep.append(abs(float(d @ u)))
            lat.append(abs(float(d[0] * -u[1] + d[1] * u[0])))
        sw = sum(1 for a, b in zip(seq, seq[1:]) if a[1] != b[1])
        if seq:
            ids = [s[1] for s in seq]
            pur = max(ids.count(i) for i in set(ids)) / len(ids)
        else:
            pur = 0.0
        tot_sw += sw; tot_pts += len(pts)
        print(f"{k:>7} | {len(pts):5d} {100*len(seq)/max(len(pts),1):7.1f}% "
              f"{np.median(err) if len(err) else float('nan'):8.2f} "
              f"{np.percentile(err,90) if len(err) else float('nan'):8.2f} "
              f"{np.median(dep) if dep else float('nan'):7.2f} "
              f"{np.median(lat) if lat else float('nan'):6.2f} | "
              f"{sw:6d} {100*pur:6.1f}%")
    print(f"\n{tot_sw} ID switches over {tot_pts} traced points "
          f"({tot_sw/max(tot_pts/150,1):.2f} per 5 s of traced robot time)")
    occ_all = sum(len(d["occ"]) for d in T.values())
    if occ_all:
        cov = 0
        for k in keys:
            for f in T[k]["occ"]:
                i = [x for x in range(len(tr)) if tr[x]["alliance"] == T[k]["alliance"]]
                cov += any(tr[x]["obs"][f] for x in i)
        print(f"occluded frames: {occ_all}  (tracker still asserts a position for "
              f"all of them -- that is interpolation, judge it by the ellipse)")


if __name__ == "__main__":
    main()
