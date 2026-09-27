"""Field-space multi-target tracker for the six robots.

Two stages, because the two failure modes want opposite settings.

  1. TRACKLETS -- tight gates, break on any ambiguity.  Optimises purity: a
     tracklet should never contain two different robots, even at the cost of
     being short.
  2. LINKING -- chain tracklets into six tracks.  This is where recall comes
     back, and it is done offline so a gap is bridged using the motion on BOTH
     sides of it rather than a forward guess.

Association lives in field metres, not pixels.  A fixed pixel gate means very
different things at Y=0.6 and Y=7.4 because apparent size changes ~2x across
the field, whereas "a robot cannot move more than 5 m/s" is true everywhere.
"""
import numpy as np, sys, os
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

DT = 1 / 30.0
VMAX = 5.0            # m/s, hard physical ceiling for an FRC drivetrain
SIG_A = 7.0           # m/s^2 process noise; robots accelerate hard
GATE = 13.8           # chi^2, 2 dof, ~99.9%
MISS_MAX = 12         # frames a tracklet may coast before it is closed
MIN_LEN = 5           # discard tracklets shorter than this

F = np.eye(4); F[0, 2] = F[1, 3] = DT
H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1
q = SIG_A ** 2
Q = q * np.array([[DT**4/4, 0, DT**3/2, 0], [0, DT**4/4, 0, DT**3/2],
                  [DT**3/2, 0, DT**2, 0], [0, DT**3/2, 0, DT**2]])


class Tracklet:
    __slots__ = ("x", "P", "fr", "xs", "al", "miss", "id", "di")
    def __init__(self, z, R, fr, al, tid):
        self.x = np.r_[z, 0.0, 0.0]
        self.P = np.diag([R[0, 0], R[1, 1], VMAX**2, VMAX**2])
        self.P[:2, :2] = R
        self.fr = [fr]; self.xs = [z.copy()]; self.al = [al]
        self.di = []                    # source detection indices
        self.miss = 0; self.id = tid

    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        s = np.hypot(*self.x[2:])
        if s > VMAX:                       # clamp: no FRC robot exceeds this
            self.x[2:] *= VMAX / s

    def update(self, z, R, fr, al, di=None):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.fr.append(fr); self.xs.append(self.x[:2].copy()); self.al.append(al)
        if di is not None:
            self.di.append(di)
        self.miss = 0

    def gate(self, z, R):
        S = H @ self.P @ H.T + R
        d = z - H @ self.x
        return float(d @ np.linalg.solve(S, d))


def build_tracklets(det, xy, cov, su_frames=None):
    """det rows: frame,t,tid,x0,y0,x1,y1,conf,alliance."""
    fr_all = det[:, 0].astype(int)
    order = np.argsort(fr_all, kind="stable")
    det, xy, cov, fr_all = det[order], xy[order], cov[order], fr_all[order]
    bounds = np.searchsorted(fr_all, np.arange(fr_all.max() + 2))

    live, done, nid = [], [], 0
    for f in range(fr_all.max() + 1):
        s, e = bounds[f], bounds[f + 1]
        for t in live:
            t.predict()
        idx = np.arange(s, e)
        if len(live) and len(idx):
            C = np.full((len(live), len(idx)), 1e6)
            for i, t in enumerate(live):
                for j, k in enumerate(idx):
                    a = int(det[k, 8])
                    ta = [v for v in t.al if v]
                    # alliance veto: a confident disagreement forbids the match
                    if a and ta and a != max(set(ta), key=ta.count):
                        continue
                    d = t.gate(xy[k], cov[k])
                    if d < GATE:
                        C[i, j] = d
            ri, ci = linear_sum_assignment(C)
            used_t, used_d = set(), set()
            for i, j in zip(ri, ci):
                if C[i, j] < GATE:
                    live[i].update(xy[idx[j]], cov[idx[j]], f,
                                   int(det[idx[j], 8]), int(idx[j]))
                    used_t.add(i); used_d.add(j)
            for j in range(len(idx)):
                if j not in used_d:
                    t = Tracklet(xy[idx[j]], cov[idx[j]], f,
                                 int(det[idx[j], 8]), nid); t.di.append(int(idx[j]))
                    live.append(t); nid += 1
            for i, t in enumerate(live[:len(C)]):
                if i not in used_t:
                    t.miss += 1
        else:
            for t in live:
                t.miss += 1
            for k in idx:
                t = Tracklet(xy[k], cov[k], f, int(det[k, 8]), nid)
                t.di.append(int(k)); live.append(t); nid += 1
        keep = []
        for t in live:
            (keep if t.miss <= MISS_MAX else done).append(t)
        live = keep
    done += live
    return [t for t in done if len(t.fr) >= MIN_LEN]


if __name__ == "__main__":
    d = np.load("out/dets_fullB.npz", allow_pickle=True)["dets"]
    xy = np.load("out/dets_xy.npy")
    R = geom.cov(xy, su=2.0, sv=2.5)          # differential noise: tighter than absolute
    T = build_tracklets(d, xy, R)
    L = np.array([len(t.fr) for t in T])
    D = np.array([(t.fr[-1] - t.fr[0]) * DT for t in T])
    print(f"{len(T)} tracklets (ByteTrack baseline: 386)")
    print(f"  length   p50 {np.median(L):.0f}  p90 {np.percentile(L,90):.0f}  max {L.max()} frames")
    print(f"  duration p50 {np.median(D):.1f}s  p90 {np.percentile(D,90):.1f}s  max {D.max():.1f}s")
    print(f"  >5 s: {(D>5).sum()}   >20 s: {(D>20).sum()}   >60 s: {(D>60).sum()}")
    print(f"  detections covered: {L.sum()}/{len(d)} = {100*L.sum()/len(d):.0f}%")
    al = []
    for t in T:
        v = [x for x in t.al if x]
        al.append(max(set(v), key=v.count) if v else 0)
    al = np.array(al)
    print(f"  alliance: red {(al==1).sum()}  blue {(al==2).sum()}  unknown {(al==0).sum()}")
    pur = [np.mean([x == a for x in t.al if x]) for t, a in zip(T, al) if any(t.al)]
    print(f"  alliance purity within tracklet: p50 {np.median(pur):.3f}  p10 {np.percentile(pur,10):.3f}")
    np.save("out/tracklets.npy", np.array(
        [np.c_[np.array(t.fr), np.array(t.xs), np.full(len(t.fr), t.id),
               np.array(t.al)] for t in T], dtype=object), allow_pickle=True)
    print("saved out/tracklets.npy")
