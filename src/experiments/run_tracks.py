"""Driver: detections -> tracklets -> chain init -> EM fit -> six trajectories."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, track as T, link as L, pipeline as P, fit6

A = np.load(os.environ.get("APP","out/appear.npz"))
# chassis only.  Measured against TRUE identity labels from hand traces, the
# box region scores AUC 0.979 / rank-1 99.8% over a 1.5 s window, while the
# region above the box scores 0.864 and the concatenation 0.853 -- the "body"
# crop mostly captures background, which tracked identity only because
# background correlates with field position.  Concatenating dilutes the signal.
FEAT = os.environ.get("FEAT", "chas")
APP = A["chas"] if FEAT == "chas" else (
      A["body"] if FEAT == "body" else np.concatenate([A["body"], A["chas"]], 1))
APP = APP / np.maximum(APP.sum(1, keepdims=True), 1e-9)

def chain_init(sub, chains, nf, k=3):
    """Choose k chains that are well SEPARATED, not merely well supported.

    Picking the top-k by support routinely selects two chains that follow the
    same robot, and EM cannot escape that: both slots stay on one robot and a
    third robot is never tracked.  Two robots cannot be within a footprint of
    each other, so score candidate triples by how often they collide.
    """
    from itertools import combinations
    sup = np.array([sum(len(sub[i].fr) for i in c) for c in chains], float)
    P = []
    for c in chains:
        f0, xs, _, _ = L.smooth(sub, c, nf)
        full = np.zeros((nf, 2))
        full[f0:f0 + len(xs)] = xs[:, :2]
        full[:f0] = xs[0, :2]; full[f0 + len(xs):] = xs[-1, :2]
        P.append(full)
    P = np.array(P)
    best, bs = None, -1e18
    for tri in combinations(range(len(chains)), k):
        col = 0.0
        for i, j in combinations(tri, 2):
            col = max(col, np.mean(np.linalg.norm(P[i] - P[j], axis=1) < 0.91))
        sc = sup[list(tri)].sum() / sup.sum() - 4.0 * col
        if sc > bs:
            bs, best = sc, tri
    print(f"    init chains {best}  support {sup[list(best)].astype(int)}  "
          f"worst-pair collision {max(np.mean(np.linalg.norm(P[i]-P[j],axis=1)<0.91) for i,j in combinations(best,2)):.1%}")
    return P[list(best)]


def _unused(sub, chains, nf, k=3):
    sup = [sum(len(sub[i].fr) for i in c) for c in chains]
    pick = np.argsort(sup)[::-1][:k]
    out = []
    for c in [chains[i] for i in pick]:
        f0, xs, _, _ = L.smooth(sub, c, nf)
        full = np.zeros((nf, 2))
        full[f0:f0 + len(xs)] = xs[:, :2]
        full[:f0] = xs[0, :2]; full[f0 + len(xs):] = xs[-1, :2]
        out.append(full)
    return np.array(out)

def main():
    d = np.load(os.environ.get("DETS","out/dets_fullB.npz"), allow_pickle=True)["dets"]
    xy = np.load(os.environ.get("XY","out/dets_xy.npy"))
    nf = int(d[:, 0].max()) + 1
    T.MISS_MAX, T.GATE = 20, 13.8
    Rass = geom.cov(xy, su=2.0, sv=2.5)
    tl = T.build_tracklets(d, xy, Rass)
    al = np.array([P.majority_alliance(t) for t in tl])
    tl, al, nm = P.merge_duplicates(tl, al)
    print(f"{len(tl)} tracklets after merging {nm} duplicate pairs")

    det_al = d[:, 8].astype(int)
    tracks = []
    for a, name in ((1, "red"), (2, "blue")):
        sub = [tl[i] for i in np.where(al == a)[0]]
        ch = None
        for c0 in (2, 4, 8, 16, 32, 64, 128):
            ch = L.cover(sub, lam=0.3, c0=c0)
            if len(ch) <= 4:
                break
        init = chain_init(sub, ch, nf)
        m = (det_al == a) | (det_al == 0)
        print(f"\n{name}: init from {len(ch)} chains, {m.sum()} candidate detections")
        tr, asg, Pc, aidx = fit6.fit_alliance(d[m, 0].astype(int), xy[m],
                                        geom.cov(xy[m], su=3.0, sv=4.0), nf, init,
                                        app=APP, ids=np.where(m)[0])
        for k in range(3):
            # a robot cannot be off the carpet; clamp before reporting
            tr[k][:, 0] = np.clip(tr[k][:, 0], 0.0, geom.LEN)
            tr[k][:, 1] = np.clip(tr[k][:, 1], 0.0, geom.WID)
            sp = np.hypot(tr[k][:, 2], tr[k][:, 3])
            obs = np.zeros(nf, bool); obs[list(asg[k])] = True
            tracks.append(dict(alliance=name, xy=tr[k][:, :2], v=tr[k][:, 2:],
                               obs=obs, P=Pc[k], aidx=aidx[k]))
            print(f"  track {len(tracks)-1}: observed {100*obs.mean():5.1f}%  "
                  f"speed p50 {np.median(sp):.2f} p99 {np.percentile(sp,99):.2f} "
                  f"max {sp.max():.2f} m/s   out-of-field "
                  f"{100*(1-geom.in_field(tr[k][:,:2],0.3).mean()):.1f}%")
    np.save(os.environ.get("OUT","out/tracks_em.npy"), np.array(tracks, dtype=object), allow_pickle=True)
    tot = sum(t["obs"].sum() for t in tracks)
    print(f"\ntotal detections used {tot}/{len(d)} = {100*tot/len(d):.0f}%")
    print("saved out/tracks_em.npy")

if __name__ == "__main__":
    main()
