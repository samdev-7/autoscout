"""End-to-end: detections -> six smoothed robot trajectories.

The one non-obvious step is duplicate merging.  Two tracklets of the same
alliance that overlap in time and sit persistently closer than a robot
footprint (0.91 m) cannot be two robots -- it is one robot detected twice.
Left in, each duplicate forces an extra chain in the path cover, which is why
the cover floors at 7 chains per alliance instead of 3.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, track as T, link as L

DUP_SEP, DUP_OVL = 0.91, 5


class Merged:
    """A tracklet-like object: sorted frames, one averaged position each."""
    __slots__ = ("fr", "xs", "al")
    def __init__(self, parts):
        acc = {}
        for p in parts:
            for f, x, a in zip(p.fr, p.xs, p.al):
                acc.setdefault(f, []).append((np.asarray(x), a))
        self.fr = sorted(acc)
        self.xs = [np.mean([v[0] for v in acc[f]], 0) for f in self.fr]
        self.al = [max(set(z), key=z.count) if (z := [v[1] for v in acc[f] if v[1]]) else 0
                   for f in self.fr]


def majority_alliance(t):
    v = [a for a in t.al if a]
    return max(set(v), key=v.count) if v else 0


def merge_duplicates(tl, al):
    par = list(range(len(tl)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    pos = [dict(zip(t.fr, [np.asarray(p) for p in t.xs])) for t in tl]
    nmerge = 0
    for a in (1, 2):
        idx = np.where(al == a)[0]
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j = idx[ii], idx[jj]
                com = set(pos[i]) & set(pos[j])
                if len(com) < DUP_OVL:
                    continue
                dd = np.median([np.linalg.norm(pos[i][f] - pos[j][f]) for f in com])
                if dd < DUP_SEP:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        par[ri] = rj; nmerge += 1
    groups = {}
    for i in range(len(tl)):
        groups.setdefault(find(i), []).append(i)
    out, oal = [], []
    for g in groups.values():
        out.append(Merged([tl[i] for i in g]))
        oal.append(majority_alliance(out[-1]))
    return out, np.array(oal), nmerge


def run(dets="out/dets_fullB.npz", xyp="out/dets_xy.npy", tag="fullB"):
    d = np.load(dets, allow_pickle=True)["dets"]
    xy = np.load(xyp)
    T.MISS_MAX, T.GATE = 20, 13.8
    tl = T.build_tracklets(d, xy, geom.cov(xy, su=2.0, sv=2.5))
    al = np.array([majority_alliance(t) for t in tl])
    print(f"tracklets {len(tl)}  (red {(al==1).sum()} blue {(al==2).sum()} "
          f"unk {(al==0).sum()})")

    tl, al, nm = merge_duplicates(tl, al)
    print(f"merged {nm} duplicate pairs -> {len(tl)} tracklets "
          f"(red {(al==1).sum()} blue {(al==2).sum()} unk {(al==0).sum()})")

    nf = int(d[:, 0].max()) + 1
    tracks = []
    for a, name in ((1, "red"), (2, "blue")):
        idx = np.where(al == a)[0]
        sub = [tl[i] for i in idx]
        occ = np.zeros(nf, int)
        for t in sub:
            occ[t.fr[0]:t.fr[-1] + 1] += 1
        floor = len(L.cover(sub, lam=0.0, c0=1e6))
        print(f"\n{name}: {len(sub)} tracklets, max concurrent {occ.max()}, "
              f">3 in {100*(occ>3).mean():.1f}% of frames, cover floor {floor}")
        ch = None
        for c0 in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            c = L.cover(sub, lam=0.3, c0=c0)
            if len(c) <= 3:
                ch = c; break
            ch = c
        print(f"  -> {len(ch)} chains")
        order = sorted(ch, key=lambda c: -(sub[c[-1]].fr[-1] - sub[c[0]].fr[0]))
        for c in order[:3]:
            f0, xs, Ps, m = L.smooth(sub, c, nf)
            tracks.append(dict(alliance=name, f0=int(f0), n=int(len(xs)),
                               xy=xs[:, :2], v=xs[:, 2:], P=Ps, obs=m,
                               ntl=len(c)))
            sp = np.hypot(*xs[:, 2:].T)
            print(f"   track {len(tracks)-1}: frames {f0}..{f0+len(xs)-1} "
                  f"({len(xs)*T.DT:.1f}s) from {len(c):3d} tracklets, "
                  f"observed {100*m.mean():.0f}%, speed p50 {np.median(sp):.2f} "
                  f"p99 {np.percentile(sp,99):.2f} max {sp.max():.2f} m/s")
    np.save(f"out/tracks_{tag}.npy", np.array(tracks, dtype=object), allow_pickle=True)
    print(f"\nsaved out/tracks_{tag}.npy  ({len(tracks)} tracks)")
    return tracks


if __name__ == "__main__":
    run()
