"""Identity assignment at TRACKLET level, not per frame.

Measured on hand-traced ground truth, the appearance descriptor scores EER ~35%
comparing one detection to a model, but 2-6% comparing two segments of >=1 s.
The EM used it in the first regime -- ~150k weak decisions per match.  Here each
tracklet (median 1.2 s) is one decision instead, so the same descriptor is used
where it actually works, and there are ~300 of them rather than 150,000.

Two further things the benchmark said, applied here:
  * keep a SET of exemplars per identity, never a mean.  Averaging blurs every
    viewpoint of a rotating robot together; nearest/percentile matching against
    exemplars cut EER by a third at short windows.
  * seed from tracklets that are alive SIMULTANEOUSLY.  Three concurrent
    tracklets must be three different robots -- that is free, certain labelling,
    and it removes the initialisation guesswork the EM never recovered from.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, track as T, fit6

VMAX = 5.0
MARGIN = float(os.environ.get("MARGIN", "0.3"))
NEX = int(os.environ.get("NEX", "24"))    # exemplars kept per tracklet
PCT = int(os.environ.get("PCT", "10"))    # percentile of pairwise distances
W_MOT = float(os.environ.get("W_MOT", "0.25"))  # motion vs appearance
DUP_M = float(os.environ.get("DUP_M", "0.7"))   # m, span-interpolated duplicate test


def bhat_sets(P, Q, pct=PCT):
    """Distance between two exemplar SETS -- percentile of all pair distances."""
    v = 1.0 - np.sqrt(P[:, None, :] * Q[None, :, :]).sum(-1)
    return float(np.percentile(v, pct))


class Tk:
    """A tracklet with its exemplar set."""
    __slots__ = ("fr", "xs", "al", "ex", "n")
    def __init__(self, fr, xs, al, ex):
        self.fr, self.xs, self.al = fr, xs, al
        self.ex, self.n = ex, len(fr)
    @property
    def f0(self): return self.fr[0]
    @property
    def f1(self): return self.fr[-1]


def build(dets, xy, app, miss=None, gate=None, dup=None):
    miss = int(os.environ.get("MISS", "20")) if miss is None else miss
    gate = float(os.environ.get("GATE", "13.8")) if gate is None else gate
    dup = float(os.environ.get("DUP", "6.0")) if dup is None else dup
    """Tracklets with exemplars, duplicates merged."""
    T.MISS_MAX, T.GATE = miss, gate
    fr = dets[:, 0].astype(int); order = np.argsort(fr, kind="stable")
    tl = T.build_tracklets(dets, xy, geom.cov(xy, su=2.0, sv=2.5))
    raw = []
    for t in tl:
        gi = order[np.array(t.di, int)] if len(t.di) else np.array([], int)
        raw.append((np.array(t.fr), np.array(t.xs), list(t.al), gi))
    # merge duplicates: same alliance, overlapping, Mahalanobis-close
    def maj(a):
        v = [x for x in a if x]
        return max(set(v), key=v.count) if v else 0
    al = [maj(r[2]) for r in raw]
    par = list(range(len(raw)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    pos = [dict(zip(r[0], r[1])) for r in raw]
    for a in (1, 2):
        idx = [i for i in range(len(raw)) if al[i] == a]
        for u in range(len(idx)):
            for v in range(u + 1, len(idx)):
                i, j = idx[u], idx[v]
                lo = max(raw[i][0][0], raw[j][0][0]); hi = min(raw[i][0][-1], raw[j][0][-1])
                if hi - lo + 1 < 3:
                    continue
                com = sorted(set(pos[i]) & set(pos[j]))
                if len(com) >= 5:
                    dd = []
                    for f in com:
                        w = pos[i][f] - pos[j][f]; m = (pos[i][f] + pos[j][f]) / 2
                        S = 2 * geom.cov(m[None], su=3.0, sv=4.0)[0]
                        dd.append(np.sqrt(w @ np.linalg.solve(S, w)))
                    same = np.median(dd) < dup
                else:
                    # Two tracklets on ONE robot rarely share an observed frame: the
                    # single detection per frame goes to one of them and the other
                    # coasts, so they alternate.  35 of 36 such duplicate pairs in the
                    # eval match had <5 common frames and slipped past the test above,
                    # and the concurrency rule then forced each pair into different
                    # identities.  Compare positions interpolated over the span instead.
                    # Traced same-alliance robots were never closer than 0.99 m, so a
                    # 0.7 m footprint threshold separates cleanly (34 pairs <0.5 m,
                    # 3 in 0.5-1.0, 14 in 1.0-2.0, 355 beyond).
                    fs = np.arange(lo, hi + 1)
                    pi = np.c_[np.interp(fs, raw[i][0], raw[i][1][:, 0]), np.interp(fs, raw[i][0], raw[i][1][:, 1])]
                    pj = np.c_[np.interp(fs, raw[j][0], raw[j][1][:, 0]), np.interp(fs, raw[j][0], raw[j][1][:, 1])]
                    same = np.median(np.linalg.norm(pi - pj, axis=1)) < DUP_M
                if same:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        par[ri] = rj
    groups = {}
    for i in range(len(raw)):
        groups.setdefault(find(i), []).append(i)
    out, oal = [], []
    rng = np.random.default_rng(0)
    for g in groups.values():
        acc = {}
        gi = []
        for i in g:
            gi.append(raw[i][3])
            for f, p, a in zip(raw[i][0], raw[i][1], raw[i][2]):
                acc.setdefault(f, []).append((p, a))
        fr_ = sorted(acc)
        xs_ = [np.mean([v[0] for v in acc[f]], 0) for f in fr_]
        al_ = [maj([v[1] for v in acc[f]]) for f in fr_]
        gi = np.concatenate(gi) if gi else np.array([], int)
        ex = app[gi] if len(gi) else np.zeros((0, app.shape[1]))
        if len(ex) > NEX:
            ex = ex[rng.choice(len(ex), NEX, replace=False)]
        out.append(Tk(np.array(fr_), np.array(xs_), al_, ex))
        oal.append(maj(al_))
    return out, np.array(oal)


OVL_TOL = 0        # frames of overlap tolerated before two tracklets are
                   # ruled to be different robots


def overlaps(a, b, tol=None):
    """Do these tracklets coexist enough to be certainly different robots?

    A strict "share any frame" test is too harsh: a residual duplicate or a
    boundary frame creates a 1-2 frame overlap, which would bar an otherwise
    perfectly good tracklet from its identity entirely.  That alone left 27% of
    blue tracklets unassigned.
    """
    tol = OVL_TOL if tol is None else tol
    return min(a.f1, b.f1) - max(a.f0, b.f0) + 1 > tol


def motion_cost(a, b):
    """How implausible is it that one robot produced both tracklets?"""
    if overlaps(a, b):
        return np.inf
    if a.f1 < b.f0:
        p, q, gap = a.xs[-1], b.xs[0], b.f0 - a.f1
    else:
        p, q, gap = b.xs[-1], a.xs[0], a.f0 - b.f1
    reach = VMAX * gap * T.DT + MARGIN
    d = np.linalg.norm(q - p)
    return np.inf if d > reach else (d / reach) ** 2


def assign(tks, K=3, iters=6):
    """Greedy seed from concurrent tracklets, then refine."""
    n = len(tks)
    if n <= K:
        return list(range(n)) + [-1] * 0
    order = np.argsort([-t.n for t in tks])
    # seed: the K longest tracklets that are pairwise concurrent (=> distinct robots)
    seed = []
    for i in order:
        if len(tks[i].ex) == 0:
            continue
        if all(overlaps(tks[i], tks[j]) for j in seed):
            seed.append(i)
        if len(seed) == K:
            break
    while len(seed) < K:                       # fall back to longest disjoint
        for i in order:
            if i not in seed:
                seed.append(i); break
    lab = np.full(n, -1)
    for c, i in enumerate(seed):
        lab[i] = c
    for it in range(iters):
        for i in order:
            if len(tks[i].ex) == 0:
                continue
            # never strand a cluster: a sole member stays put, otherwise a
            # cluster can empty out and we silently end up with <K robots
            if lab[i] >= 0 and sum(lab == lab[i]) == 1:
                continue
            best, bc = None, np.inf
            for c in range(K):
                mem = [j for j in range(n) if lab[j] == c and j != i]
                if not mem:
                    continue
                if any(overlaps(tks[i], tks[j]) for j in mem):
                    continue
                prev = [j for j in mem if tks[j].f1 < tks[i].f0]
                nxt = [j for j in mem if tks[j].f0 > tks[i].f1]
                nbs = ([max(prev, key=lambda j: tks[j].f1)] if prev else []) + \
                      ([min(nxt, key=lambda j: tks[j].f0)] if nxt else [])
                mcs = [motion_cost(tks[i], tks[j]) for j in nbs]
                if not mcs or not all(np.isfinite(m) for m in mcs):
                    continue
                mc = max(mcs)
                ex = np.concatenate([tks[j].ex for j in mem])
                if len(ex) > 200:
                    ex = ex[::max(1, len(ex) // 200)]
                cost = bhat_sets(tks[i].ex, ex) + W_MOT * mc
                if cost < bc:
                    bc, best = cost, c
            lab[i] = best if best is not None else -1
    return lab
