"""Multi-camera tracker: pure tracklets from every view -> fused -> exact identity.

Why this shape (see docs/COMPONENTS.md for the measurements behind each line):
  * Tracklets from `track.py` are 100% pure, so identity is decided ONLY at the
    tracklet level.  Every camera contributes tracklets to one pool in field
    metres; the tracker never needs to know which camera a tracklet came from.
  * Station tracklets are NEVER merged into wide tracklets.  Their ellipses are
    large (depth 1-sigma up to 3 m at Z=0.8 m), so one can genuinely agree with
    two different wide tracklets, and a wrong merge makes an impure tracklet --
    the one error the tracklet stage must never make (v1 of this file did
    exactly that: a 310-frame tracklet spanning r1 and r2, and one spanning b2
    and r2).  Instead a station tracklet is LINK EVIDENCE: if it agrees with
    wide tracklet A at A's end and with B at B's start, and was observed all the
    way across the gap, it BRIDGES A->B.  Bridged links are cheap; unbridged
    links pay a per-second gap penalty.  A station mistake then costs one
    link's weight, never a tracklet's purity.  After the solve, the bridging
    station measurements fill the gap for position (information-weighted).
  * Identity is an exact minimum-cost cover by K=3 node-disjoint time-ordered
    paths per alliance (min-cost flow).  Concurrency exclusion is implicit in the
    time ordering, and "the other two are alive so this must be the third"
    (elimination) falls out of the global optimum instead of a greedy pass.
  * Link cost is motion + physical height (+ colour where both sides are wide
    tracklets); within-alliance colour alone was AUC 0.72-0.82, height 0.75-0.92.
  * Fusion's purpose is to SHORTEN GAPS: a station tracklet through a wide-view
    occlusion turns one uninformative 2 s gap into short gaps where the reach
    term is decisive.
"""
import os, sys, numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, mcam, track as T, fit6, assign as AS

E = os.environ.get
# default = the winning detection set: epoch60 @ conf>=0.30 + field NMS (see docs/COMPONENTS.md)
WIDE_DETS, WIDE_XY, WIDE_APP = (E("DETS", "out/dets_e60c30.npz"), E("XY", "out/dets_e60c30_xy.npy"),
                                E("APP", "out/dets_e60c30_app.npz"))
RED, BLUE = E("RED", "out/dets_red_station.npz"), E("BLUE", "out/dets_blue_station.npz")
OUT = E("OUT", "out/tracks_fused.npy")
WIDE_ONLY = E("WIDE_ONLY", "0") == "1"
CONF_ST = float(E("CONF_ST", "0.30"))
MISS = int(E("MISS", "20"))
DUP_M = float(E("DUP_M", "0.7"))       # same-view alternating duplicates, metres
MAH_X = float(E("MAH_X", "3.0"))       # cross-view same-robot test, median Mahalanobis
W_M, W_H, W_C, W_R = (float(E("W_M", "2.0")), float(E("W_H", "1.0")),
                      float(E("W_C", "0.5")), float(E("W_R", "0.10")))
SIG_H = float(E("SIG_H", "0.10"))      # m, height difference that costs 1 unit at W_H=1
MAXGAP_S = float(E("MAXGAP_S", "10"))
XOVL = int(E("XOVL", "8"))             # min frames a station tracklet must overlap a wide one to vouch for it
W_G = float(E("W_G", "0.5"))           # cost per second of gap that NO camera explains
BRG_GAP = int(E("BRG_GAP", "20"))      # max unobserved run (frames) inside a bridging station tracklet
USE_ST = E("WIDE_ONLY", "0") != "1"
MARGIN = 0.3
NF = 4921
K = 3


class Tk:
    __slots__ = ("fr", "xs", "cv", "al", "views", "gi", "h", "hs", "ex", "n")
    def __init__(self, fr, xs, cv, al, views, hs, ex):
        self.fr, self.xs, self.cv, self.al = fr, xs, cv, al
        self.views, self.hs, self.ex, self.n = views, hs, ex, len(fr)
        self.h = float(np.median(hs)) if len(hs) >= 3 else np.nan
    @property
    def f0(self): return int(self.fr[0])
    @property
    def f1(self): return int(self.fr[-1])
    def at(self, fs):
        """Position and covariance interpolated at frames fs (span-internal)."""
        p = np.c_[np.interp(fs, self.fr, self.xs[:, 0]), np.interp(fs, self.fr, self.xs[:, 1])]
        c = np.stack([np.interp(fs, self.fr, self.cv[:, i, j]) for i in range(2) for j in range(2)], 1).reshape(-1, 2, 2)
        return p, c


def heights(cam, boxes, XY, band=0):
    """Vectorised robot height from box top given ground position (NaN if top clipped)."""
    b = np.atleast_2d(boxes); v = b[:, 1] + band
    lo, hi = np.full(len(b), 0.05), np.full(len(b), 2.5)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        pv = cam.project(np.c_[XY, mid])[:, 1]
        up = pv > v                                  # projected top still below the observed top
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    h = 0.5 * (lo + hi)
    h[b[:, 1] <= 2] = np.nan                         # box touches the top of the panel: clipped
    return h


def view_tracklets(name, det, xy, cov, hs, ex=None):
    """Pure Kalman tracklets for one view, carrying per-frame covariance and heights."""
    T.MISS_MAX = MISS
    order = np.argsort(det[:, 0].astype(int), kind="stable")
    tl = T.build_tracklets(det, xy, cov)
    out = []
    for t in tl:
        gi = order[np.array(t.di, int)]
        al = [a for a in t.al if a]; al = max(set(al), key=al.count) if al else 0
        fr = np.array(t.fr); xs = np.array(t.xs)
        cv = cov[gi] if len(gi) == len(fr) else mcam.cam(name).cov(xs)
        h = hs[gi]; h = h[np.isfinite(h)]
        e = ex[gi] if ex is not None else np.zeros((0, 1))
        out.append(Tk(fr, xs, cv, al, {name}, h, e))
    return out


def load_views():
    tks = []
    # ---- wide: detections already filtered/NMS'd; positions from geom ----
    d = np.load(WIDE_DETS, allow_pickle=True)["dets"]; xy = np.load(WIDE_XY)
    A = np.load(WIDE_APP); app = A["chas"]; app = app / np.maximum(app.sum(1, keepdims=True), 1e-9)
    w = mcam.cam("wide")
    hs = heights(w, d[:, 3:7], xy, band=geom.BAND0)
    tks += view_tracklets("wide", d, xy, geom.cov(xy, su=2.0, sv=2.5), hs, app)
    n_w = len(tks)
    if not USE_ST:
        return tks, n_w
    # ---- stations: split red at the camera bump; corner-correct per camera ----
    for path, views in ((RED, (("red_station", lambda f: f < mcam.RED_CUT0),
                               ("red_station_b", lambda f: f >= mcam.RED_CUT1))),
                        (BLUE, (("blue_station", lambda f: np.ones_like(f, bool)),))):
        d = np.load(path, allow_pickle=True)["dets"]; fr = d[:, 0].astype(int)
        for name, sel in views:
            c = mcam.cam(name); m = sel(fr) & (d[:, 7] >= CONF_ST)
            dd = d[m]; xy = c.box_to_field(dd[:, 3:7])
            ok = np.isfinite(xy).all(1) & geom.in_field(xy)
            dd, xy = dd[ok], xy[ok]
            keep = geom.field_nms(dd[:, 0], xy, dd[:, 7])
            dd, xy = dd[keep], xy[keep]
            cov = c.cov(xy); good = np.isfinite(cov).all((1, 2))
            dd, xy, cov = dd[good], xy[good], cov[good]
            hs = heights(c, dd[:, 3:7], xy)
            vt = view_tracklets(name, dd, xy, cov, hs)
            print(f"  {name:14s}: {len(dd):5d} dets -> {len(vt):3d} tracklets")
            tks += vt
    return tks, n_w


def merge_same_view(tks):
    """Union-find merge of alternating duplicates WITHIN one camera (0.7 m over the
    span overlap; same-robot tracklets rarely share an observed frame)."""
    n = len(tks); par = list(range(n))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tks[i], tks[j]
            if not (a.views & b.views): continue
            if a.al and b.al and a.al != b.al: continue
            lo, hi = max(a.f0, b.f0), min(a.f1, b.f1)
            if hi - lo + 1 < 3: continue
            fs = np.arange(lo, hi + 1)
            if np.median(np.linalg.norm(a.at(fs)[0] - b.at(fs)[0], axis=1)) < DUP_M:
                ri, rj = find(i), find(j)
                if ri != rj: par[ri] = rj
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(tks[g[0]]); continue
        acc = {}
        for i in g:
            for f, p, c in zip(tks[i].fr, tks[i].xs, tks[i].cv):
                acc.setdefault(int(f), []).append((p, c))
        fr = np.array(sorted(acc)); xs = np.zeros((len(fr), 2)); cv = np.zeros((len(fr), 2, 2))
        for k, f in enumerate(fr):
            I = sum(np.linalg.inv(c) for _, c in acc[f]); b = sum(np.linalg.inv(c) @ p for p, c in acc[f])
            cv[k] = np.linalg.inv(I); xs[k] = cv[k] @ b
        al = [tks[i].al for i in g if tks[i].al]; al = max(set(al), key=al.count) if al else 0
        hs = np.concatenate([tks[i].hs for i in g])
        exs = [tks[i].ex for i in g if len(tks[i].ex)]
        ex = np.concatenate(exs) if exs else np.zeros((0, 1))
        if len(ex) > AS.NEX: ex = ex[np.random.default_rng(0).choice(len(ex), AS.NEX, replace=False)]
        out.append(Tk(fr, xs, cv, al, tks[g[0]].views, hs, ex))
    return out


def agrees(w, s):
    """Does station tracklet s vouch for wide tracklet w?  Needs >= XOVL frames of
    overlap and agreement over nearly all of it under the summed covariances."""
    lo, hi = max(w.f0, s.f0), min(w.f1, s.f1)
    if hi - lo + 1 < XOVL: return False
    fs = np.arange(lo, hi + 1)
    pw, cw = w.at(fs); ps, cs = s.at(fs); d = pw - ps
    md = np.sqrt(np.einsum("ij,ijk,ik->i", d, np.linalg.inv(cw + cs), d))
    return np.percentile(md, 75) < MAH_X


def bridges(wide, st):
    """{(i, j): (station tracklet, uncovered seconds)} for wide pairs i -> j that a
    single station tracklet carries across the gap.  A station tracklet that
    vouches for two wide tracklets that are ALIVE AT THE SAME TIME is ambiguous
    and is dropped entirely -- that is the failure that made v1 impure."""
    B = {}
    n_amb = 0
    for s in st:
        vouch = [i for i, w in enumerate(wide) if agrees(w, s)]
        if len(vouch) < 2: continue
        amb = any(min(wide[a].f1, wide[b].f1) - max(wide[a].f0, wide[b].f0) + 1 >= 3
                  for x, a in enumerate(vouch) for b in vouch[x + 1:])
        if amb:
            n_amb += 1; continue
        vouch.sort(key=lambda i: wide[i].f0)
        for a, b in zip(vouch, vouch[1:]):
            A, Bw = wide[a], wide[b]
            if Bw.f0 <= A.f1: continue
            inside = s.fr[(s.fr > A.f1) & (s.fr < Bw.f0)]
            runs = np.diff(np.r_[A.f1, inside, Bw.f0]) - 1
            if runs.max() > BRG_GAP: continue           # the station lost it too
            unc = runs.sum() * T.DT
            if (a, b) not in B or unc < B[(a, b)][1]: B[(a, b)] = (s, unc)
    return B, n_amb


def link_cost(a, b, br=None):
    """a strictly before b.  inf if physically unreachable.  A bridge replaces the
    gap penalty by the part of the gap the station did not observe either."""
    gap = (b.f0 - a.f1) * T.DT
    if gap <= 0 or gap > MAXGAP_S: return np.inf
    d = np.linalg.norm(b.xs[0] - a.xs[-1]); reach = T.VMAX * gap + MARGIN
    if d > reach: return np.inf
    c = W_M * (d / reach) ** 2
    c += W_G * (br[1] if br is not None else gap)
    if np.isfinite(a.h) and np.isfinite(b.h):
        c += W_H * min(((a.h - b.h) / SIG_H) ** 2, 9.0)
    if len(a.ex) and len(b.ex) and a.ex.shape[1] == b.ex.shape[1] and a.ex.shape[1] > 1:
        c += W_C * max(AS.bhat_sets(a.ex, b.ex) - 0.03, 0.0) / 0.02
    return c


def solve(tks, k=K, br=None):
    """Exact min-cost cover by k node-disjoint time-ordered paths (min-cost flow)."""
    n = len(tks); G = nx.DiGraph(); SC = 1000
    G.add_node("s", demand=-k); G.add_node("t", demand=k)
    for i in range(n):
        G.add_edge("s", ("i", i), capacity=1, weight=0)
        G.add_edge(("i", i), ("o", i), capacity=1, weight=-int(round(SC * W_R * tks[i].n)))
        G.add_edge(("o", i), "t", capacity=1, weight=0)
    for i in range(n):
        for j in range(n):
            if i == j or tks[j].f0 <= tks[i].f1: continue
            c = link_cost(tks[i], tks[j], (br or {}).get((i, j)))
            if np.isfinite(c): G.add_edge(("o", i), ("i", j), capacity=1, weight=int(round(SC * c)))
    flow = nx.min_cost_flow(G)
    lab = np.full(n, -1); paths = []
    for i in range(n):
        if flow["s"].get(("i", i), 0) != 1: continue
        p, cur = [], i
        while True:
            p.append(cur); lab[cur] = len(paths)
            nxt = [v for v, f in flow[("o", cur)].items() if f == 1 and v != "t"]
            if not nxt: break
            cur = nxt[0][1]
        paths.append(p)
    return lab, paths


def gap_stats(tks, lab):
    """Unobserved run lengths inside each identity (the thing fusion should shorten)."""
    gaps = []
    for c in range(lab.max() + 1):
        fr = np.unique(np.concatenate([tks[i].fr for i in np.where(lab == c)[0]]))
        g = np.diff(fr) - 1; gaps += list(g[g > 0] * T.DT)
    gaps = np.array(gaps) if gaps else np.zeros(1)
    return f"gaps>0: n={len(gaps)} p50 {np.median(gaps):.2f}s p90 {np.percentile(gaps,90):.2f}s max {gaps.max():.2f}s  >1s: {(gaps>1).sum()}"


def main():
    tks, n_w = load_views()
    wide = merge_same_view([t for t in tks if "wide" in t.views])
    st = merge_same_view([t for t in tks if "wide" not in t.views]) if USE_ST else []
    print(f"{n_w} wide tracklets -> {len(wide)} after duplicate merge;  {len(st)} station tracklets")
    tracks = []
    for a, name in ((1, "red"), (2, "blue")):
        sub = [t for t in wide if t.al == a]
        br, n_amb = bridges(sub, st) if st else ({}, 0)
        lab, paths = solve(sub, br=br)
        used = [(i, j) for p in paths for i, j in zip(p, p[1:]) if (i, j) in br]
        print(f"\n{name}: {len(sub)} wide tracklets, {len(br)} bridged pairs ({n_amb} ambiguous station tracklets dropped), "
              f"{len(used)} bridges used -> {[len(p) for p in paths]} per identity, {(lab<0).sum()} unassigned")
        print(f"  wide-only gaps: {gap_stats(sub, lab)}")
        for c, p in enumerate(paths):
            obs = {}
            for i in p:
                for f, z, R in zip(sub[i].fr, sub[i].xs, sub[i].cv): obs[int(f)] = (z, R)
            nfill = 0
            for i, j in zip(p, p[1:]):                    # station measurements fill bridged gaps
                if (i, j) in br:
                    s_ = br[(i, j)][0]
                    for f, z, R in zip(s_.fr, s_.xs, s_.cv):
                        if sub[i].f1 < f < sub[j].f0 and int(f) not in obs: obs[int(f)] = (z, R); nfill += 1
            f0 = min(obs); xs, P = fit6.rts(obs, NF, np.r_[obs[f0][0], 0, 0])
            xs[:, 0] = np.clip(xs[:, 0], 0, geom.LEN); xs[:, 1] = np.clip(xs[:, 1], 0, geom.WID)
            m = np.zeros(NF, bool); m[list(obs)] = True
            hs = [sub[i].h for i in p if np.isfinite(sub[i].h)]
            fr = np.array(sorted(obs)); g = (np.diff(fr) - 1) * T.DT; g = g[g > 0]
            print(f"  id{c}: {len(p):3d} tracklets, observed {100*m.mean():5.1f}% ({nfill} frames from stations), "
                  f"height p50 {np.median(hs) if hs else float('nan'):.2f} m, gaps>1s {(g>1).sum()} max {g.max() if len(g) else 0:.2f}s")
            tracks.append(dict(alliance=name, xy=xs[:, :2], v=xs[:, 2:], obs=m, P=P, aidx=[]))
    np.save(OUT, np.array(tracks, dtype=object), allow_pickle=True)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
