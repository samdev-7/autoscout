"""Chain tracklets into six tracks, then smooth.

Linking is a MINIMUM-COST PATH COVER on a DAG.  Tracklets are ordered in time
and a link may only go forward, so any matching is automatically acyclic and
therefore decomposes into chains -- which means the whole thing reduces to one
bipartite assignment (Hungarian) rather than a flow solver.

Costs are physical: a link is feasible only if a robot could actually cover the
distance in the gap without exceeding VMAX, and is cheap only if the implied
accelerations at both ends are within what a drivetrain can do.
"""
import numpy as np, sys, os
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, track as T

VMAX, AMAX = 5.0, 8.0
BIG = 1e7


def ends(t):
    """Endpoint state of a tracklet: start/end frame, position, velocity."""
    f = np.array(t.fr); p = np.array(t.xs)
    k = min(5, len(f) - 1)
    v0 = (p[k] - p[0]) / max((f[k] - f[0]) * T.DT, 1e-6) if k else np.zeros(2)
    v1 = (p[-1] - p[-1 - k]) / max((f[-1] - f[-1 - k]) * T.DT, 1e-6) if k else np.zeros(2)
    return f[0], f[-1], p[0], p[-1], v0, v1


def link_cost(a, b, lam):
    _, e_i, _, p_i, _, v_i = a
    s_j, _, p_j, _, v_j, _ = b
    gap = s_j - e_i
    if gap < 1:
        return BIG
    dt = gap * T.DT
    d = p_j - p_i
    vr = d / dt
    if np.linalg.norm(vr) > VMAX:
        return BIG                      # physically unreachable
    acc = (np.linalg.norm(vr - v_i) + np.linalg.norm(v_j - vr)) / dt
    return (np.linalg.norm(vr) / VMAX) ** 2 + (acc / AMAX) ** 2 + lam * dt


def cover(tl, lam, c0):
    """Min-cost path cover -> list of chains (lists of tracklet indices)."""
    n = len(tl)
    E = [ends(t) for t in tl]
    C = np.full((n, 2 * n), BIG)
    for i in range(n):
        C[i, n + i] = c0                       # "end the chain here"
        for j in range(n):
            if i != j:
                C[i, j] = link_cost(E[i], E[j], lam)
    r, c = linear_sum_assignment(C)
    nxt = {i: (j if j < n and C[i, j] < BIG else None) for i, j in zip(r, c)}
    prv = {j: i for i, j in nxt.items() if j is not None}
    chains = []
    for i in range(n):
        if i in prv:
            continue
        ch, k = [], i
        while k is not None:
            ch.append(k); k = nxt.get(k)
        chains.append(ch)
    return chains


def smooth(tl, chain, nf):
    """RTS smoother over a chain -> per-frame state + observed mask."""
    obs = {}
    for k in chain:
        for f, p in zip(tl[k].fr, tl[k].xs):
            obs.setdefault(f, []).append(p)
    obs = {f: np.mean(v, 0) for f, v in obs.items()}
    f0, f1 = min(obs), max(obs)
    N = f1 - f0 + 1
    xf = np.zeros((N, 4)); Pf = np.zeros((N, 4, 4))
    xp = np.zeros((N, 4)); Pp = np.zeros((N, 4, 4))
    x = np.r_[obs[f0], 0, 0]; P = np.diag([.05, .05, VMAX**2, VMAX**2])
    for i in range(N):
        f = f0 + i
        if i:
            x = T.F @ x; P = T.F @ P @ T.F.T + T.Q
        xp[i], Pp[i] = x, P
        if f in obs:
            R = geom.cov(obs[f][None], su=2.0, sv=2.5)[0]
            S = T.H @ P @ T.H.T + R
            K = P @ T.H.T @ np.linalg.inv(S)
            x = x + K @ (obs[f] - T.H @ x)
            P = (np.eye(4) - K @ T.H) @ P
        xf[i], Pf[i] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for i in range(N - 2, -1, -1):
        G = Pf[i] @ T.F.T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + G @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + G @ (Ps[i + 1] - Pp[i + 1]) @ G.T
    m = np.array([(f0 + i) in obs for i in range(N)])
    return f0, xs, Ps, m


if __name__ == "__main__":
    d = np.load("out/dets_fullB.npz", allow_pickle=True)["dets"]
    xy = np.load("out/dets_xy.npy")
    T.MISS_MAX, T.GATE = 20, 13.8
    tl = T.build_tracklets(d, xy, geom.cov(xy, su=2.0, sv=2.5))
    al = []
    for t in tl:
        v = [a for a in t.al if a]
        al.append(max(set(v), key=v.count) if v else 0)
    al = np.array(al)
    nf = int(d[:, 0].max()) + 1
    print(f"{len(tl)} tracklets  (red {(al==1).sum()}, blue {(al==2).sum()}, "
          f"unknown {(al==0).sum()})\n")
    print(f"{'c0':>6} | {'red chains':>10} {'blue chains':>11}")
    for c0 in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        row = []
        for a in (1, 2):
            idx = np.where((al == a) | (al == 0))[0]
            row.append(len(cover([tl[i] for i in idx], lam=0.3, c0=c0)))
        print(f"{c0:6.1f} | {row[0]:10d} {row[1]:11d}")
