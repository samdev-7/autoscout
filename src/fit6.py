"""Fit exactly three trajectories per alliance by EM.

A path cover cannot go below the maximum number of concurrently-alive
tracklets, and occlusion truncation reliably produces two tracklets for one
robot (the bottom edge jumps a metre or two when a robot is half-hidden, the
association gate breaks, and a second tracklet opens before the first closes).
So instead of covering tracklets, fit the thing we actually know exists: three
continuous trajectories per alliance, present for the whole match.

  E step  per frame, Hungarian-assign this alliance's detections to the three
          slots; anything beyond the gate becomes an outlier and is simply not
          used.  One detection per slot per frame -- a robot is in one place.
  M step  re-fit each slot with an RTS smoother over its assigned detections,
          which fills gaps by interpolating between both sides.

Truncated boxes therefore stop corrupting a track: they fail the gate and drop
out, rather than dragging the estimate a metre downrange.
"""
import numpy as np, sys, os
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, track as T

GATE_EM = 25.0        # chi^2, 2 dof -- generous; outliers are truncated boxes
ITERS = 12
W_APP = float(os.environ.get("W_APP", "0"))    # appearance vs Mahalanobis.
# Zero by default and that is deliberate: an appearance model LEARNED from the
# current assignment reinforces whatever swap has already happened, so EM sits
# at a self-consistent wrong answer.  Appearance is applied after the fit, in
# repair.py, comparing segments OUTSIDE the ambiguous window.
SEP_OK = 2.0          # m; only learn appearance where a track is unambiguous
EXCL = 0.91           # m; two robots cannot occupy the same footprint


def rts(obs, nf, x0):
    """obs: {frame: (z, R)}.  Returns smoothed state (nf,4) and covariance."""
    xf = np.zeros((nf, 4)); Pf = np.zeros((nf, 4, 4))
    xp = np.zeros((nf, 4)); Pp = np.zeros((nf, 4, 4))
    x = x0.copy(); P = np.diag([1.0, 1.0, T.VMAX**2, T.VMAX**2])
    for f in range(nf):
        if f:
            x = T.F @ x; P = T.F @ P @ T.F.T + T.Q
        xp[f], Pp[f] = x, P
        if f in obs:
            z, R = obs[f]
            S = T.H @ P @ T.H.T + R
            K = P @ T.H.T @ np.linalg.inv(S)
            x = x + K @ (z - T.H @ x)
            P = (np.eye(4) - K @ T.H) @ P
        xf[f], Pf[f] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for f in range(nf - 2, -1, -1):
        G = Pf[f] @ T.F.T @ np.linalg.inv(Pp[f + 1])
        xs[f] = xf[f] + G @ (xs[f + 1] - xp[f + 1])
        Ps[f] = Pf[f] + G @ (Ps[f + 1] - Pp[f + 1]) @ G.T
    return xs, Ps


def _bhat(p, q):
    return 1.0 - np.sqrt(p * q).sum(-1)


def fit_alliance(fr, xy, R, nf, init, iters=ITERS, verbose=True, app=None, ids=None):
    """init: (K,nf,2) starting trajectories.  Returns states, assignment, cov.

    If `app` (per-detection appearance descriptors) is given, each track keeps an
    appearance model learned ONLY from frames where it sits more than SEP_OK from
    every teammate.  Learning it everywhere would let a crossing poison the very
    model meant to resolve that crossing.
    """
    K = len(init)
    tr = [np.c_[init[k], np.zeros((nf, 2))] for k in range(K)]
    model = [None] * K
    order = np.argsort(fr, kind="stable")
    fr, xy, R = fr[order], xy[order], R[order]
    # `app` is indexed by GLOBAL detection id, not by position in this subset,
    # because aidx (and therefore the learned model) carries global ids.
    ids = np.asarray(ids)[order] if ids is not None else np.arange(len(fr))
    b = np.searchsorted(fr, np.arange(nf + 1))
    prev = None
    for it in range(iters):
        assign = [dict() for _ in range(K)]
        aidx = [[] for _ in range(K)]
        last = [None] * K          # (frame, position) of last accepted detection
        nout = 0
        for f in range(nf):
            s, e = b[f], b[f + 1]
            if e <= s:
                continue
            idx = np.arange(s, e)
            C = np.full((K, len(idx)), 1e6)   # assignment cost (motion + appearance)
            M = np.full((K, len(idx)), 1e6)   # motion only -- this is the gate
            for k in range(K):
                for j, i in enumerate(idx):
                    # HARD physical gate: a robot cannot cover more than VMAX*dt,
                    # no matter what the process noise says is "probable".
                    if last[k] is not None:
                        dtk = (f - last[k][0]) * T.DT
                        if np.linalg.norm(xy[i] - last[k][1]) > T.VMAX * dtk + 0.5:
                            continue
                    d = xy[i] - tr[k][f, :2]
                    S = R[i] + np.diag([0.25, 0.25])   # slack for track error
                    m = float(d @ np.linalg.solve(S, d))
                    if m < GATE_EM:
                        # Appearance chooses WHICH slot takes a detection; it must
                        # never decide whether the detection is usable at all.
                        # Gating on the inflated cost throws away good detections
                        # whose appearance is merely noisy, which is why coverage
                        # collapsed as W_APP rose.
                        M[k, j] = m
                        C[k, j] = m
                        if app is not None and model[k] is not None:
                            C[k, j] += W_APP * _bhat(model[k], app[int(ids[i])])
            ri, ci = linear_sum_assignment(C)
            # MUTUAL EXCLUSION.  Hungarian stops two slots taking the SAME
            # detection, but not two slots taking two duplicate detections of
            # one robot.  That is the dominant identity failure: two slots take
            # turns on one robot while a third robot goes unclaimed, so the
            # robot looks well covered while its identity churns.  Where it
            # happens, keep the better-scoring slot and leave the other empty --
            # an empty slot re-seeks, a wrongly-filled one does not.
            take = [(k, j) for k, j in zip(ri, ci) if M[k, j] < GATE_EM]
            drop = set()
            for u in range(len(take)):
                for v in range(u + 1, len(take)):
                    ku, ju = take[u]; kv, jv = take[v]
                    if np.linalg.norm(xy[idx[ju]] - xy[idx[jv]]) < EXCL:
                        drop.add(v if C[ku, ju] <= C[kv, jv] else u)
            take = [t for n, t in enumerate(take) if n not in drop]
            hit = 0
            for k, j in take:
                if True:
                    assign[k][f] = (xy[idx[j]], R[idx[j]])
                    aidx[k].append((f, int(ids[idx[j]])))
                    last[k] = (f, xy[idx[j]]); hit += 1
            nout += len(idx) - hit
        for k in range(K):
            if assign[k]:
                f0 = min(assign[k])
                tr[k], _ = rts(assign[k], nf, np.r_[assign[k][f0][0], 0, 0])
        if app is not None:
            for k in range(K):
                keep = [i for f, i in aidx[k]
                        if all(np.linalg.norm(tr[k][f, :2] - tr[o][f, :2]) > SEP_OK
                               for o in range(K) if o != k)]
                if len(keep) >= 20:
                    m_ = app[keep].mean(0)
                    model[k] = m_ / max(m_.sum(), 1e-9)
        cnt = [len(a) for a in assign]
        if verbose:
            print(f"    iter {it:2d}  assigned {cnt}  outliers {nout}")
        if prev == cnt:
            break
        prev = cnt
    P = [rts(assign[k], nf, np.r_[assign[k][min(assign[k])][0], 0, 0])[1]
         if assign[k] else None for k in range(K)]
    return tr, assign, P, aidx
