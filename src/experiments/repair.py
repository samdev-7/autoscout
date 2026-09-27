"""Post-hoc identity repair at crossings.

A crossing is where position stops being informative, so the tracker's per-frame
choice there is close to a coin flip.  This does not try to make that per-frame
choice better.  Instead it treats each crossing as a single binary decision --
did the two tracks swap or not -- and answers it by comparing appearance over
the CLEAN segments before and after, where the robots are well separated and
descriptors average down to AUC ~0.94.

Doing it after the fit, rather than inside it, is the whole point: an appearance
model learned during EM is built from the current assignment and so reinforces
whatever swap already happened.  Here the evidence comes from outside the window
the decision is about.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, track as T, fit6

CROSS_D = 2.0      # m; closer than this and identity is ambiguous
JOIN = 15          # merge windows separated by less than this many frames
WIN = 90           # frames of clean evidence to average on each side
MINS = 12          # minimum descriptors per side to trust a decision
MARGIN = 0.010     # required advantage before we swap


def _bhat(p, q):
    return float(1.0 - np.sqrt(p * q).sum())


def windows(pi, pj, nf):
    close = np.linalg.norm(pi - pj, axis=1) < CROSS_D
    out, s = [], None
    for f in range(nf):
        if close[f] and s is None:
            s = f
        elif not close[f] and s is not None:
            out.append([s, f - 1]); s = None
    if s is not None:
        out.append([s, nf - 1])
    m = []
    for w in out:
        if m and w[0] - m[-1][1] < JOIN:
            m[-1][1] = w[1]
        else:
            m.append(w)
    return [w for w in m if w[1] - w[0] >= 3]


def model(asg, app, f0, f1):
    d = [app[i] for f, i in asg if f0 <= f < f1]
    if len(d) < MINS:
        return None
    m = np.mean(d, 0)
    return m / max(m.sum(), 1e-9)


def repair(tracks, app, xy, nf, rounds=3, verbose=True):
    asg = [list(t["aidx"]) for t in tracks]
    pos = [t["xy"].copy() for t in tracks]
    nsw = 0
    for rd in range(rounds):
        cand = []
        for a in (0, 3):
            for i in range(a, a + 3):
                for j in range(i + 1, a + 3):
                    for w in windows(pos[i], pos[j], nf):
                        cand.append((w[0], w[1], i, j))
        cand.sort()
        did = 0
        for w0, w1, i, j in cand:
            bi = model(asg[i], app, w0 - WIN, w0)
            bj = model(asg[j], app, w0 - WIN, w0)
            ai = model(asg[i], app, w1 + 1, w1 + 1 + WIN)
            aj = model(asg[j], app, w1 + 1, w1 + 1 + WIN)
            if any(x is None for x in (bi, bj, ai, aj)):
                continue
            keep = _bhat(bi, ai) + _bhat(bj, aj)
            swap = _bhat(bi, aj) + _bhat(bj, ai)
            if swap < keep - MARGIN:
                ki = [(f, k) for f, k in asg[i] if f > w1]
                kj = [(f, k) for f, k in asg[j] if f > w1]
                asg[i] = [(f, k) for f, k in asg[i] if f <= w1] + kj
                asg[j] = [(f, k) for f, k in asg[j] if f <= w1] + ki
                did += 1; nsw += 1
        for k in range(len(tracks)):
            asg[k].sort()
            o = {f: (xy[i], geom.cov(xy[i][None], su=3.0, sv=4.0)[0])
                 for f, i in asg[k]}
            if not o:
                continue
            f0 = min(o)
            pos[k] = fit6.rts(o, nf, np.r_[o[f0][0], 0, 0])[0][:, :2]
        if verbose:
            print(f"  round {rd}: {len(cand)} crossing windows, {did} swaps applied")
        if did == 0:
            break
    out = []
    for k, t in enumerate(tracks):
        o = {f: (xy[i], geom.cov(xy[i][None], su=3.0, sv=4.0)[0]) for f, i in asg[k]}
        f0 = min(o)
        xs, P = fit6.rts(o, nf, np.r_[o[f0][0], 0, 0])
        xs[:, 0] = np.clip(xs[:, 0], 0, geom.LEN)
        xs[:, 1] = np.clip(xs[:, 1], 0, geom.WID)
        obs = np.zeros(nf, bool); obs[[f for f, _ in asg[k]]] = True
        out.append(dict(alliance=t["alliance"], xy=xs[:, :2], v=xs[:, 2:],
                        obs=obs, P=P, aidx=asg[k]))
    print(f"total swaps applied: {nsw}")
    return out


if __name__ == "__main__":
    tr = list(np.load("out/tracks_em.npy", allow_pickle=True))
    A = np.load("out/appear.npz")
    app = np.concatenate([A["body"], A["chas"]], 1)
    app = app / np.maximum(app.sum(1, keepdims=True), 1e-9)
    xy = np.load("out/dets_xy.npy")
    nf = len(tr[0]["xy"])
    out = repair(tr, app, xy, nf)
    np.save("out/tracks_rep.npy", np.array(out, dtype=object), allow_pickle=True)
    print("saved out/tracks_rep.npy")
