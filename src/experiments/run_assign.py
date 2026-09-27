"""Driver: tracklet-level identity assignment -> six smoothed trajectories."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, track as T, fit6, assign as AS

DETS = os.environ.get("DETS", "out/f.npz")
XY = os.environ.get("XY", "out/fxy.npy")
APP = os.environ.get("APP", "out/fapp.npz")
OUT = os.environ.get("OUT", "out/tracks_asg.npy")

d = np.load(DETS, allow_pickle=True)["dets"]
xy = np.load(XY)
A = np.load(APP)
app = A[os.environ.get("FEAT", "chas")]
app = app / np.maximum(app.sum(1, keepdims=True), 1e-9)
nf = 4921

tks, al = AS.build(d, xy, app)
print(f"{len(tks)} tracklets  (red {(al==1).sum()} blue {(al==2).sum()} "
      f"unk {(al==0).sum()})")
fr_all = d[:, 0].astype(int); order = np.argsort(fr_all, kind="stable")

tracks = []
for a, name in ((1, "red"), (2, "blue")):
    idx = np.where(al == a)[0]
    sub = [tks[i] for i in idx]
    lab = AS.assign(sub, K=3)
    print(f"\n{name}: {len(sub)} tracklets -> "
          f"{[int((lab==c).sum()) for c in range(3)]} per identity, "
          f"{int((lab<0).sum())} unassigned")
    for c in range(3):
        mem = [sub[i] for i in range(len(sub)) if lab[i] == c]
        obs = {}
        for t in mem:
            for f, p in zip(t.fr, t.xs):
                obs.setdefault(int(f), []).append(p)
        obs = {f: (np.mean(v, 0), geom.cov(np.mean(v, 0)[None], su=3.0, sv=4.0)[0])
               for f, v in obs.items()}
        if not obs:
            continue
        f0 = min(obs)
        xs, P = fit6.rts(obs, nf, np.r_[obs[f0][0], 0, 0])
        xs[:, 0] = np.clip(xs[:, 0], 0, geom.LEN)
        xs[:, 1] = np.clip(xs[:, 1], 0, geom.WID)
        m = np.zeros(nf, bool); m[list(obs)] = True
        sp = np.hypot(xs[:, 2], xs[:, 3])
        tracks.append(dict(alliance=name, xy=xs[:, :2], v=xs[:, 2:], obs=m, P=P,
                           aidx=[]))
        print(f"  track {len(tracks)-1}: {len(mem):3d} tracklets, observed "
              f"{100*m.mean():5.1f}%, speed p50 {np.median(sp):.2f} "
              f"p99 {np.percentile(sp,99):.2f} max {sp.max():.2f} m/s")
np.save(OUT, np.array(tracks, dtype=object), allow_pickle=True)
print(f"\nsaved {OUT}")
