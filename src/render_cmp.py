"""Side-by-side tracker comparison: two independent columns.

Each column is one tracker -- its own broadcast panel above its own field view.
Detection boxes in the broadcast panel are coloured by whichever track claims
them, using the same palette as the field markers, so an identity swap is
visible as a colour change in BOTH panels at once rather than having to be
inferred from position.
"""
import cv2, numpy as np, av, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, mcam
from geom import LEN, WID, BAND0, BAND1

CW = int(os.environ.get("CW", "1280"))      # width of one column
SRC_W, SRC_H = 1920, BAND1 - BAND0
VH = int(round(SRC_H * CW / SRC_W))         # video panel height in a column
FB = cv2.imread("out/field_box.png")
FH = int(round(FB.shape[0] * CW / FB.shape[1]))
FBH = cv2.resize(FB, (CW, FH), interpolation=cv2.INTER_AREA)
SC = CW / SRC_W
SX, SY = CW / LEN, FH / WID
SW, SH = CW // 2, int(round(540 * (CW // 2) / 960))   # one station panel in the strip
SSC = SW / 960
CLAIM_ST = 2.0     # m; station depth error is far larger than the wide view's
ST_CROP = {"red": (540, 1080, 0, 960), "blue": (540, 1080, 960, 1920)}
_ST = None


def station_dets():
    """Station detections (conf>=0.30) with corner-corrected field positions, by frame."""
    global _ST
    if _ST is not None:
        return _ST
    _ST = {}
    for panel, path in (("red", "out/dets_red_station.npz"), ("blue", "out/dets_blue_station.npz")):
        d = np.load(path, allow_pickle=True)["dets"]; fr = d[:, 0].astype(int)
        xy = np.full((len(d), 2), np.nan)
        if panel == "red":
            for name, m in (("red_station", fr < mcam.RED_CUT0), ("red_station_b", fr >= mcam.RED_CUT1)):
                xy[m] = mcam.cam(name).box_to_field(d[m, 3:7])
        else:
            xy[:] = mcam.cam("blue_station").box_to_field(d[:, 3:7])
        ok = (d[:, 7] >= 0.30) & np.isfinite(xy).all(1)
        byf = {}
        for k in np.where(ok)[0]:
            byf.setdefault(int(fr[k]), []).append((d[k, 3:7], xy[k]))
        _ST[panel] = byf
    return _ST

TAIL, BANDS, MAXGAP, CLAIM = 90, 6, 4, 1.2       # 3 s tail, still broken at observation gaps
LOST_M = 1.5       # 2-sigma beyond this and the track is not really localised
CAP_M = 3.0        # never draw a bigger ellipse than this
COLS = [(50, 50, 240), (60, 150, 255), (120, 60, 200),
        (240, 120, 40), (225, 200, 60), (200, 90, 130)]


def f2px(P):
    P = np.atleast_2d(np.asarray(P, float))
    return np.c_[CW * (1 - P[:, 0] / LEN), FH * (P[:, 1] / WID)]


def tail_pts(tr, f):
    obs = tr["obs"]
    if not obs[max(f - MAXGAP, 0):f + 1].any():
        return None
    s, gap = f, 0
    while s > 0 and f - s < TAIL:
        s -= 1
        gap = 0 if obs[s] else gap + 1
        if gap > MAXGAP:
            s += gap; break
    return tr["xy"][s:f + 1] if f - s >= 2 else None


def ell(tr, f):
    """2-sigma ellipse, plus whether the track is effectively lost.

    An unobserved track's covariance grows without bound, which is honest but
    visually swamps the frame.  Past LOST_M we stop filling it and cap the drawn
    size: the reader still sees "uncertain", without a third of the field being
    covered by one robot's doubt.
    """
    if tr.get("P") is None:
        return None
    w, V = np.linalg.eigh(tr["P"][f][:2, :2])
    ax = np.sqrt(np.maximum(w, 1e-9)) * 2.0
    lost = float(ax.max()) > LOST_M
    ax = np.minimum(ax, CAP_M)
    return (max(int(ax[1] * SX), 2), max(int(ax[0] * SY), 2)), \
        float(np.degrees(np.arctan2(V[1, 1], V[0, 1]))), lost


def video_panel(frame, tracks, f, dets, xy):
    """Broadcast crop with boxes coloured by the track that claims them."""
    img = cv2.resize(frame, (CW, VH), interpolation=cv2.INTER_AREA)
    if dets is not None and len(dets):
        m = dets[:, 0].astype(int) == f
        for b, p in zip(dets[m], xy[m]):
            col, best = (170, 170, 170), CLAIM
            for k, tr in enumerate(tracks):
                if not tr["obs"][max(f - MAXGAP, 0):f + 1].any():
                    continue
                d = np.linalg.norm(tr["xy"][f] - p)
                if d < best:
                    best, col = d, COLS[k % 6]
            x0, y0, x1, y1 = (b[3:7] * SC).astype(int)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
    for k, tr in enumerate(tracks):
        P = tr["xy"][f]
        if not np.isfinite(P).all():
            continue
        u, v = geom.field_to_img(P[None])[0]
        u, v = int(u * SC), int((v - BAND0) * SC)
        live = tr["obs"][max(f - MAXGAP, 0):f + 1].any()
        cv2.circle(img, (u, v), 5, COLS[k % 6], -1 if live else 2, cv2.LINE_AA)
        cv2.circle(img, (u, v), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def station_strip(full, tracks, f):
    """Red and blue station panels side by side; boxes coloured by the track that
    claims them, each track's estimate projected into the view as a dot."""
    ST = station_dets(); out = []
    for panel in ("red", "blue"):
        y0, y1, x0, x1 = ST_CROP[panel]
        img = cv2.resize(full[y0:y1, x0:x1], (SW, SH), interpolation=cv2.INTER_AREA)
        view = mcam.red_view(f) if panel == "red" else "blue_station"
        for b, p in ST[panel].get(f, []):
            col, best = (170, 170, 170), CLAIM_ST
            for k, tr in enumerate(tracks):
                if not tr["obs"][max(f - MAXGAP, 0):f + 1].any():
                    continue
                d = np.linalg.norm(tr["xy"][f] - p)
                if d < best:
                    best, col = d, COLS[k % 6]
            bx0, by0, bx1, by1 = (b * SSC).astype(int)
            cv2.rectangle(img, (bx0, by0), (bx1, by1), col, 2)
        if view is not None:
            cam = mcam.cam(view)
            for k, tr in enumerate(tracks):
                P = tr["xy"][f]
                if not np.isfinite(P).all():
                    continue
                q = cam.project(P[None])[0]
                if not np.isfinite(q).all() or not (0 <= q[0] < 960 and 0 <= q[1] < 540):
                    continue
                u, v = int(q[0] * SSC), int(q[1] * SSC)
                live = tr["obs"][max(f - MAXGAP, 0):f + 1].any()
                cv2.circle(img, (u, v), 5, COLS[k % 6], -1 if live else 2, cv2.LINE_AA)
                cv2.circle(img, (u, v), 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (SW, 22), (0, 0, 0), -1)
        cv2.putText(img, f"{panel} station" + ("" if view else "  (camera moving)"), (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(img)
    strip = np.hstack(out)
    cv2.line(strip, (SW, 0), (SW, SH), (25, 25, 25), 2)
    return strip


CAM_PX = {n: f2px(mcam.cam(n).C[:2])[0] for n in ("wide", "red_station", "red_station_b", "blue_station")}
ORANGE, GREEN = np.array([0, 140, 255], float), np.array([60, 200, 60], float)


def cam_lines(img, tracks, f, dets, xy, cams):
    """One line from each robot to every camera that detected it this frame,
    coloured by that camera's share of the fused information (orange = little
    weight, green = most).  Weight is tr(R^-1) at the robot's position, so a
    station looking along its depth axis is orange and the wide view is green;
    a robot seen by all three cameras has three lines."""
    ST = station_dets() if any(c != "wide" for c in cams) else {}
    for k, tr in enumerate(tracks):
        P = tr["xy"][f]
        if not np.isfinite(P).all() or not tr["obs"][max(f - MAXGAP, 0):f + 1].any():
            continue
        seen = []
        if "wide" in cams and dets is not None:
            m = dets[:, 0].astype(int) == f
            if m.any() and np.linalg.norm(xy[m] - P, axis=1).min() < CLAIM:
                seen.append("wide")
        for panel in ("red", "blue"):
            view = mcam.red_view(f) if panel == "red" else "blue_station"
            if view is None or view not in cams and panel + "_station" not in cams:
                continue
            ds = ST.get(panel, {}).get(f, [])
            if ds and min(np.linalg.norm(p - P) for _, p in ds) < CLAIM_ST:
                seen.append(view)
        if not seen:
            continue
        info = []
        for n in seen:
            C = mcam.cam(n).cov(P[None])[0]
            info.append(np.trace(np.linalg.inv(C)) if np.isfinite(C).all() else 0.0)
        tot = sum(info) or 1.0
        p0 = tuple(int(v) for v in f2px(P)[0])
        for n, w in zip(seen, info):
            share = w / tot
            col = tuple(int(v) for v in (ORANGE + (GREEN - ORANGE) * share))
            q = CAM_PX[n]
            ok, a, b = cv2.clipLine((0, 0, CW, FH), p0, (int(q[0]), int(q[1])))
            if ok:
                cv2.line(img, a, b, col, 2, cv2.LINE_AA)


def field_panel(tracks, f, dets=None, xy=None, cams=("wide",)):
    img = FBH.copy()
    cam_lines(img, tracks, f, dets, xy, cams)
    segs = []
    for k, tr in enumerate(tracks):
        p = tail_pts(tr, f)
        if p is not None and len(p) >= 2:
            segs.append((COLS[k % 6], f2px(p), np.linspace(0, 1, len(p))))
    for b in range(BANDS):
        lo, hi = b / BANDS, (b + 1) / BANDS
        ov, any_ = img.copy(), False
        for c, p, a in segs:
            idx = np.where((a >= lo) & (a < hi))[0]
            if len(idx) and idx[-1] + 1 < len(p):
                idx = np.r_[idx, idx[-1] + 1]          # share the boundary point so bands join
            if len(idx) < 2:
                continue
            th = max(2, int(2 + 4 * (1 - (lo + hi) / 2)))
            pts = [p[idx].astype(np.int32)]
            cv2.polylines(ov, pts, False, (30, 30, 30), th + 2, cv2.LINE_AA)   # dark outline for contrast
            cv2.polylines(ov, pts, False, c, th, cv2.LINE_AA)
            any_ = True
        if any_:
            al = 0.40 + 0.60 * (1 - (lo + hi) / 2)
            cv2.addWeighted(ov, al, img, 1 - al, 0, img)
    ov = img.copy()
    for k, tr in enumerate(tracks):
        P = tr["xy"][f]
        e = ell(tr, f)
        if not np.isfinite(P).all() or e is None or e[2]:
            continue                       # lost tracks are not filled
        px, py = f2px(P)[0]
        cv2.ellipse(ov, (int(px), int(py)), e[0], e[1], 0, 360, COLS[k % 6], -1)
    cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
    for k, tr in enumerate(tracks):
        P = tr["xy"][f]
        if not np.isfinite(P).all():
            continue
        px, py = f2px(P)[0]
        e = ell(tr, f)
        if e:
            cv2.ellipse(img, (int(px), int(py)), e[0], e[1], 0, 360,
                        COLS[k % 6], 1, cv2.LINE_AA)
        live = tr["obs"][max(f - MAXGAP, 0):f + 1].any()
        r = 7 if not (e and e[2]) else 5
        cv2.circle(img, (int(px), int(py)), r, COLS[k % 6], -1 if live else 2, cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py)), r, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def column(frame, tracks, f, dets, xy, label, full=None, cams=("wide",)):
    parts = [video_panel(frame, tracks, f, dets, xy), field_panel(tracks, f, dets, xy, cams)]
    if full is not None:
        parts.append(station_strip(full, tracks, f))
    col = np.vstack(parts)
    cv2.line(col, (0, VH), (CW, VH), (40, 40, 40), 2)
    if full is not None:
        cv2.line(col, (0, VH + FH), (CW, VH + FH), (40, 40, 40), 2)
    cv2.rectangle(col, (0, 0), (CW, 26), (0, 0, 0), -1)
    cv2.putText(col, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return col


ALL_CAMS = ("wide", "red_station", "red_station_b", "blue_station")


def render(A, B, la, lb, t0, t1, out, dA=None, xA=None, dB=None, xB=None, stations=False,
           camsA=("wide",), camsB=ALL_CAMS):
    c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
    H = VH + FH + (SH if stations else 0)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * CW, H))
    n = 0
    for fr in c.decode(s):
        if fr.time is None or fr.time < t0:
            continue
        if fr.time > t1:
            break
        f = int(round((fr.time - 8.0) * 30))
        if f < 0 or f >= len(A[0]["xy"]):
            continue
        full = cv2.cvtColor(fr.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
        frame = full[BAND0:BAND1]
        fs = full if stations else None
        img = np.hstack([column(frame, A, f, dA, xA, la, fs, camsA),
                         column(frame, B, f, dB, xB, lb, fs, camsB)])
        cv2.line(img, (CW, 0), (CW, H), (25, 25, 25), 3)
        cv2.rectangle(img, (10, VH + 6), (172, VH + 34), (0, 0, 0), -1)
        cv2.putText(img, f"t = {fr.time:6.2f} s", (16, VH + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img); n += 1
    vw.release(); c.close()
    print(f"wrote {out} ({n} frames, {2*CW}x{H})")


def render_one(tracks, label, t0, t1, out, dets=None, xy=None, stations=True, cams=ALL_CAMS):
    """Single column: one tracker, broadcast panel over field diagram (over station strip)."""
    c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
    s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
    H = VH + FH + (SH if stations else 0)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (CW, H))
    n = 0
    for fr in c.decode(s):
        if fr.time is None or fr.time < t0:
            continue
        if fr.time > t1:
            break
        f = int(round((fr.time - 8.0) * 30))
        if f < 0 or f >= len(tracks[0]["xy"]):
            continue
        full = cv2.cvtColor(fr.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
        img = column(full[BAND0:BAND1], tracks, f, dets, xy, label, full if stations else None, cams)
        cv2.rectangle(img, (10, VH + 6), (172, VH + 34), (0, 0, 0), -1)
        cv2.putText(img, f"t = {fr.time:6.2f} s", (16, VH + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img); n += 1
    vw.release(); c.close()
    print(f"wrote {out} ({n} frames, {CW}x{H})")
