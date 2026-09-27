"""Field <-> image geometry for the locked-off wide camera.

Everything assumes robots sit on the carpet (Z=0).  Height and depth are not
separable from one view, so the flat-ground assumption *is* the resolution --
we do not model height at all.  What that buys us is that the only measurement
that matters is the box's bottom edge, and the only thing that can go wrong is
that edge not being a real floor contact (occlusion, truncation).

Image coordinates are FULL-FRAME (origin at the top of the 1920x490 wide
panel).  Detections come from the band crop, so add BAND0 to their v.
"""
import numpy as np, cv2

LEN, WID = 16.541, 8.069          # field, metres (2026 REBUILT)
BAND0, BAND1 = 98, 442            # rows of the wide panel the field occupies
FOOTPRINT = 0.91                  # robot square incl. bumpers, metres (36 in)

_p = np.load("out/cam_params.npy")
K = np.array([[_p[0], 0, _p[1]], [0, _p[0], _p[2]], [0, 0, 1]])
DIST = np.zeros(5); DIST[0] = _p[9]
RVEC, TVEC = _p[3:6].copy(), _p[6:9].copy()
R, _ = cv2.Rodrigues(RVEC)
CAM = (-R.T @ TVEC).ravel()       # camera centre in field coords

# Yaw-averaged distance from footprint centre to the corner nearest the camera.
# Support function of a square of side a in direction t is (a/2)(|cos|+|sin|).
_t = np.linspace(0, np.pi / 2, 20001)
_s = (FOOTPRINT / 2) * (np.abs(np.cos(_t)) + np.abs(np.sin(_t)))
CORNER_MEAN, CORNER_STD = float(_s.mean()), float(_s.std())


def field_to_img(P):
    """(...,3) or (...,2) field points -> (...,2) full-frame pixels."""
    P = np.asarray(P, float)
    if P.shape[-1] == 2:
        P = np.concatenate([P, np.zeros(P.shape[:-1] + (1,))], -1)
    q, _ = cv2.projectPoints(P.reshape(-1, 1, 3), RVEC, TVEC, K, DIST)
    return q.reshape(P.shape[:-1] + (2,))


def img_to_plane(uv, Z=0.0):
    """(...,2) full-frame pixels -> (...,2) field XY on the horizontal plane Z.

    Z != 0 is how a click on the TOP of a robot is used: the top sits directly
    above the base, so intersecting the ray with the plane at the robot's height
    gives its ground XY without ever seeing the floor contact.
    """
    uv = np.asarray(uv, float).reshape(-1, 1, 2)
    n = cv2.undistortPoints(uv, K, DIST).reshape(-1, 2)      # normalised, undistorted
    d = np.concatenate([n, np.ones((len(n), 1))], 1) @ R      # ray dirs in field frame
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (Z - CAM[2]) / d[:, 2]
    G = CAM[None, :] + s[:, None] * d
    G[(s <= 0) | ~np.isfinite(s)] = np.nan
    return G[:, :2]


def img_to_ground(uv):
    """(...,2) full-frame pixels -> (...,2) field XY on the Z=0 plane."""
    return img_to_plane(uv, 0.0)


def height_of(uv_top, XY):
    """Height (m) that puts a robot's top at uv_top, given its ground XY."""
    lo, hi = 0.05, 2.5
    for _ in range(40):
        mid = (lo + hi) / 2
        v = field_to_img(np.c_[np.atleast_2d(XY), np.full((1, 1), mid)])[0][1]
        if v > np.asarray(uv_top, float).reshape(-1)[1]:
            lo = mid                      # projected point still too low
        else:
            hi = mid
    return (lo + hi) / 2


def _away(XY):
    """Horizontal unit vectors pointing from the camera towards each point."""
    v = np.atleast_2d(np.asarray(XY, float)) - CAM[:2]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def point_to_field(uv, corner=CORNER_MEAN):
    """A (x-centre, nearest-ground-corner) image point -> footprint centre.

    This is the convention BOTH the detector and the human tracer produce: the
    horizontal coordinate is the robot's left/right middle, the vertical one is
    where the bumper corner closest to the camera meets the carpet.  That point
    is about 0.58 m nearer the camera than the footprint centre, so it has to be
    pushed back along the view direction.  Skipping this puts every robot ~0.6 m
    too close to the camera.
    """
    G = img_to_ground(uv)
    return G + corner * _away(G)


def box_to_field(boxes, band_offset=BAND0, corner=CORNER_MEAN):
    """Detection boxes (N,4) xyxy in BAND-CROP pixels -> footprint centres (N,2).

    The AABB's bottom edge is the robot's ground corner *nearest* the camera,
    not its centre, so back-project that and then push away from the camera by
    the yaw-averaged corner distance.  Ignoring this biases every robot ~0.6 m
    towards the camera.
    """
    b = np.atleast_2d(np.asarray(boxes, float))
    uv = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3] + band_offset]
    return point_to_field(uv, corner)


def jac(XY):
    """d(u,v)/d(X,Y) at each ground point -> (N,2,2)."""
    P = np.atleast_2d(np.asarray(XY, float))
    J = np.empty((len(P), 2, 2))
    h = 1e-3
    for j in range(2):
        e = np.zeros(2); e[j] = h
        J[:, :, j] = (field_to_img(P + e) - field_to_img(P - e)) / (2 * h)
    return J


def cov(XY, su=3.0, sv=4.0, yaw=CORNER_STD):
    """Field-space measurement covariance (N,2,2) from pixel noise.

    Pixel noise is pushed through the inverse Jacobian, so the ellipse is
    automatically anisotropic and depth-dependent.  The yaw term is the part of
    the corner correction we cannot know without the robot's heading; it acts
    purely along the view direction.
    """
    P = np.atleast_2d(np.asarray(XY, float))
    Ji = np.linalg.inv(jac(P))
    S = Ji @ np.diag([su ** 2, sv ** 2]) @ Ji.transpose(0, 2, 1)
    u = _away(P)[:, :, None]
    return S + (yaw ** 2) * (u @ u.transpose(0, 2, 1))


def field_nms(frames, XY, conf, sep=0.70):
    """Suppress same-frame detections closer than `sep` metres.

    Image-space NMS uses one IoU threshold everywhere, but apparent size changes
    ~2x across the field, so it suppresses inconsistently with depth.  In field
    coordinates the rule is physical: two robots cannot have footprint centres
    closer than their own footprint (0.91 m), so anything nearer is one robot
    detected twice.  Keeps the highest-confidence detection of each cluster.
    """
    frames = np.asarray(frames).astype(int)
    XY = np.atleast_2d(np.asarray(XY, float))
    keep = np.zeros(len(XY), bool)
    for f in np.unique(frames):
        idx = np.where(frames == f)[0]
        idx = idx[np.argsort(-np.asarray(conf)[idx])]
        taken = []
        for i in idx:
            p = XY[i]
            if not np.isfinite(p).all():
                continue
            if all(np.linalg.norm(p - XY[j]) >= sep for j in taken):
                taken.append(i); keep[i] = True
    return keep


def in_field(XY, pad=0.5):
    P = np.atleast_2d(np.asarray(XY, float))
    return ((P[:, 0] > -pad) & (P[:, 0] < LEN + pad) &
            (P[:, 1] > -pad) & (P[:, 1] < WID + pad))


if __name__ == "__main__":
    print(f"camera centre  X {CAM[0]:.2f}  Y {CAM[1]:.2f}  Z {CAM[2]:.2f} m")
    print(f"corner offset  {CORNER_MEAN:.3f} +/- {CORNER_STD:.3f} m "
          f"(range {_s.min():.3f}..{_s.max():.3f})\n")

    g = np.array([[x, y] for x in np.linspace(0.3, LEN - .3, 25)
                          for y in np.linspace(0.3, WID - .3, 12)])
    rt = img_to_ground(field_to_img(g))
    err = np.linalg.norm(rt - g, axis=1)
    print(f"round trip field->image->field over {len(g)} points: "
          f"max {err.max()*1000:.3f} mm, mean {err.mean()*1000:.3f} mm")

    print("\nmeasurement 1-sigma from 3 px lateral / 4 px vertical noise:")
    print(f"{'X':>5} {'Y':>5} | {'major':>7} {'minor':>7} | {'sig_X':>6} {'sig_Y':>6}")
    for Y in (0.6, 4.0, 7.4):
        for X in (2.0, 8.27, 14.5):
            S = cov([[X, Y]])[0]
            w = np.sqrt(np.linalg.eigvalsh(S))
            print(f"{X:5.1f} {Y:5.1f} | {w[1]:7.3f} {w[0]:7.3f} | "
                  f"{np.sqrt(S[0,0]):6.3f} {np.sqrt(S[1,1]):6.3f}")

    print("\nsanity: field corners in image (must sit inside the 98..442 band)")
    for X, Y in [(0, 0), (LEN, 0), (0, WID), (LEN, WID)]:
        u, v = field_to_img([[X, Y]])[0]
        print(f"  ({X:5.2f},{Y:4.2f}) -> ({u:7.1f},{v:6.1f})")
