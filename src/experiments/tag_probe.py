"""Can we detect the field's AprilTags in each panel? 36h11 is the FRC family."""
import av, numpy as np, cv2

PATH = "data/match1_qual.mp4"
def grab(times):
    out=[]; c=av.open(PATH); s=c.streams.video[0]
    s.codec_context.options={"hwaccel":"videotoolbox"}
    for t in times:
        c.seek(int(t/s.time_base), stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time>=t-0.05: out.append((t,f.to_ndarray(format="rgb24"))); break
    c.close(); return out

det = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
    cv2.aruco.DetectorParameters())

PANELS = {"TOP":(0,0,1920,490), "BL":(5,545,955,1060), "BR":(965,545,1915,1060)}
frames = grab([20.0, 60.0, 100.0, 140.0, 170.0])

for name,(x0,y0,x1,y1) in PANELS.items():
    print(f"\n=== {name} ({x1-x0}x{y1-y0}) ===")
    for t,f in frames:
        crop = f[y0:y1, x0:x1]
        g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        found = {}
        for up,lab in ((1,"1x"),(2,"2x"),(4,"4x")):
            gg = g if up==1 else cv2.resize(g,None,fx=up,fy=up,interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = det.detectMarkers(gg)
            if ids is not None:
                for i,cid in enumerate(ids.flatten()):
                    px = cv2.contourArea(corners[i][0])**0.5/up
                    found.setdefault(int(cid), (lab, px))
        if found:
            desc = ", ".join(f"id{k}@{v[1]:.0f}px({v[0]})" for k,v in sorted(found.items()))
        else:
            desc = "none"
        print(f"  t={t:5.0f}s  {len(found)} tags: {desc}")
