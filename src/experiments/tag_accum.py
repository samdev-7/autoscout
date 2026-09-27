"""Accumulate AprilTag detections across the match.

Cameras are static within a match, so a tag seen in ANY frame pins a point for
ALL frames. Per-match accumulation = auto-recalibration when rigging changes.
"""
import av, numpy as np, cv2, time
from collections import defaultdict

PATH="data/match1_qual.mp4"; T0,T1=7.1,172.8; STRIDE=15   # ~2 fps
PANELS={"TOP":(0,0,1920,490),"BL":(5,545,955,1060),"BR":(965,545,1915,1060)}
det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                            cv2.aruco.DetectorParameters())

acc={p:defaultdict(list) for p in PANELS}
c=av.open(PATH); s=c.streams.video[0]
s.codec_context.options={"hwaccel":"videotoolbox"}; s.thread_type="AUTO"
n=0; t0=time.time()
for i,f in enumerate(c.decode(s)):
    if f.time is None or not (T0<=f.time<=T1) or i%STRIDE: continue
    a=f.to_ndarray(format="rgb24"); n+=1
    for p,(x0,y0,x1,y1) in PANELS.items():
        g=cv2.cvtColor(a[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
        for up in (1,2):
            gg=g if up==1 else cv2.resize(g,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            cor,ids,_=det.detectMarkers(gg)
            if ids is None: continue
            for j,cid in enumerate(ids.flatten()):
                acc[p][int(cid)].append(cor[j][0].reshape(4,2)/up)
c.close()
print(f"scanned {n} frames in {time.time()-t0:.1f}s ({n/(time.time()-t0):.0f} fps)\n")

for p in PANELS:
    d=acc[p]
    print(f"=== {p} : {len(d)} distinct tags over {n} frames ===")
    for cid in sorted(d):
        arr=np.stack(d[cid])                 # (k,4,2)
        ctr=arr.mean(axis=1)                 # (k,2) centre per detection
        med=np.median(ctr,axis=0)
        spread=np.median(np.abs(ctr-med),axis=0).max()   # robust scatter, px
        size=np.median([cv2.contourArea(x.astype(np.float32))**0.5 for x in arr])
        print(f"   id{cid:<3} seen {len(arr):4d}/{n} ({len(arr)/n*100:5.1f}%)  "
              f"centre=({med[0]:7.1f},{med[1]:6.1f})  scatter={spread:4.2f}px  size={size:4.1f}px")
    print()
