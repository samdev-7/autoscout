"""Segment the carpet on the median plate and extract its near/far boundaries."""
import numpy as np, cv2

p = np.load("out/top_plate.npy")
hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV)
Hh, S, V = hsv[...,0].astype(int), hsv[...,1].astype(int), hsv[...,2].astype(int)

# carpet: low saturation, mid brightness, neutral grey
mask = ((S < 60) & (V > 55) & (V < 175)).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))

# keep the single largest connected component = the playing surface
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
carpet = (lab == i).astype(np.uint8)
print(f"carpet component: area={stats[i,cv2.CC_STAT_AREA]} "
      f"bbox x{stats[i,cv2.CC_STAT_LEFT]}..{stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH]} "
      f"y{stats[i,cv2.CC_STAT_TOP]}..{stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]}")

top_y, bot_y, xs = [], [], []
for x in range(carpet.shape[1]):
    col = np.where(carpet[:, x])[0]
    if len(col) < 40: continue
    xs.append(x); top_y.append(col.min()); bot_y.append(col.max())
xs, top_y, bot_y = np.array(xs), np.array(top_y, float), np.array(bot_y, float)

def robust_line(x, y, it=6):
    m = np.polyfit(x, y, 1)
    for _ in range(it):
        r = y - np.polyval(m, x)
        k = np.abs(r) < max(2.0, 2.0*r.std())
        if k.sum() < 50: break
        m = np.polyfit(x[k], y[k], 1)
    r = y - np.polyval(m, x); k = np.abs(r) < max(2.0, 2.0*r.std())
    return m, k

mt, kt = robust_line(xs, top_y)
mb, kb = robust_line(xs, bot_y)
print(f"\nfar  edge: y = {mt[0]:+.5f}x + {mt[1]:7.2f}   inliers {kt.sum()}/{len(xs)}")
print(f"near edge: y = {mb[0]:+.5f}x + {mb[1]:7.2f}   inliers {kb.sum()}/{len(xs)}")
for x in (181, 500, 960, 1400, 1740):
    print(f"  x={x:4d}  far_y={np.polyval(mt,x):6.1f}  near_y={np.polyval(mb,x):6.1f}"
          f"  height={np.polyval(mb,x)-np.polyval(mt,x):6.1f}px")

vis = cv2.cvtColor(p, cv2.COLOR_RGB2BGR).copy()
vis[carpet.astype(bool)] = (0.6*vis[carpet.astype(bool)] + 0.4*np.array([0,180,0])).astype(np.uint8)
for x, y, k in zip(xs, top_y, kt):
    cv2.circle(vis,(int(x),int(y)),1,(0,255,255) if k else (0,0,255),-1)
for x, y, k in zip(xs, bot_y, kb):
    cv2.circle(vis,(int(x),int(y)),1,(255,255,0) if k else (0,0,255),-1)
for m,c in ((mt,(0,255,255)),(mb,(255,255,0))):
    cv2.line(vis,(0,int(np.polyval(m,0))),(1919,int(np.polyval(m,1919))),c,1)
cv2.imwrite("out/carpet.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY,93])
np.save("out/carpet_mask.npy", carpet)
print("\nwrote out/carpet.jpg")
