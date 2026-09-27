"""Project the calibrated field frame onto live match frames."""
import av, numpy as np, cv2, json
LEN,WID,IN=16.541,8.069,0.0254
H=np.load("out/H_field_to_img.npy")
tags=json.load(open("out/tags.json"))
P=lambda X,Y: cv2.perspectiveTransform(np.array([[[X,Y]]],float),H).reshape(2)

def grab(ts):
    out=[];c=av.open("data/match1_qual.mp4");s=c.streams.video[0]
    s.codec_context.options={"hwaccel":"videotoolbox"}
    for t in ts:
        c.seek(int(t/s.time_base),stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time>=t-0.05:
                out.append((t,f.to_ndarray(format="rgb24")[0:490])); break
    c.close(); return out

for t,fr in grab([35.0,95.0,150.0]):
    im=cv2.cvtColor(fr,cv2.COLOR_RGB2BGR)
    for X in np.arange(0,LEN+1e-6,1.0):            # 1 m grid
        pl=[P(min(X,LEN),y) for y in np.linspace(0,WID,30)]
        cv2.polylines(im,[np.int32(pl)],False,(255,220,0),1,cv2.LINE_AA)
    for Y in np.arange(0,WID+1e-6,1.0):
        pl=[P(x,min(Y,WID)) for x in np.linspace(0,LEN,60)]
        cv2.polylines(im,[np.int32(pl)],False,(255,220,0),1,cv2.LINE_AA)
    bd=[P(*c) for c in [(0,0),(LEN,0),(LEN,WID),(0,WID)]]
    cv2.polylines(im,[np.int32(bd)],True,(0,80,255),3,cv2.LINE_AA)
    for tg in tags:                                 # every tag's carpet footprint
        p=P(tg["X"],tg["Y"])
        if -200<p[0]<2120 and -100<p[1]<600:
            cv2.drawMarker(im,tuple(np.int32(p)),(255,0,255),cv2.MARKER_TILTED_CROSS,10,2)
    for tid,uo,vo in [(2,616.8,210.9),(11,583.3,211.2),(21,1303.8,204.6)]:
        tg=next(g for g in tags if g["id"]==tid); p=P(tg["X"],tg["Y"])
        cv2.line(im,tuple(np.int32(p)),(int(uo),int(vo)),(0,255,255),2,cv2.LINE_AA)
        cv2.circle(im,(int(uo),int(vo)),7,(0,255,255),2)
        cv2.putText(im,f"tag{tid}",(int(uo)+9,int(vo)-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
    cv2.putText(im,f"t={t:.0f}s  1m grid  magenta=tag carpet pts  cyan=observed tag (44.25in up)",
                (12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(f"out/overlay_t{int(t)}.jpg",im,[cv2.IMWRITE_JPEG_QUALITY,93])
    print(f"wrote out/overlay_t{int(t)}.jpg")

# scale check: how many inches does one pixel span, near vs far?
print("\nground resolution across the field (in/px):")
for Y in (0.5,2.0,4.0,6.0,7.5):
    a,b=P(8.0,Y),P(8.0+0.1,Y); sx=0.1/np.linalg.norm(b-a)/IN
    c,d=P(8.0,Y),P(8.0,Y+0.1); sy=0.1/np.linalg.norm(d-c)/IN
    print(f"  Y={Y:4.1f}m  along-X {sx:6.2f} in/px   along-Y {sy:6.2f} in/px")
