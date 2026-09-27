"""Colour-gated proposals: foreground AND saturated red/blue bumper.

v1 failed because moving balls dominate the foreground and merge robots into
oversized blobs. Bumpers are strongly red/blue; balls are yellow; carpet is grey.
Gate the difference mask on bumper colour BEFORE any morphology.
"""
import av, numpy as np, cv2
LEN,WID=16.541,8.069
p=np.load("out/cam_params.npy")
K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]); d=np.zeros(5); d[0]=p[9]
rv,tv=p[3:6],p[6:9]
def f2i(X,Y,Z=0.0):
    q,_=cv2.projectPoints(np.array([[[X,Y,Z]]]),rv,tv,K,d); return q.reshape(2)
plate=np.load("out/top_plate.npy")
poly=np.int32([f2i(x,y) for x,y in [(0,0),(LEN,0),(LEN,WID),(0,WID)]])
mask=np.zeros((490,1920),np.uint8); cv2.fillPoly(mask,[poly],255)
mask=cv2.erode(mask,np.ones((9,9),np.uint8))        # shrink: keep spectators out

def bumper(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    H,S,V=hsv[...,0].astype(int),hsv[...,1].astype(int),hsv[...,2].astype(int)
    red =(((H<=9)|(H>=168))&(S>110)&(V>55))
    blue=((H>=100)&(H<=130)&(S>110)&(V>45))
    return red.astype(np.uint8),blue.astype(np.uint8)

pr,pb=bumper(plate)
static=cv2.dilate(((pr|pb)*255).astype(np.uint8),np.ones((7,7),np.uint8))  # goals/ramps
bg=plate.astype(np.int16)

def grab(ts):
    out=[];c=av.open("data/match1_qual.mp4");s=c.streams.video[0]
    s.codec_context.options={"hwaccel":"videotoolbox"}
    for t in ts:
        c.seek(int(t/s.time_base),stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time>=t-0.05:
                out.append((t,f.to_ndarray(format="rgb24")[0:490])); break
    c.close(); return out

print("  t(s)  red  blue  total   (3 red + 3 blue expected)")
for t,fr in grab([12.,18.,40.,70.,100.,140.]):
    diff=(np.abs(fr.astype(np.int16)-bg).sum(axis=2)>55).astype(np.uint8)*255
    r,b=bumper(fr)
    im=cv2.cvtColor(fr,cv2.COLOR_RGB2BGR); nr=nb=0
    for col,cm,name,bgr in ((0,r,'R',(0,0,255)),(1,b,'B',(255,150,0))):
        m=cv2.bitwise_and(cm*255,diff)
        m=cv2.bitwise_and(m,mask)
        m=cv2.bitwise_and(m,cv2.bitwise_not(static))   # drop static red/blue field parts
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
        for i in range(1,n):
            x,y,w,h,a=st[i]
            if a<60 or w<10 or w>120 or h<6 or h>70: continue
            if name=='R': nr+=1
            else: nb+=1
            cv2.rectangle(im,(x,y),(x+w,y+h),bgr,2)
            cv2.circle(im,(x+w//2,y+h),4,(255,255,255),-1)
            cv2.putText(im,f"{name}{w}x{h}",(x,y-3),cv2.FONT_HERSHEY_SIMPLEX,.4,bgr,1,cv2.LINE_AA)
    cv2.polylines(im,[poly],True,(0,255,0),1)
    cv2.putText(im,f"t={t:.0f}s  R={nr} B={nb}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,.6,
                (255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(f"out/p2_t{int(t)}.jpg",im,[cv2.IMWRITE_JPEG_QUALITY,90])
    print(f"  {t:5.0f} {nr:5d} {nb:5d} {nr+nb:6d}")
print("wrote out/p2_t*.jpg")
