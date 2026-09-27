"""Auto-propose robot boxes: background subtraction + field mask + bumper colour.

Purpose is bootstrapping training labels, not final detection. The field mask
comes from the calibration, which kills crowd/table/floor clutter for free.
"""
import av, numpy as np, cv2, json
IN=0.0254; LEN,WID=16.541,8.069
p=np.load("out/cam_params.npy")
K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]); d=np.zeros(5); d[0]=p[9]
rv,tv=p[3:6],p[6:9]
def f2i(X,Y,Z=0.0):
    q,_=cv2.projectPoints(np.array([[[X,Y,Z]]]),rv,tv,K,d); return q.reshape(2)

plate=np.load("out/top_plate.npy")
# field polygon, dilated a little so robots on the boundary survive
poly=np.int32([f2i(x,y) for x,y in
      [(0,0),(LEN,0),(LEN,WID),(0,WID)]])
mask=np.zeros((490,1920),np.uint8); cv2.fillPoly(mask,[poly],255)
mask=cv2.dilate(mask,np.ones((25,25),np.uint8))
print(f"field mask covers {mask.mean()/255*100:.1f}% of the panel")

def grab(ts):
    out=[];c=av.open("data/match1_qual.mp4");s=c.streams.video[0]
    s.codec_context.options={"hwaccel":"videotoolbox"}
    for t in ts:
        c.seek(int(t/s.time_base),stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time>=t-0.05:
                out.append((t,f.to_ndarray(format="rgb24")[0:490])); break
    c.close(); return out

bg=plate.astype(np.int16)
TS=[12.0,18.0,40.0,70.0,100.0,140.0]
tiles=[];summary=[]
for t,fr in grab(TS):
    diff=np.abs(fr.astype(np.int16)-bg).sum(axis=2).astype(np.uint8)
    fg=((diff>60)*255).astype(np.uint8)
    fg=cv2.bitwise_and(fg,mask)
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(fg,8)
    hsv=cv2.cvtColor(fr,cv2.COLOR_RGB2HSV)
    im=cv2.cvtColor(fr,cv2.COLOR_RGB2BGR); keep=0
    for i in range(1,n):
        x,y,w,h,a=st[i]
        if not(14<=w<=110 and 12<=h<=95 and a>220): continue
        sub=(lab[y:y+h,x:x+w]==i)
        H=hsv[y:y+h,x:x+w,0][sub]; S=hsv[y:y+h,x:x+w,1][sub]; V=hsv[y:y+h,x:x+w,2][sub]
        sat=S>90
        yellow=((H>20)&(H<38)&sat).mean()
        red   =(((H<10)|(H>170))&sat&(V>60)).mean()
        blue  =((H>100)&(H<135)&sat&(V>50)).mean()
        if yellow>0.35: continue                    # a pile of game pieces
        col=(0,0,255) if red>blue else (255,140,0)
        cls='R' if red>blue else 'B'
        if max(red,blue)<0.02: col,cls=(0,255,255),'?'
        keep+=1
        cv2.rectangle(im,(x,y),(x+w,y+h),col,2)
        cv2.circle(im,(x+w//2,y+h),4,(255,255,255),-1)      # floor-contact point
        f=f2i(0,0)  # placeholder to keep import used
        cv2.putText(im,f"{cls}{w}x{h}",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,.42,col,1,cv2.LINE_AA)
    cv2.polylines(im,[poly],True,(0,255,0),1)
    cv2.putText(im,f"t={t:.0f}s  proposals={keep}",(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
    summary.append((t,keep))
    tiles.append(im[150:470])
    cv2.imwrite(f"out/prop_t{int(t)}.jpg",im,[cv2.IMWRITE_JPEG_QUALITY,90])
print("\n  t(s)  proposals   (6 robots expected)")
for t,k in summary: print(f"  {t:5.0f} {k:10d}")
cv2.imwrite("out/proposals.jpg",np.vstack(tiles[:3]),[cv2.IMWRITE_JPEG_QUALITY,88])
print("\nwrote out/proposals.jpg + per-frame files")
