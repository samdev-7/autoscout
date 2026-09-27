"""How much field-position error does ignoring distortion actually cause?"""
import numpy as np, cv2
IN=0.0254; LEN,WID=16.541,8.069
p=np.load("out/cam_params.npy"); nk=1
f,cx,cy=p[0],p[1],p[2]; rv=p[3:6]; tv=p[6:9]; k1=p[9]
K=np.array([[f,0,cx],[0,f,cy],[0,0,1]]); d=np.zeros(5); d[0]=k1
R,_=cv2.Rodrigues(rv); C=-R.T@tv
H=np.load("out/H_field_to_img.npy"); Hi=np.linalg.inv(H)

def field_to_img(X,Y):
    q,_=cv2.projectPoints(np.array([[[X,Y,0.0]]]),rv,tv,K,d)
    return q.reshape(2)
def img_to_field_full(u,v):
    n=cv2.undistortPoints(np.array([[[u,v]]],float),K,d).reshape(2)
    ray=R.T@np.array([n[0],n[1],1.0]); t=-C[2]/ray[2]
    return C[:2]+t*ray[:2]

print("field-position error from ignoring lens distortion")
print(f"{'field X,Y (m)':>16} {'image u,v':>16} {'homography XY':>18} {'error in':>9}")
worst=0
for X in (1.0,4.0,8.27,12.5,15.5):
    for Y in (1.0,4.0,7.0):
        u,v=field_to_img(X,Y)
        if not(0<u<1920 and 0<v<490): continue
        fh=(Hi@np.array([u,v,1.0])); fh=fh[:2]/fh[2]
        err=np.linalg.norm(fh-np.array([X,Y]))/IN
        worst=max(worst,err)
        print(f"{X:7.2f},{Y:6.2f} {u:9.1f},{v:6.1f} {fh[0]:9.2f},{fh[1]:7.2f} {err:9.1f}")
print(f"\nworst-case field error if distortion is ignored: {worst:.1f} in ({worst/12:.2f} ft)")
print(f"camera: X {C[0]:.2f} Y {C[1]:.2f} Z {C[2]:.2f} m   focal {f:.0f}px   k1 {k1:+.4f}")

# redraw the grid using the full model
import av
c=av.open("data/match1_qual.mp4"); s=c.streams.video[0]
s.codec_context.options={"hwaccel":"videotoolbox"}
c.seek(int(95/s.time_base),stream=s)
for fr in c.decode(s):
    if fr.time is not None and fr.time>=94.95: im=fr.to_ndarray(format="rgb24")[0:490]; break
c.close()
im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
for X in np.arange(0,LEN+1e-6,1.0):
    pl=[field_to_img(min(X,LEN),y) for y in np.linspace(0,WID,40)]
    cv2.polylines(im,[np.int32(pl)],False,(255,220,0),1,cv2.LINE_AA)
for Y in np.arange(0,WID+1e-6,1.0):
    pl=[field_to_img(x,min(Y,WID)) for x in np.linspace(0,LEN,80)]
    cv2.polylines(im,[np.int32(pl)],False,(255,220,0),1,cv2.LINE_AA)
bd=[field_to_img(*a) for a in [(0,0),(LEN,0),(LEN,WID),(0,WID)]]
cv2.polylines(im,[np.int32(bd)],True,(0,80,255),3,cv2.LINE_AA)
cv2.putText(im,"full camera model: K + R,t + k1  (distortion corrected)",
            (12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
cv2.imwrite("out/overlay_full.jpg",im,[cv2.IMWRITE_JPEG_QUALITY,93])
print("wrote out/overlay_full.jpg")
