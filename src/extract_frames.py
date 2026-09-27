"""Sample frames sparsely across the match for labelling.

Every 2 s, not every frame: consecutive frames are near-duplicates and add
labelling effort without adding information (and would corrupt any train/val
split later).
"""
import av, numpy as np, cv2, json, os
LEN,WID=16.541,8.069
p=np.load("out/cam_params.npy")
K=np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]]); d=np.zeros(5); d[0]=p[9]
rv,tv=p[3:6],p[6:9]
def f2i(X,Y,Z=0.0):
    q,_=cv2.projectPoints(np.array([[[X,Y,Z]]]),rv,tv,K,d); return q.reshape(2)

# vertical band the field occupies, padded for robot height
pts=[f2i(x,y) for x in np.linspace(0,LEN,40) for y in (0,WID)]
pts+=[f2i(x,y,1.2) for x in np.linspace(0,LEN,40) for y in (0,WID)]
ys=[q[1] for q in pts]
Y0=max(0,int(min(ys))-25); Y1=min(490,int(max(ys))+25)
print(f"field band: y {Y0}..{Y1}  ({Y1-Y0}px tall)")

os.makedirs("out/frames",exist_ok=True)
for f in os.listdir("out/frames"): os.remove(f"out/frames/{f}")
T0,T1,STEP=8.0,172.0,2.0
times=np.arange(T0,T1+1e-6,STEP)
c=av.open("data/match1_qual.mp4"); s=c.streams.video[0]
s.codec_context.options={"hwaccel":"videotoolbox"}; s.thread_type="AUTO"
want=list(times); man=[]; i=0
for fr in c.decode(s):
    if fr.time is None or i>=len(want): continue
    if fr.time+1e-6 < want[i]: continue
    a=fr.to_ndarray(format="rgb24")[0:490]
    name=f"f{i:03d}.jpg"
    cv2.imwrite(f"out/frames/{name}",cv2.cvtColor(a,cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY,92])
    man.append({"file":name,"t":round(float(fr.time),2)}); i+=1
c.close()
json.dump({"band":[Y0,Y1],"w":1920,"h":490,"frames":man},open("out/frames.json","w"))
sz=sum(os.path.getsize(f"out/frames/{m['file']}") for m in man)/2**20
print(f"wrote {len(man)} frames ({sz:.1f} MiB) spanning t={man[0]['t']}..{man[-1]['t']}s")
