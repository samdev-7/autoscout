import av, numpy as np, cv2
c=av.open("data/match1_qual.mp4"); s=c.streams.video[0]
s.codec_context.options={"hwaccel":"videotoolbox"}
c.seek(int(100/s.time_base), stream=s)
for f in c.decode(s):
    if f.time and f.time>=99.95: a=f.to_ndarray(format="rgb24"); break
c.close()
top=a[0:490]
big=cv2.resize(top,None,fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
for (cid,x,y) in [(2,616.5,210.9),(11,583.2,211.0),(21,1303.9,204.5)]:
    p=(int(x*2),int(y*2))
    cv2.circle(big,p,26,(0,255,255),3)
    cv2.putText(big,f"id{cid}",(p[0]-20,p[1]-34),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),3)
cv2.line(big,(int(583*2),int(211*2)),(int(1303.9*2),int(204.5*2)),(0,255,255),2)
cv2.imwrite("out/top_panel_2x.jpg",cv2.cvtColor(big,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92])
print("wrote out/top_panel_2x.jpg", big.shape)
