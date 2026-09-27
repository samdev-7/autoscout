"""Temporal median of the TOP panel over the play window.

Static camera => median across time removes every moving object (robots, balls,
refs) and averages away sensor/compression noise. Result: a clean background
plate. Two uses: (1) far better AprilTag detection, (2) the reference image for
background-subtraction robot detection later.
"""
import av, numpy as np, cv2, time

PATH="data/match1_qual.mp4"; T0,T1=7.1,172.8; STRIDE=5
Y0,Y1=0,490

c=av.open(PATH); s=c.streams.video[0]
s.codec_context.options={"hwaccel":"videotoolbox"}; s.thread_type="AUTO"
buf=[]; t0=time.time()
for i,f in enumerate(c.decode(s)):
    if f.time is None or not (T0<=f.time<=T1) or i%STRIDE: continue
    buf.append(f.to_ndarray(format="rgb24")[Y0:Y1])
c.close()
stack=np.stack(buf); del buf
print(f"stacked {stack.shape[0]} frames {stack.shape[1:]} in {time.time()-t0:.1f}s "
      f"({stack.nbytes/2**30:.2f} GiB)")

t0=time.time()
med=np.median(stack,axis=0).astype(np.uint8)
print(f"median in {time.time()-t0:.1f}s")

# how much moving content did the median remove?
mad=np.abs(stack.astype(np.int16)-med).mean(axis=(1,2,3))
print(f"mean |frame - median| : {mad.mean():.1f}  (higher => more motion removed)")

cv2.imwrite("out/top_plate.png", cv2.cvtColor(med,cv2.COLOR_RGB2BGR))
np.save("out/top_plate.npy", med)
print("wrote out/top_plate.png")
