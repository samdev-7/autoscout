import io, os, contextlib, glob
KEY=open(".roboflow_key").read().strip()
clean=lambda s: s.replace(KEY,"<KEY>") if KEY else s
from roboflow import Roboflow
JOBS=[("adam-k","frc-robots-lb58n",1,"data/rf/adamk"),
      ("frc4419","frc-robots-equcn",3,"data/rf/frc4419"),
      ("aaahhhh-coral-marx","frc-robots-2",1,"data/rf/coralmarx"),
      ("6036","6036-ai-htnmi",2,"data/rf/t6036"),
      ("grain-dzomt","scouting-cqm98",7,"data/rf/grain")]
buf=io.StringIO(); errs=[]
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rf=Roboflow(api_key=KEY)
    for ws,pr,ver,loc in JOBS:
        if os.path.isdir(loc): continue
        try: rf.workspace(ws).project(pr).version(ver).download("yolov8",location=loc)
        except Exception as e: errs.append((loc,repr(e)))
for loc,e in errs: print(f"  DOWNLOAD FAILED {loc}: {clean(e)[:120]}")
print(f"\n{'dataset':22}{'imgs':>7}{'labels':>8}{'boxes':>8}{'box/img':>9}  classes / sample size")
import collections
for _,_,_,loc in JOBS:
    if not os.path.isdir(loc): print(f"  {loc:20} -- not downloaded"); continue
    imgs=glob.glob(f"{loc}/*/images/*.*"); lbs=glob.glob(f"{loc}/*/labels/*.txt")
    nb=0; cls=collections.Counter()
    for l in lbs:
        for line in open(l):
            p=line.split()
            if len(p)>=5: nb+=1; cls[p[0]]+=1
    names=[]
    y=f"{loc}/data.yaml"
    if os.path.exists(y):
        t=open(y).read()
        if "names:" in t: names=t.split("names:")[1].split("\n")[0:6]
    import cv2
    sz=""
    if imgs:
        im=cv2.imread(imgs[0]); sz=f"{im.shape[1]}x{im.shape[0]}" if im is not None else "?"
    print(f"  {os.path.basename(loc):20}{len(imgs):7d}{len(lbs):8d}{nb:8d}{nb/max(len(imgs),1):9.2f}  "
          f"{dict(cls)} img0={sz}")
