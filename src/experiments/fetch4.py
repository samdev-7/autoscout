import io, os, contextlib, glob
KEY=open(".roboflow_key").read().strip()
clean=lambda s: s.replace(KEY,"<KEY>") if KEY else s
from roboflow import Roboflow
JOBS=[("-gw5oi","scouting-v7cfq",9,"data/rf/scout_v7"),
      ("aj-oj2mu","scouting-2025",1,"data/rf/scout2025"),
      ("pack-of-scouts","1294-ai-scouting",14,"data/rf/pack1294"),
      ("capstone-roipa","frc-automatic-scouting",30,"data/rf/capstone")]
buf=io.StringIO(); errs=[]
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rf=Roboflow(api_key=KEY)
    for ws,pr,ver,loc in JOBS:
        if os.path.isdir(loc): continue
        try: rf.workspace(ws).project(pr).version(ver).download("yolov8",location=loc)
        except Exception as e: errs.append((loc,repr(e)))
for loc,e in errs: print(f"  FAILED {loc}: {clean(e)[:110]}")
for _,_,_,loc in JOBS:
    if not os.path.isdir(loc): continue
    y=open(f"{loc}/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    n=len(glob.glob(f"{loc}/*/images/*.*"))
    print(f"  {os.path.basename(loc):12} {n:5d} imgs  classes={names}")
