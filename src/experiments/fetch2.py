import io, os, contextlib, glob
KEY=open(".roboflow_key").read().strip()
clean=lambda s: s.replace(KEY,"<KEY>") if KEY else s
from roboflow import Roboflow
JOBS=[("frcrobots","frc-robots-1zts0",5,"data/rf/frcrobots_v5"),
      ("frc-9wt6x","jerrywu_frc114_robot_v2-przxh",2,"data/rf/jerrywu_v2"),
      ("frc-9wt6x","jerrywu_frc114_robot_v2-przxh",1,"data/rf/jerrywu_v1")]
os.makedirs("data/rf",exist_ok=True)
buf=io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rf=Roboflow(api_key=KEY)
    for ws,proj,ver,loc in JOBS:
        if os.path.isdir(loc): continue
        try: rf.workspace(ws).project(proj).version(ver).download("yolov8",location=loc)
        except Exception as e: print("ERR",repr(e))
tail=clean(buf.getvalue())
print(tail[-600:] if "ERR" in tail else "(sdk output suppressed)")
print("\n=== DOWNLOADED ===")
for _,_,ver,loc in JOBS:
    if not os.path.isdir(loc): print(f"  {loc}: MISSING"); continue
    n=len(glob.glob(f"{loc}/**/images/*.*",recursive=True))
    lb=len(glob.glob(f"{loc}/**/labels/*.txt",recursive=True))
    sz=sum(os.path.getsize(p) for p in glob.glob(f"{loc}/**/*",recursive=True) if os.path.isfile(p))
    splits=[d for d in ("train","valid","test") if os.path.isdir(f"{loc}/{d}")]
    print(f"  {loc:28} {n:5d} imgs  {lb:5d} lbl  {sz/2**20:7.1f} MiB  splits={splits}")
    y=f"{loc}/data.yaml"
    if os.path.exists(y):
        for line in open(y):
            if line.startswith(("nc:","names:")): print(f"      {line.strip()}")
