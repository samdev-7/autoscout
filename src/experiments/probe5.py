import io, os, contextlib, glob
KEY=open(".roboflow_key").read().strip()
clean=lambda s: s.replace(KEY,"<KEY>") if KEY else s
from roboflow import Roboflow
buf=io.StringIO(); out=[]
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rf=Roboflow(api_key=KEY)
    try:
        p=rf.workspace("alis-workspace-c17hf").project("frc-scouting-tqul1")
        vs=sorted(int(v.version) for v in p.versions())
        out.append((p.classes,p.__dict__.get("images"),vs,None))
        if vs and not os.path.isdir("data/rf/alis"):
            rf.workspace("alis-workspace-c17hf").project("frc-scouting-tqul1")\
              .version(vs[-1]).download("yolov8",location="data/rf/alis")
    except Exception as e: out.append((None,None,None,repr(e)))
for cls,n,vs,err in out:
    if err: print("ERROR",clean(err)[:120])
    else:
        print(f"frc-scouting-tqul1  imgs={n}  versions={vs}")
        for k,v in sorted((cls or {}).items(),key=lambda kv:-kv[1]): print(f"    {k:26} {v}")
if os.path.isdir("data/rf/alis"):
    y=open("data/rf/alis/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    print(f"  downloaded: {len(glob.glob('data/rf/alis/*/images/*.*'))} imgs  classes={names}")
