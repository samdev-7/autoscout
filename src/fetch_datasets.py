"""Download the external training datasets. Never emit the API key."""
import io, os, contextlib, sys
KEY=open(".roboflow_key").read().strip()
def clean(s): return s.replace(KEY,"<KEY>") if KEY else s

from roboflow import Roboflow
buf=io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rf=Roboflow(api_key=KEY)
    out=[]
    for ws,proj in (("frcrobots","frc-robots-1zts0"),
                    ("frc-9wt6x","jerrywu_frc114_robot_v2-przxh")):
        try:
            p=rf.workspace(ws).project(proj)
            vs=[v.version for v in p.versions()]
            out.append((ws,proj,vs,None))
        except Exception as e:
            out.append((ws,proj,None,repr(e)))
print(clean(buf.getvalue())[-1500:])
print("=== PROJECT VERSIONS ===")
for ws,proj,vs,err in out:
    print(f"  {proj:38} versions={vs}  {clean(err) if err else ''}")
