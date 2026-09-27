import io, contextlib
KEY=open(".roboflow_key").read().strip()
clean=lambda s: s.replace(KEY,"<KEY>") if KEY else s
from roboflow import Roboflow
PROJ=[("adam-k","frc-robots-lb58n"),
      ("frc4419","frc-robots-equcn"),
      ("aaahhhh-coral-marx","frc-robots-2"),
      ("6036","6036-ai-htnmi"),
      ("grain-dzomt","scouting-cqm98")]
buf=io.StringIO(); res=[]
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rf=Roboflow(api_key=KEY)
    for ws,pr in PROJ:
        try:
            p=rf.workspace(ws).project(pr)
            vs=sorted([int(v.version) for v in p.versions()])
            res.append(dict(ws=ws,pr=pr,vs=vs,cls=p.classes,n=p.__dict__.get("images"),
                            lic=p.__dict__.get("license"),err=None))
        except Exception as e:
            res.append(dict(ws=ws,pr=pr,err=repr(e)))
print(f"{'project':34}{'imgs':>7}  {'versions':<14}{'license':<16}classes")
for r in res:
    if r.get("err"): print(f"{r['pr']:34}  ERROR {clean(r['err'])[:70]}"); continue
    cls=r.get("cls") or {}
    cn=", ".join(f"{k}({v})" for k,v in list(cls.items())[:5]) if isinstance(cls,dict) else str(cls)[:50]
    print(f"{r['ws']+'/'+r['pr']:34}{str(r.get('n')):>7}  {str(r['vs']):<14}{str(r.get('lic')):<16}{cn}")
