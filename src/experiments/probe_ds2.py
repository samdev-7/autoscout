import io, contextlib
KEY=open(".roboflow_key").read().strip()
clean=lambda s: s.replace(KEY,"<KEY>") if KEY else s
from roboflow import Roboflow
PROJ=[("-gw5oi","scouting-v7cfq"),("aj-oj2mu","scouting-2025"),
      ("pack-of-scouts","1294-ai-scouting"),("capstone-roipa","frc-automatic-scouting")]
buf=io.StringIO(); res=[]
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    rf=Roboflow(api_key=KEY)
    for ws,pr in PROJ:
        try:
            p=rf.workspace(ws).project(pr)
            res.append(dict(pr=pr,vs=sorted(int(v.version) for v in p.versions()),
                            cls=p.classes,n=p.__dict__.get("images"),err=None))
        except Exception as e: res.append(dict(pr=pr,err=repr(e)))
for r in res:
    if r.get("err"): print(f"{r['pr']:28} ERROR {clean(r['err'])[:80]}"); continue
    print(f"{r['pr']:28} imgs={str(r['n']):>6}  versions={r['vs']}")
    cls=r.get('cls') or {}
    for k,v in sorted(cls.items(),key=lambda kv:-kv[1]): print(f"      {k:28} {v}")
