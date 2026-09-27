"""Per-SOURCE-VIDEO quality audit across every dataset, including rejected ones.

Dataset-level verdicts are too coarse: a dataset can hold good clips and bad ones.
Image dimensions are read lazily (no pixel decode) so this stays cheap.
"""
import glob,os,re,json,collections,numpy as np
from PIL import Image
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
rf=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$",re.I)
DS={"frc":"data/rf/frcrobots_v5","jw":"data/rf/jerrywu_v2","t6036":"data/rf/t6036",
    "grain":"data/rf/grain","f4419":"data/rf/frc4419","capstone":"data/rf/capstone",
    "scout_v7":"data/rf/scout_v7","scout2025":"data/rf/scout2025",
    "pack1294":"data/rf/pack1294","alis":"data/rf/alis","adamk":"data/rf/adamk"}
def source_of(fn):
    s=rf.sub("",fn); s=re.sub(r"\.(jpg|png|jpeg)$","",s,flags=re.I)
    s=re.sub(r"[-_]?\d{2,6}$","",s); s=re.sub(r"[-_]?(mp4|mkv|webm)$","",s,flags=re.I)
    return s.strip("-_ ").lower() or "(none)"
rec=collections.defaultdict(lambda: dict(f=0,b=0,area=[],asp=[],empty=0,ex=[]))
for tag,loc in DS.items():
    if not os.path.isdir(loc): continue
    y=open(f"{loc}/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    keep={i for i,n in enumerate(names) if n in ROBOT}
    seen=set()
    for im in sorted(glob.glob(f"{loc}/*/images/*.*")):
        stem=rf.sub("",os.path.basename(im))
        if stem in seen: continue                       # skip roboflow aug copies
        seen.add(stem)
        lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lb): continue
        try: W,H=Image.open(im).size
        except Exception: continue
        rows=[l.split() for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep]
        k=(tag,source_of(os.path.basename(im))); r=rec[k]
        if not rows: r["empty"]+=1; continue
        r["f"]+=1; r["b"]+=len(rows)
        if len(r["ex"])<8: r["ex"].append(im)
        for q in rows:
            bw,bh=float(q[3]),float(q[4])
            r["area"].append(bw*bh*100)
            if bh>0: r["asp"].append((bw*W)/(bh*H))
out=[]
for (tag,src),r in rec.items():
    if r["f"]<3: continue
    a=np.array(r["area"]); s=np.array(r["asp"])
    out.append(dict(ds=tag,src=src,frames=r["f"],boxes=r["b"],bpf=r["b"]/r["f"],
        area=float(np.median(a)), big=float((a>12).mean()), tiny=float((a<0.02).mean()),
        asp=float(np.median(s)), aspwide=float((s>3).mean()), asptall=float((s<0.45).mean()),
        empty=r["empty"]/max(r["f"]+r["empty"],1), ex=r["ex"]))
json.dump(out,open("out/source_audit.json","w"))
print(f"{len(out)} source clips with >=3 frames, across {len(DS)} datasets\n")
def flags(o):
    f=[]
    if o["bpf"]<2.2: f.append("under-labelled")
    if o["bpf"]>9: f.append("many-boxes")
    if o["big"]>0.25: f.append(f"oversized {100*o['big']:.0f}%")
    if o["tiny"]>0.35: f.append(f"tiny {100*o['tiny']:.0f}%")
    if o["aspwide"]>0.25: f.append("wide-boxes")
    if o["asptall"]>0.30: f.append("tall-boxes")
    if o["empty"]>0.45: f.append(f"empty {100*o['empty']:.0f}%")
    return f
for o in out: o["flags"]=flags(o)
bad=[o for o in out if o["flags"]]
print(f"clips with at least one flag: {len(bad)}/{len(out)}\n")
per=collections.Counter()
for o in out: per[o["ds"]]+=1
perbad=collections.Counter(o["ds"] for o in bad)
print(f"{'dataset':10}{'clips':>7}{'flagged':>9}{'clean':>7}{'clean frames':>14}")
for d in DS:
    cf=sum(o["frames"] for o in out if o["ds"]==d and not o["flags"])
    print(f"{d:10}{per[d]:7d}{perbad[d]:9d}{per[d]-perbad[d]:7d}{cf:14d}")
print("\nworst-flagged clips (sample):")
for o in sorted(bad,key=lambda x:-x["frames"])[:14]:
    print(f"  {o['ds']:9} {o['src'][:30]:30} f={o['frames']:4d} bpf={o['bpf']:5.2f} "
          f"area={o['area']:6.2f}% asp={o['asp']:5.2f}  {','.join(o['flags'])}")
