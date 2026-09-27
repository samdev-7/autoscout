"""Hold out whole SOURCE CLIPS from scout_v7 + t6036 as a multi-event val set.

Our 83-frame val is one match/camera/event. Selecting models on it alone risks
picking whatever suits this arena. These two sources are ~1 frame per match, so
holding out N clips buys N different events/cameras for the cost of N images.
"""
import glob,os,re,json,random,shutil,collections
random.seed(17)
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
rf=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$",re.I)
def clip(fn):
    t=rf.sub("",fn); t=re.sub(r"\.(jpg|png|jpeg)$","",t,flags=re.I)
    return re.sub(r"[-_]?\d{2,6}$","",t).lower()
def load(loc):
    y=open(f"{loc}/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    keep={i for i,n in enumerate(names) if n in ROBOT}
    seen=set(); out=[]
    for im in sorted(glob.glob(f"{loc}/*/images/*.*")):
        st=rf.sub("",os.path.basename(im))
        if st in seen: continue
        lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lb): continue
        rows=[l.split() for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep]
        if not rows: continue
        seen.add(st); out.append((im,rows,clip(os.path.basename(im))))
    return out
sv=load("data/rf/scout_v7"); t6=load("data/rf/t6036")
cs=sorted({c for _,_,c in sv}); ct=sorted({c for _,_,c in t6})
hold=set(random.sample(cs,50))|set(random.sample(ct,30))
print(f"scout_v7 {len(sv)} frames / {len(cs)} clips ; t6036 {len(t6)} frames / {len(ct)} clips")
root="out/val_multi"; shutil.rmtree(root,ignore_errors=True)
os.makedirs(f"{root}/images/val"); os.makedirs(f"{root}/labels/val")
n=nb=0; src=collections.Counter()
for im,rows,c in sv+t6:
    if c not in hold: continue
    n+=1; nb+=len(rows); src[c]+=1; o=f"mv_{n:04d}"
    shutil.copy(im,f"{root}/images/val/{o}{os.path.splitext(im)[1]}")
    open(f"{root}/labels/val/{o}.txt","w").write(
        "".join("0 "+" ".join(r[1:5])+"\n" for r in rows))
open(f"{root}/data.yaml","w").write(
  f"path: {os.path.abspath(root)}\ntrain: images/val\nval: images/val\nnc: 1\nnames: [robot]\n")
json.dump(sorted(hold),open("out/heldout_clips.json","w"))
print(f"MULTI-EVENT VAL: {n} frames, {nb} boxes, {len(src)} distinct events/cameras")
print(f"   frames per clip: {min(src.values())}-{max(src.values())}")
