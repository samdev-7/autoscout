"""Arms A (curated broadcast) and B (A + jw), with held-out val clips removed."""
import glob,os,re,json,shutil,collections
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
rf=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$",re.I)
HOLD=set(json.load(open("out/heldout_clips.json")))
def clip(fn):
    t=rf.sub("",fn); t=re.sub(r"\.(jpg|png|jpeg)$","",t,flags=re.I)
    return re.sub(r"[-_]?\d{2,6}$","",t).lower()
def collect(specs):
    out=[]; drop=0
    for loc,tag in specs:
        y=open(f"{loc}/data.yaml").read()
        names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
        keep={i for i,n in enumerate(names) if n in ROBOT}
        seen=set()
        for im in sorted(glob.glob(f"{loc}/*/images/*.*")):
            st=rf.sub("",os.path.basename(im))
            if st in seen: continue
            if clip(os.path.basename(im)) in HOLD: drop+=1; continue   # never train on val clips
            lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
            if not os.path.exists(lb): continue
            rows=[l.split() for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep]
            if not rows: continue
            seen.add(st); out.append((im,rows,tag))
    return out,drop
BC=[("data/rf/frcrobots_v5","frc"),("data/rf/t6036","t6036"),("data/rf/grain","grain"),
    ("data/rf/frc4419","f4419"),("data/rf/capstone","capstone"),("data/rf/scout_v7","scout_v7")]
JW=[("data/rf/jerrywu_v2","jw")]
def write(items,root):
    shutil.rmtree(root,ignore_errors=True)
    os.makedirs(f"{root}/images/train"); os.makedirs(f"{root}/labels/train")
    per=collections.Counter(); nb=0
    for i,(im,rows,tag) in enumerate(items,1):
        per[tag]+=1; nb+=len(rows); o=f"{tag}_{i:05d}"
        shutil.copy(im,f"{root}/images/train/{o}{os.path.splitext(im)[1]}")
        open(f"{root}/labels/train/{o}.txt","w").write(
            "".join("0 "+" ".join(r[1:5])+"\n" for r in rows))
    open(f"{root}/data.yaml","w").write(
      f"path: {os.path.abspath(root)}\ntrain: images/train\n"
      f"val: {os.path.abspath('out/yolo_val_full')}/images/val\nnc: 1\nnames: [robot]\n")
    return len(items),nb,dict(per)
b,d1=collect(BC); j,_=collect(JW)
for nm,items,root in (("ARM A curated broadcast",b,"out/armA"),
                      ("ARM B  A + all jw",b+j,"out/armB")):
    n,nb,per=write(items,root)
    print(f"{nm:26} {n:5d} imgs {nb:6d} boxes  {(n+15)//16:4d} it/ep  {per}")
print(f"held-out frames excluded from training: {d1}")
