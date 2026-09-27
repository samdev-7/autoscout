"""Build arm A (broadcast only) and arm B (broadcast + jw). Single class 'robot'."""
import glob,os,re,shutil,collections
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
rf=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$",re.I)
BCAST=[("data/rf/frcrobots_v5","frc"),("data/rf/t6036","t6036"),("data/rf/grain","grain"),
       ("data/rf/frc4419","f4419"),("data/rf/capstone","capstone"),("data/rf/scout_v7","scout_v7")]
JW=[("data/rf/jerrywu_v2","jw")]
def collect(specs):
    out=[]
    for loc,tag in specs:
        y=open(f"{loc}/data.yaml").read()
        names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
        keep={i for i,n in enumerate(names) if n in ROBOT}
        seen=set()
        for im in sorted(glob.glob(f"{loc}/*/images/*.*")):
            st=rf.sub("",os.path.basename(im))
            if st in seen: continue
            lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
            if not os.path.exists(lb): continue
            rows=[l.split() for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep]
            if not rows: continue
            seen.add(st); out.append((im,rows,tag))
    return out
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
b=collect(BCAST); j=collect(JW)
for name,items,root in (("ARM A (broadcast only)",b,"out/arm_A"),
                        ("ARM B (broadcast + jw)",b+j,"out/arm_B")):
    n,nb,per=write(items,root)
    it=(n+15)//16
    print(f"{name:26} {n:5d} imgs  {nb:6d} boxes  {it:4d} it/epoch   {per}")
