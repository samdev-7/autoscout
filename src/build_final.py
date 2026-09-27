"""Final training pool: normalize classes, drop non-robots, dedupe perceptually."""
import glob,os,re,shutil,cv2,numpy as np,collections,json
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
# rejected: alis (under-labelled+close-up), adamk (mosaic aug),
#           scout2025 & pack1294 (bad boxes, confirmed by inspection)
SRC=[("data/rf/frcrobots_v5","frc"),("data/rf/jerrywu_v2","jw"),
     ("data/rf/t6036","t6036"),("data/rf/grain","grain"),("data/rf/frc4419","f4419"),
     ("data/rf/capstone","capstone"),("data/rf/scout_v7","scout_v7")]
stem_re=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$")

# --- resolve the scout_v7 convention question: bumper-only or whole robot? ---
def pixel_aspect(loc,keep):
    a=[]
    for im in glob.glob(f"{loc}/*/images/*.*")[:400]:
        lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lb): continue
        img=cv2.imread(im)
        if img is None: continue
        H,W=img.shape[:2]
        for l in open(lb):
            p=l.split()
            if len(p)>=5 and int(p[0]) in keep:
                w,h=float(p[3])*W,float(p[4])*H
                if h>0: a.append(w/h)
    return np.median(a) if a else 0
y=open("data/rf/scout_v7/data.yaml").read()
nm=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
print(f"scout_v7 pixel aspect = {pixel_aspect('data/rf/scout_v7',{i for i,n in enumerate(nm) if n in ROBOT}):.2f}"
      f"   (ours 1.47; a bumper-only box would be 3-5)")

items=[]; drop_cls=collections.Counter(); empty=collections.Counter()
for loc,tag in SRC:
    y=open(f"{loc}/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    keep={i for i,n in enumerate(names) if n in ROBOT}
    for im in sorted(glob.glob(f"{loc}/*/images/*.*")):
        lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lb): continue
        rows=[l.split() for l in open(lb) if len(l.split())>=5]
        drop_cls[tag]+=sum(1 for r in rows if int(r[0]) not in keep)
        rows=[r for r in rows if int(r[0]) in keep]
        if not rows: empty[tag]+=1; continue
        items.append((im,rows,tag))
print(f"\nnon-robot boxes discarded: {dict(drop_cls)}")
print(f"images dropped (no robot after filter): {dict(empty)}")

def dhash(p,s=8):
    im=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    r=cv2.resize(im,(s+1,s),interpolation=cv2.INTER_AREA)
    return np.packbits((r[:,1:]>r[:,:-1]).flatten()).view(np.uint64)[0]
H=np.zeros(len(items),np.uint64); ok=np.ones(len(items),bool)
for i,(im,_,_) in enumerate(items):
    h=dhash(im)
    if h is None: ok[i]=False
    else: H[i]=h
print(f"\nhashing {len(items)} images -> full pairwise dedupe (threshold<=5, "
      f"unrelated floor measured at 18)")
keepmask=np.ones(len(items),bool); CH=512
for s in range(0,len(H),CH):
    e=min(s+CH,len(H))
    d=np.bitwise_count(H[s:e,None]^H[None,:])
    for r in range(e-s):
        i=s+r
        if not keepmask[i] or not ok[i]: continue
        j=np.where((d[r]<=5)&(np.arange(len(H))<i))[0]
        if any(keepmask[jj] for jj in j): keepmask[i]=False
print(f"  near-duplicates removed: {(~keepmask).sum()}  -> {keepmask.sum()} unique")

root="out/yolo_train3"; shutil.rmtree(root,ignore_errors=True)
os.makedirs(f"{root}/images/train"); os.makedirs(f"{root}/labels/train")
per=collections.Counter(); n=0; nb=0
for i,(im,rows,tag) in enumerate(items):
    if not keepmask[i] or not ok[i]: continue
    n+=1; per[tag]+=1; nb+=len(rows)
    out=f"{tag}_{n:05d}"
    shutil.copy(im,f"{root}/images/train/{out}{os.path.splitext(im)[1]}")
    open(f"{root}/labels/train/{out}.txt","w").write(
        "".join("0 "+" ".join(r[1:5])+"\n" for r in rows))
open(f"{root}/data.yaml","w").write(
  f"path: {os.path.abspath(root)}\ntrain: images/train\n"
  f"val: {os.path.abspath('out/yolo_val_full')}/images/val\nnc: 1\nnames: [robot]\n")
print(f"\nFINAL: {n} images, {nb} boxes")
for k,v in per.most_common(): print(f"    {k:10} {v:5d}")
bv=n-per['jw']
print(f"  broadcast-view {bv} ({100*bv/n:.0f}%)   other-viewpoint {per['jw']} ({100*per['jw']/n:.0f}%)")
print(f"  previous pool: 3485 imgs / 2443 broadcast-view")
