"""Pool external datasets into one training set: unique sources only, single class."""
import glob,os,re,shutil,collections,numpy as np,cv2
SRC=[("data/rf/frcrobots_v5","frc"),("data/rf/jerrywu_v2","jw")]
root="out/yolo_train"; shutil.rmtree(root,ignore_errors=True)
os.makedirs(f"{root}/images/train"); os.makedirs(f"{root}/labels/train")
seen=set(); n=0; per=collections.Counter()
stem_re=re.compile(r"[._](jpg|png)\.rf\.[0-9a-f]+\.(jpg|png)$")
for loc,tag in SRC:
    for sp in ("train","valid","test"):
        for im in sorted(glob.glob(f"{loc}/{sp}/images/*.*")):
            b=os.path.basename(im); s=tag+"|"+stem_re.sub("",b)
            if s in seen: continue                  # drop augmented duplicates
            lb=f"{loc}/{sp}/labels/{os.path.splitext(b)[0]}.txt"
            if not os.path.exists(lb): continue
            seen.add(s); n+=1; per[tag]+=1
            out=f"{tag}_{n:05d}"
            shutil.copy(im,f"{root}/images/train/{out}{os.path.splitext(b)[1]}")
            # force class 0 (all sources are single-class 'robot')
            rows=[l.split() for l in open(lb) if l.strip()]
            open(f"{root}/labels/train/{out}.txt","w").write(
                "".join("0 "+" ".join(r[1:])+"\n" for r in rows))
open(f"{root}/data.yaml","w").write(
  f"path: {os.path.abspath(root)}\ntrain: images/train\n"
  f"val: {os.path.abspath('out/yolo_val')}/images/val\nnc: 1\nnames: [robot]\n")
print(f"training images: {n}  ({dict(per)})")

def widths(imgdir,lbldir,cap=1200):
    w=[]
    fs=sorted(glob.glob(f"{imgdir}/*.*"))[:cap]
    for f in fs:
        l=f"{lbldir}/{os.path.splitext(os.path.basename(f))[0]}.txt"
        if not os.path.exists(l): continue
        for line in open(l):
            p=line.split()
            if len(p)>=5: w.append(float(p[3]))     # normalised width
    return np.array(w)
tr=widths(f"{root}/images/train",f"{root}/labels/train")
va=widths("out/yolo_val/images/val","out/yolo_val/labels/val")
print(f"\nbox width as fraction of image width:")
for nm,a in (("external train",tr),("our val tiles",va)):
    print(f"  {nm:15} n={len(a):5d}  median {np.median(a):.3f}  "
          f"16-84pct {np.percentile(a,16):.3f}-{np.percentile(a,84):.3f}")
print(f"\n  at imgsz=640 -> train robots ~{np.median(tr)*640:.0f}px, "
      f"our val robots ~{np.median(va)*640:.0f}px")
r=np.median(va)/np.median(tr)
print(f"  scale ratio val/train = {r:.2f}x  -> "
      f"{'close enough; scale aug covers it' if 0.6<r<1.7 else 'MISMATCH: needs scale augmentation'}")
