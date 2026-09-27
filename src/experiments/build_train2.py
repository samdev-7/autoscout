"""Pool all usable sources, filter junk, and dedupe perceptually.

Roboflow re-encodes images, so byte hashes miss cross-dataset copies. dHash
(64-bit, gradient-based) survives re-encoding and mild resizing.
"""
import glob,os,re,shutil,cv2,numpy as np,collections,json
# src dir, keep-classes (None = all), tag
SRC=[("data/rf/frcrobots_v5",None,"frc"),
     ("data/rf/jerrywu_v2",  None,"jw"),
     ("data/rf/t6036",       None,"t6036"),
     ("data/rf/grain",       {1,3},"grain"),     # 0=Algae 2=Coral are game pieces
     ("data/rf/frc4419",     None,"f4419")]
# adamk excluded: images are baked-in flip/mosaic augmentations of 41 real frames

def dhash(path,s=8):
    im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    r=cv2.resize(im,(s+1,s),interpolation=cv2.INTER_AREA)
    return int("".join('1' if r[y,x+1]>r[y,x] else '0'
               for y in range(s) for x in range(s)),2)
pop=lambda x: bin(x).count("1")

items=[]; stats=collections.Counter(); empty=collections.Counter()
for loc,keep,tag in SRC:
    for sp in ("train","valid","test"):
        for im in sorted(glob.glob(f"{loc}/{sp}/images/*.*")):
            lb=f"{loc}/{sp}/labels/{os.path.splitext(os.path.basename(im))[0]}.txt"
            if not os.path.exists(lb): continue
            rows=[l.split() for l in open(lb) if len(l.split())>=5]
            rows=[r for r in rows if keep is None or int(r[0]) in keep]
            if not rows: empty[tag]+=1; continue          # junk/title-card frames
            items.append((im,rows,tag)); stats[tag]+=1
print(f"usable images per source: {dict(stats)}")
print(f"dropped (no robot boxes after class filter): {dict(empty)}")

print("\nhashing for perceptual dedupe...")
H=[]
for im,rows,tag in items:
    h=dhash(im); H.append(h)
buckets=collections.defaultdict(list)
for i,h in enumerate(H):
    if h is None: continue
    buckets[h>>40].append(i)                              # prefix bucket, then exact compare
seen=set(); dup=0; keepidx=[]; cross=collections.Counter()
for i,h in enumerate(H):
    if h is None: continue
    hit=None
    for j in buckets[h>>40]:
        if j>=i or j in seen: continue
        if pop(h ^ H[j])<=5: hit=j; break
    if hit is not None:
        dup+=1; cross[f"{items[i][2]}<-{items[hit][2]}"]+=1
    else:
        keepidx.append(i); seen.add(i)
print(f"near-duplicates removed: {dup}  ({100*dup/max(len(items),1):.1f}%)")
for k,v in cross.most_common(8): print(f"    {k}: {v}")

root="out/yolo_train2"; shutil.rmtree(root,ignore_errors=True)
os.makedirs(f"{root}/images/train"); os.makedirs(f"{root}/labels/train")
per=collections.Counter(); n=0
for i in keepidx:
    im,rows,tag=items[i]; n+=1; per[tag]+=1
    out=f"{tag}_{n:05d}"
    shutil.copy(im,f"{root}/images/train/{out}{os.path.splitext(im)[1]}")
    open(f"{root}/labels/train/{out}.txt","w").write(
        "".join("0 "+" ".join(r[1:5])+"\n" for r in rows))     # single class 'robot'
open(f"{root}/data.yaml","w").write(
  f"path: {os.path.abspath(root)}\ntrain: images/train\n"
  f"val: {os.path.abspath('out/yolo_val_full')}/images/val\nnc: 1\nnames: [robot]\n")
nb=sum(len(open(f).readlines()) for f in glob.glob(f"{root}/labels/train/*.txt"))
print(f"\nFINAL training set: {n} images, {nb} boxes  ({dict(per)})")
print(f"  previous set was 3485 images  ->  {'+' if n>3485 else ''}{n-3485}")
bv=per['frc']+per['t6036']+per['grain']+per['f4419']
print(f"  broadcast-view: {bv} ({100*bv/n:.0f}%)   other viewpoints: {per['jw']} ({100*per['jw']/n:.0f}%)")
