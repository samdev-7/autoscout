"""Proper all-pairs perceptual dedupe, with a validity test of the hash itself."""
import glob,os,cv2,numpy as np,collections,json
def dhash(path,s=8):
    im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    r=cv2.resize(im,(s+1,s),interpolation=cv2.INTER_AREA)
    bits=(r[:,1:]>r[:,:-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]

# --- sanity test: does dHash survive re-encoding / resize, and separate distinct images? ---
fs=sorted(glob.glob("out/yolo_train2/images/train/*.jpg"))[:6]
im=cv2.imread(fs[0])
cv2.imwrite("/tmp/_q40.jpg",im,[cv2.IMWRITE_JPEG_QUALITY,40])
cv2.imwrite("/tmp/_half.jpg",cv2.resize(im,(im.shape[1]//2,im.shape[0]//2)))
a=dhash(fs[0]); b=dhash("/tmp/_q40.jpg"); c=dhash("/tmp/_half.jpg"); d=dhash(fs[1])
pc=lambda x:int(np.bitwise_count(np.uint64(x)))
print("hash sanity check (Hamming distance):")
print(f"  same image, JPEG q40      : {pc(a^b)}   (expect 0-3)")
print(f"  same image, half size     : {pc(a^c)}   (expect 0-5)")
print(f"  different image           : {pc(a^d)}   (expect >15)")

paths=sorted(glob.glob("out/yolo_train2/images/train/*.*"))
H=np.zeros(len(paths),dtype=np.uint64); ok=np.ones(len(paths),bool)
for i,p in enumerate(paths):
    h=dhash(p)
    if h is None: ok[i]=False
    else: H[i]=h
print(f"\nhashed {ok.sum()} images; running FULL pairwise ({len(paths)**2/2e6:.1f}M pairs)")
tag=[os.path.basename(p).split("_")[0] for p in paths]
dupes=[]; CH=512
for s in range(0,len(H),CH):
    e=min(s+CH,len(H))
    d=np.bitwise_count(H[s:e,None]^H[None,:])
    for r in range(e-s):
        i=s+r
        j=np.where((d[r]<=5)&(np.arange(len(H))<i))[0]
        for jj in j: dupes.append((i,int(jj)))
print(f"near-duplicate pairs (Hamming<=5): {len(dupes)}")
pair=collections.Counter(tuple(sorted((tag[i],tag[j]))) for i,j in dupes)
for k,v in pair.most_common(10): print(f"    {k[0]:6} <-> {k[1]:6} : {v}")
drop=set()
for i,j in dupes:
    if j not in drop: drop.add(i)
print(f"\nimages to drop: {len(drop)}  -> {len(paths)-len(drop)} unique remain")
json.dump(sorted(drop),open("out/dupe_idx.json","w"))
json.dump(paths,open("out/dupe_paths.json","w"))
