"""Calibrate the dedupe threshold with a proper control and a visual check."""
import glob,os,cv2,numpy as np,collections,random,json
random.seed(0)
def dhash(path,s=8):
    im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    r=cv2.resize(im,(s+1,s),interpolation=cv2.INTER_AREA)
    return np.packbits((r[:,1:]>r[:,:-1]).flatten()).view(np.uint64)[0]
pc=lambda x:int(np.bitwise_count(np.uint64(x)))
paths=sorted(glob.glob("out/yolo_train2/images/train/*.*"))
tag=[os.path.basename(p).split("_")[0] for p in paths]
by=collections.defaultdict(list)
for i,p in enumerate(paths): by[tag[i]].append(i)
H=np.array([dhash(p) or np.uint64(0) for p in paths],dtype=np.uint64)

print("CONTROL: unrelated images (different source datasets)")
a=random.sample(by['frc'],40); b=random.sample(by['jw'],40)
ds=[pc(H[i]^H[j]) for i in a for j in b]
print(f"  frc vs jw   : median {np.median(ds):.0f}  min {min(ds)}  5th pct {np.percentile(ds,5):.0f}")
a=random.sample(by['grain'],40); b=random.sample(by['jw'],40)
ds2=[pc(H[i]^H[j]) for i in a for j in b]
print(f"  grain vs jw : median {np.median(ds2):.0f}  min {min(ds2)}  5th pct {np.percentile(ds2,5):.0f}")

print("\nDISTANCE DISTRIBUTION over all pairs")
CH=512; hist=np.zeros(65,dtype=np.int64)
for s in range(0,len(H),CH):
    e=min(s+CH,len(H))
    d=np.bitwise_count(H[s:e,None]^H[None,:])
    m=np.arange(len(H))[None,:]<np.arange(s,e)[:,None]
    hist+=np.bincount(d[m].ravel(),minlength=65)
tot=hist.sum()
for t in (0,1,2,3,5,8,10,12):
    c=hist[:t+1].sum(); print(f"  pairs with distance <= {t:2d}: {c:8d}  ({100*c/tot:.3f}% of all pairs)")

# visual check: sample flagged pairs at distance 0-2 and 4-5
def sample_pairs(lo,hi,k=3):
    out=[]
    for _ in range(60000):
        i,j=random.randrange(len(H)),random.randrange(len(H))
        if i==j: continue
        d=pc(H[i]^H[j])
        if lo<=d<=hi: out.append((i,j,d))
        if len(out)>=k: break
    return out
rows=[]
for lo,hi,lab in ((0,0,"dist 0"),(1,2,"dist 1-2"),(4,5,"dist 4-5"),(8,10,"dist 8-10")):
    for i,j,d in sample_pairs(lo,hi,2):
        ims=[]
        for k in (i,j):
            im=cv2.imread(paths[k]); im=cv2.resize(im,(300,180))
            cv2.putText(im,f"{tag[k]} {os.path.basename(paths[k])[:18]}",(5,15),
                        cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1)
            ims.append(im)
        pair=np.hstack(ims)
        cv2.putText(pair,f"{lab} (d={d})",(5,172),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
        rows.append(pair)
if rows:
    cv2.imwrite("out/dupe_pairs.jpg",np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,82])
    print(f"\nwrote out/dupe_pairs.jpg ({len(rows)} sampled pairs for visual check)")
