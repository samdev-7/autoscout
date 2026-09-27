"""Dedupe by SOURCE VIDEO, not by pixels.

Perceptual hashing provably cannot separate 'duplicate' from 'different frame of
the same static-camera match'. But the filenames carry source-clip identity, and
cross-dataset compilation is what we actually need to detect.
"""
import glob,os,re,collections
DS=[("data/rf/frcrobots_v5","frc"),("data/rf/jerrywu_v2","jw"),("data/rf/t6036","t6036"),
    ("data/rf/grain","grain"),("data/rf/frc4419","f4419"),("data/rf/capstone","capstone"),
    ("data/rf/scout_v7","scout_v7")]
rf_re=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$",re.I)
def source_of(fn):
    s=rf_re.sub("",fn)
    s=re.sub(r"\.(jpg|png|jpeg)$","",s,flags=re.I)
    s=re.sub(r"[-_]?\d{2,6}$","",s)                    # trailing frame number
    s=re.sub(r"[-_]?(mp4|mkv|webm)$","",s,flags=re.I)
    s=re.sub(r"[-_]?frame$","",s,flags=re.I)
    return s.strip("-_ ").lower()
src=collections.defaultdict(lambda: collections.Counter())
for loc,tag in DS:
    for im in glob.glob(f"{loc}/*/images/*.*"):
        src[tag][source_of(os.path.basename(im))]+=1
print(f"{'dataset':10}{'imgs':>7}{'sources':>9}{'med f/src':>11}   top sources")
for loc,tag in DS:
    c=src[tag]; n=sum(c.values())
    med=sorted(c.values())[len(c)//2] if c else 0
    top=", ".join(f"{k[:26]}({v})" for k,v in c.most_common(3))
    print(f"{tag:10}{n:7d}{len(c):9d}{med:11d}   {top}")
print("\nCROSS-DATASET source overlap (the compilation check):")
allsrc={t:set(c) for t,c in src.items()}
found=False
tags=[t for _,t in DS]
for i in range(len(tags)):
    for j in range(i+1,len(tags)):
        ov=allsrc[tags[i]]&allsrc[tags[j]]
        ov={o for o in ov if len(o)>6}
        if ov:
            found=True
            print(f"  {tags[i]} <-> {tags[j]}: {len(ov)} shared sources  e.g. {list(ov)[:3]}")
if not found: print("  none — no dataset is a compilation of another (by source id)")
