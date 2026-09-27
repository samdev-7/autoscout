"""Detect broadcast frames by flat-colour overlay regions (score bug).

A score bug contributes large LOW-VARIANCE patches that are strongly red, strongly
blue, or near-black. Natural scenes rarely produce flat saturated regions.
Compression noise is tolerated by measuring local std rather than exact equality.
"""
import glob,os,re,cv2,numpy as np,random,collections,json
random.seed(1)
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}

def flat_stats(path):
    im=cv2.imread(path)
    if im is None: return None
    im=cv2.resize(im,(320,180),interpolation=cv2.INTER_AREA)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    m=cv2.blur(g,(9,9)); s2=cv2.blur(g*g,(9,9))
    std=np.sqrt(np.maximum(s2-m*m,0))
    uni=std<4.5                                    # tolerant of JPEG noise
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    H,S,V=hsv[...,0].astype(int),hsv[...,1].astype(int),hsv[...,2].astype(int)
    red =uni&(((H<=8)|(H>=172))&(S>105)&(V>55))
    blue=uni&((H>=100)&(H<=134)&(S>105)&(V>45))
    dark=uni&(V<45)
    return float(red.mean()),float(blue.mean()),float(dark.mean())

def is_broadcast(r,b,d):
    return (r>0.0015 and b>0.0015 and d>0.010) or (d>0.045 and (r+b)>0.004)

DS=[("data/rf/frcrobots_v5","frc","broadcast"),("data/rf/jerrywu_v2","jw","mixed/POV"),
    ("data/rf/t6036","t6036","broadcast"),("data/rf/grain","grain","broadcast"),
    ("data/rf/frc4419","f4419","broadcast"),("data/rf/capstone","capstone","broadcast"),
    ("data/rf/scout_v7","scout_v7","broadcast")]
print("VALIDATION — detector vs known nature of each source (400-image sample)")
print(f"{'source':10}{'known':11}{'red%':>7}{'blue%':>7}{'dark%':>7}{'-> broadcast':>14}")
rate={}
for loc,tag,known in DS:
    fs=glob.glob(f"{loc}/*/images/*.*"); random.shuffle(fs); fs=fs[:400]
    R=[];B=[];D=[];hit=0
    for f in fs:
        s=flat_stats(f)
        if not s: continue
        R.append(s[0]);B.append(s[1]);D.append(s[2]); hit+=is_broadcast(*s)
    rate[tag]=hit/max(len(R),1)
    print(f"{tag:10}{known:11}{100*np.median(R):7.2f}{100*np.median(B):7.2f}"
          f"{100*np.median(D):7.2f}{100*rate[tag]:13.0f}%")

def pool_counts(specs):
    """exact image + box counts after class filter and empty removal"""
    ni=nb=0; per={}
    for loc,tag in specs:
        y=open(f"{loc}/data.yaml").read()
        names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
        keep={i for i,n in enumerate(names) if n in ROBOT}
        i0=b0=0
        for im in glob.glob(f"{loc}/*/images/*.*"):
            lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
            if not os.path.exists(lb): continue
            rows=[l for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep]
            if not rows: continue
            i0+=1; b0+=len(rows)
        per[tag]=(i0,b0); ni+=i0; nb+=b0
    return ni,nb,per

OLD=[("data/rf/frcrobots_v5","frc"),("data/rf/jerrywu_v2","jw")]
NEW=OLD+[("data/rf/t6036","t6036"),("data/rf/grain","grain"),("data/rf/frc4419","f4419"),
         ("data/rf/capstone","capstone"),("data/rf/scout_v7","scout_v7")]
for label,spec in (("BEFORE",OLD),("AFTER",NEW)):
    ni,nb,per=pool_counts(spec)
    bcast_i=sum(rate.get(t,0)*v[0] for t,v in per.items())
    bcast_b=sum(rate.get(t,0)*v[1] for t,v in per.items())
    print(f"\n=== {label} ===")
    print(f"  images {ni:6d}   boxes {nb:6d}   boxes/img {nb/ni:.2f}")
    print(f"  broadcast (est) images {bcast_i:7.0f} ({100*bcast_i/ni:4.1f}%)   "
          f"boxes {bcast_b:7.0f} ({100*bcast_b/nb:4.1f}%)")
    print(f"  non-broadcast   images {ni-bcast_i:7.0f} ({100*(1-bcast_i/ni):4.1f}%)")
    for t,(i0,b0) in per.items(): print(f"     {t:10} {i0:6d} imgs {b0:6d} boxes  bcast~{100*rate.get(t,0):3.0f}%")
