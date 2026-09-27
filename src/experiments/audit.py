"""Audit every candidate source. Flag anything that risks garbage-in."""
import glob,os,re,cv2,numpy as np,collections,json
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
DISCARD={"Algae","Coral","note","RedSpkr","RedAmp","BlueSpkr","BlueAmp",
         "SB","TIME","SR","Qualification","Auto","Tele Op","'3'"}
DS=[("data/rf/frcrobots_v5","frc"),("data/rf/jerrywu_v2","jw"),
    ("data/rf/t6036","t6036"),("data/rf/grain","grain"),("data/rf/frc4419","f4419"),
    ("data/rf/scout_v7","scout_v7"),("data/rf/scout2025","scout2025"),
    ("data/rf/pack1294","pack1294"),("data/rf/capstone","capstone"),
    ("data/rf/alis","alis"),("data/rf/adamk","adamk")]
stem_re=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$")
print(f"{'source':10}{'imgs':>7}{'uniq':>7}{'aug':>6}{'rbox':>7}{'b/img':>7}"
      f"{'aspect':>8}{'area%':>7}{'empty':>7}  unknown classes")
rep={}
for loc,tag in DS:
    if not os.path.isdir(loc): continue
    y=open(f"{loc}/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    unk=[n for n in names if n not in ROBOT and n not in DISCARD]
    keep={i for i,n in enumerate(names) if n in ROBOT}
    imgs=sorted(glob.glob(f"{loc}/*/images/*.*"))
    stems=set(); nb=0; empty=0; asp=[]; area=[]
    for im in imgs:
        stems.add(stem_re.sub("",os.path.basename(im)))
        lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lb): empty+=1; continue
        rows=[l.split() for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep]
        if not rows: empty+=1; continue
        nb+=len(rows)
        for r in rows:
            w,h=float(r[3]),float(r[4])
            if h>0: asp.append(w/h); area.append(w*h*100)
    ni=len(imgs)
    rep[tag]=dict(imgs=ni,uniq=len(stems),rbox=nb,bpi=nb/max(ni-empty,1),
                  asp=float(np.median(asp)) if asp else 0,
                  area=float(np.median(area)) if area else 0,empty=empty,unk=unk)
    print(f"{tag:10}{ni:7d}{len(stems):7d}{ni/max(len(stems),1):6.1f}{nb:7d}"
          f"{rep[tag]['bpi']:7.2f}{rep[tag]['asp']:8.2f}{rep[tag]['area']:7.2f}{empty:7d}  {unk}")
json.dump(rep,open("out/audit.json","w"))
print("\nreference: OUR labels")
L=json.load(open("out/labels.json"))
a=[];ar=[]
for f,bs in L["frames"].items():
    for b in bs:
        w=(b["x1"]-b["x0"])/1920; h=(b["y1"]-b["y0"])/344
        a.append((b["x1"]-b["x0"])/(b["y1"]-b["y0"])); ar.append(w*h*100)
print(f"{'ours':10}{83:7d}{83:7d}{1.0:6.1f}{421:7d}{421/83:7.2f}"
      f"{np.median(a):8.2f}{np.median(ar):7.2f}{0:7d}")
print("\nFLAGS")
for t,r in rep.items():
    fl=[]
    if r["bpi"]<3.5: fl.append(f"LOW boxes/img {r['bpi']:.2f} (under-labelled?)")
    if r["bpi"]>7.5: fl.append(f"HIGH boxes/img {r['bpi']:.2f}")
    if r["asp"]>2.2: fl.append(f"WIDE boxes aspect {r['asp']:.2f} (bumper-only convention?)")
    if r["imgs"]/max(r["uniq"],1)>1.6: fl.append(f"augmented x{r['imgs']/r['uniq']:.1f}")
    if r["unk"]: fl.append(f"unknown classes {r['unk']}")
    if r["empty"]/max(r["imgs"],1)>0.15: fl.append(f"{100*r['empty']/r['imgs']:.0f}% empty")
    if fl: print(f"  {t:10} {'; '.join(fl)}")
