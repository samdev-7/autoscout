"""One PDF page per dataset: random samples with ONLY robot-class boxes drawn."""
import glob,os,cv2,numpy as np,random
from PIL import Image
random.seed(7)
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
DS=[("data/rf/frcrobots_v5","frcrobots_v5","broadcast, 2024mibel only"),
    ("data/rf/jerrywu_v2","jerrywu_v2","mixed viewpoints, 2019-2023"),
    ("data/rf/t6036","6036-ai","broadcast 4K"),
    ("data/rf/grain","grain-scouting","broadcast, game pieces filtered out"),
    ("data/rf/frc4419","frc4419","broadcast strip 1280x490"),
    ("data/rf/capstone","capstone","broadcast strip 680x139"),
    ("data/rf/scout_v7","scout_v7","2026 REBUILT broadcast"),
    ("data/rf/scout2025","scout2025","flagged: cutout aug, loose boxes"),
    ("data/rf/pack1294","pack1294","flagged: oversized boxes"),
    ("data/rf/alis","alis","flagged: under-labelled, close-up"),
    ("data/rf/adamk","adamk","flagged: baked-in mosaic aug")]
PW,PH=1180,850; COLS,ROWS=3,2
CW,CH=PW//COLS, (PH-70)//ROWS
pages=[]
for loc,name,note in DS:
    if not os.path.isdir(loc): continue
    y=open(f"{loc}/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    keep={i for i,n in enumerate(names) if n in ROBOT}
    kept=[n for n in names if n in ROBOT]; dropped=[n for n in names if n not in ROBOT]
    fs=sorted(glob.glob(f"{loc}/*/images/*.*"))
    # prefer frames that actually contain robot boxes
    cand=[]
    for f in random.sample(fs,min(len(fs),140)):
        lb=f.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lb): continue
        nb=sum(1 for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep)
        if nb: cand.append((f,nb))
        if len(cand)>=COLS*ROWS: break
    page=np.full((PH,PW,3),18,np.uint8)
    cv2.putText(page,name,(22,34),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(page,f"{note}   |   {len(fs)} imgs   |   robot classes kept: {kept}",
                (22,58),cv2.FONT_HERSHEY_SIMPLEX,0.52,(150,200,255),1,cv2.LINE_AA)
    if dropped:
        cv2.putText(page,f"discarded classes (not drawn): {dropped}",(22,76),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(140,140,140),1,cv2.LINE_AA)
    for i,(f,nb) in enumerate(cand):
        im=cv2.imread(f)
        if im is None: continue
        H,W=im.shape[:2]
        lb=f.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        for l in open(lb):
            p=l.split()
            if len(p)<5 or int(p[0]) not in keep: continue      # robot boxes ONLY
            x,yy,bw,bh=[float(v) for v in p[1:5]]
            x0,y0=int((x-bw/2)*W),int((yy-bh/2)*H); x1,y1=int((x+bw/2)*W),int((yy+bh/2)*H)
            cv2.rectangle(im,(x0,y0),(x1,y1),(0,230,255),max(2,W//420))
        s=min((CW-16)/W,(CH-30)/H); im=cv2.resize(im,(int(W*s),int(H*s)))
        r,c=divmod(i,COLS); ox,oy=c*CW+8, 88+r*CH
        page[oy:oy+im.shape[0], ox:ox+im.shape[1]]=im
        cv2.putText(page,f"{nb} robots  {W}x{H}  {os.path.basename(f)[:30]}",
                    (ox,oy+im.shape[0]+16),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1,cv2.LINE_AA)
    pages.append(Image.fromarray(cv2.cvtColor(page,cv2.COLOR_BGR2RGB)))
    print(f"  page: {name} ({len(cand)} samples)")
pages[0].save("out/dataset_samples.pdf",save_all=True,append_images=pages[1:],resolution=96,quality=52,optimize=True)
print(f"\nwrote out/dataset_samples.pdf — {len(pages)} pages")
