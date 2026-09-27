"""Our 83 labelled frames become a VALIDATION SET ONLY.

Nothing from this video enters training. That keeps the measurement honest: it
answers "does an externally-trained model transfer to an unseen event/camera",
which is the actual deployment question.
"""
import json, os, shutil, numpy as np, cv2
TW, STRIDE = 640, 427
L=json.load(open("out/labels.json")); F=L["frames"]; B0,B1=L["band"]; BH=B1-B0
xs=list(range(0,1920-TW+1,STRIDE))
if xs[-1]!=1920-TW: xs.append(1920-TW)

root="out/yolo_val"; shutil.rmtree(root,ignore_errors=True)
os.makedirs(f"{root}/images/val"); os.makedirs(f"{root}/labels/val")
nb=nt=0; alli={}
for fn,boxes in sorted(F.items()):
    img=cv2.imread(f"out/frames/{fn}")
    if img is None: continue
    band=img[B0:B1]
    for xi,x0 in enumerate(xs):
        lines=[]; meta=[]
        for b in boxes:
            bx0,bx1=b["x0"]-x0,b["x1"]-x0; by0,by1=b["y0"]-B0,b["y1"]-B0
            ix0,ix1=max(0,bx0),min(TW,bx1); iy0,iy1=max(0,by0),min(BH,by1)
            if ix1<=ix0 or iy1<=iy0: continue
            if (ix1-ix0)*(iy1-iy0)/((bx1-bx0)*(by1-by0)) < 0.70: continue
            lines.append(f"0 {(ix0+ix1)/2/TW:.6f} {(iy0+iy1)/2/BH:.6f} "
                         f"{(ix1-ix0)/TW:.6f} {(iy1-iy0)/BH:.6f}")   # single class
            meta.append({"c":b["c"],"box":[ix0,iy0,ix1,iy1]})
        if not lines: continue
        stem=f"{fn[:-4]}_t{xi}"
        cv2.imwrite(f"{root}/images/val/{stem}.jpg",band[:,x0:x0+TW],[cv2.IMWRITE_JPEG_QUALITY,94])
        open(f"{root}/labels/val/{stem}.txt","w").write("\n".join(lines)+"\n")
        alli[stem]=meta; nt+=1; nb+=len(lines)
json.dump(alli,open(f"{root}/alliance_gt.json","w"))
open(f"{root}/data.yaml","w").write(
    f"path: {os.path.abspath(root)}\ntrain: images/val\nval: images/val\n"
    f"nc: 1\nnames: [robot]\n")
r=sum(1 for m in alli.values() for x in m if x["c"]=="R")
print(f"validation set: {nt} tiles, {nb} boxes  ({r} red / {nb-r} blue for the colour test)")
print(f"single class 'robot' for detection; alliance kept separately in alliance_gt.json")
print(f"source: all 83 frames — none of this is available for training")
