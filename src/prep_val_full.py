"""Validation = FULL field-band frames, matching how training images are framed.

Tiling zoomed our robots to 2.75x the external training scale. Evaluating full
frames instead puts both distributions on the same footing, so the result
measures transfer rather than a scale gap I introduced.
"""
import json,os,shutil,numpy as np,cv2
L=json.load(open("out/labels.json")); F=L["frames"]; B0,B1=L["band"]; BH=B1-B0; W=1920
root="out/yolo_val_full"; shutil.rmtree(root,ignore_errors=True)
os.makedirs(f"{root}/images/val"); os.makedirs(f"{root}/labels/val")
gt={}; nb=0
for fn,boxes in sorted(F.items()):
    img=cv2.imread(f"out/frames/{fn}")
    if img is None: continue
    stem=fn[:-4]
    cv2.imwrite(f"{root}/images/val/{stem}.jpg",img[B0:B1],[cv2.IMWRITE_JPEG_QUALITY,94])
    lines=[];meta=[]
    for b in boxes:
        y0,y1=b["y0"]-B0,b["y1"]-B0
        lines.append(f"0 {(b['x0']+b['x1'])/2/W:.6f} {(y0+y1)/2/BH:.6f} "
                     f"{(b['x1']-b['x0'])/W:.6f} {(y1-y0)/BH:.6f}")
        meta.append({"c":b["c"],"box":[b["x0"],y0,b["x1"],y1]}); nb+=1
    open(f"{root}/labels/val/{stem}.txt","w").write("\n".join(lines)+"\n")
    gt[stem]=meta
json.dump(gt,open(f"{root}/alliance_gt.json","w"))
open(f"{root}/data.yaml","w").write(
  f"path: {os.path.abspath('out/yolo_train')}\ntrain: images/train\n"
  f"val: {os.path.abspath(root)}/images/val\nnc: 1\nnames: [robot]\n")
print(f"val: {len(gt)} full-band frames ({W}x{BH}), {nb} boxes")

import glob
def widths(i,l,cap=1500):
    w=[]
    for f in sorted(glob.glob(f"{i}/*.*"))[:cap]:
        p=f"{l}/{os.path.splitext(os.path.basename(f))[0]}.txt"
        if os.path.exists(p):
            w+=[float(x.split()[3]) for x in open(p) if len(x.split())>=5]
    return np.array(w)
tr=widths("out/yolo_train/images/train","out/yolo_train/labels/train")
va=widths(f"{root}/images/val",f"{root}/labels/val")
print(f"\nnormalised box width: train median {np.median(tr):.4f}   val median {np.median(va):.4f}")
for sz in (640,960,1280):
    print(f"  imgsz {sz:4d}: train ~{np.median(tr)*sz:5.1f}px   val ~{np.median(va)*sz:5.1f}px"
          f"   ratio {np.median(va)/np.median(tr):.2f}x")
