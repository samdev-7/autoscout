"""Build a YOLO dataset: tile the field band, split by TIME BLOCK.

Tiling keeps robots ~100 px at imgsz 640 instead of shrinking them, and turns
83 frames into several hundred samples. The split is a contiguous time block, not
random: frames 2 s apart still share lighting/robots, so a random split would
leak and flatter the score.
"""
import json, os, shutil, numpy as np, cv2

TW, STRIDE = 640, 427          # tile width, stride (overlap 213 > widest robot)
VAL_T = 120.0                  # val = contiguous tail of the match
CLS = {"R":0, "B":1}

L=json.load(open("out/labels.json")); F=L["frames"]; B0,B1=L["band"]
man={m["file"]:m["t"] for m in json.load(open("out/frames.json"))["frames"]}
BH=B1-B0
xs=list(range(0,1920-TW+1,STRIDE))
if xs[-1]!=1920-TW: xs.append(1920-TW)
print(f"band h={BH}  tiles/frame={len(xs)}  x starts={xs}")

root="out/yolo"
shutil.rmtree(root,ignore_errors=True)
for sp in ("train","val"):
    os.makedirs(f"{root}/images/{sp}",exist_ok=True)
    os.makedirs(f"{root}/labels/{sp}",exist_ok=True)

stat={"train":[0,0],"val":[0,0]}   # tiles, boxes
for fn,boxes in sorted(F.items()):
    t=man[fn]; sp="val" if t>=VAL_T else "train"
    img=cv2.imread(f"out/frames/{fn}")
    if img is None: continue
    band=img[B0:B1]
    for xi,x0 in enumerate(xs):
        tile=band[:, x0:x0+TW]
        lines=[]
        for b in boxes:
            bx0,bx1=b["x0"]-x0, b["x1"]-x0
            by0,by1=b["y0"]-B0, b["y1"]-B0
            ix0,ix1=max(0,bx0),min(TW,bx1); iy0,iy1=max(0,by0),min(BH,by1)
            if ix1<=ix0 or iy1<=iy0: continue
            area=(bx1-bx0)*(by1-by0); vis=(ix1-ix0)*(iy1-iy0)
            if vis/area < 0.70: continue          # skip badly clipped boxes
            cx=(ix0+ix1)/2/TW; cy=(iy0+iy1)/2/BH
            w=(ix1-ix0)/TW;   h=(iy1-iy0)/BH
            lines.append(f"{CLS[b['c']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines: continue                     # drop empty tiles
        stem=f"{fn[:-4]}_t{xi}"
        cv2.imwrite(f"{root}/images/{sp}/{stem}.jpg",tile,[cv2.IMWRITE_JPEG_QUALITY,94])
        open(f"{root}/labels/{sp}/{stem}.txt","w").write("\n".join(lines)+"\n")
        stat[sp][0]+=1; stat[sp][1]+=len(lines)

open(f"{root}/data.yaml","w").write(
    f"path: {os.path.abspath(root)}\ntrain: images/train\nval: images/val\n"
    f"nc: 2\nnames: [red_robot, blue_robot]\n")
print(f"\n{'split':>6} {'tiles':>7} {'boxes':>7} {'boxes/tile':>11}")
for sp in ("train","val"):
    t,b=stat[sp]; print(f"{sp:>6} {t:7d} {b:7d} {b/max(t,1):11.2f}")
tf=sum(1 for fn in F if man[fn]<VAL_T); vf=len(F)-tf
print(f"\nsource frames: {tf} train (t<{VAL_T:.0f}s), {vf} val (t>={VAL_T:.0f}s)")
ws=[float(l.split()[3])*TW for sp in ("train","val")
    for f in os.listdir(f"{root}/labels/{sp}") for l in open(f"{root}/labels/{sp}/{f}")]
print(f"box widths in tiles: mean {np.mean(ws):.0f}px  min {np.min(ws):.0f}  max {np.max(ws):.0f}"
      f"   (imgsz 640 -> robots stay ~{np.mean(ws):.0f}px, not small-object regime)")
