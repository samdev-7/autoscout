"""Full run, checkpointed so it can be stopped at any time and still be useful.

Fixes applied:
  - best.pt selected on COMBINED val (169 frames, 80+ events) not one match
  - iteration-comparable epochs: armA 45 / armB 25
  - imgsz 640 for the arms, 960 for the winner's continuation
  - patience 20 so a plateau frees budget instead of burning it
  - save_period=5 plus a STATUS.md refreshed at every stage
"""
import os,time,json,glob,traceback
def log(m):
    s=f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s,flush=True); open("out/full.log","a").write(s+"\n")

def status(note=""):
    r=json.load(open("out/results.json")) if os.path.exists("out/results.json") else []
    best=max(r,key=lambda x:(x["multi"]["mAP50"],x["ours"]["mAP50"])) if r else None
    L=["# autoscout — training status","",f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}","",
       f"**{note}**","","## best model so far",""]
    if best:
        L+=[f"- weights: `{best['weights']}`",
            f"- our 421 boxes : mAP50 {best['ours']['mAP50']:.3f}  R {best['ours']['R']:.3f}  "
            f"P {best['ours']['P']:.3f}  mAP50-95 {best['ours']['mAP']:.3f}",
            f"- 80 events     : mAP50 {best['multi']['mAP50']:.3f}  R {best['multi']['R']:.3f}  "
            f"P {best['multi']['P']:.3f}  mAP50-95 {best['multi']['mAP']:.3f}",""]
    else: L+=["- none evaluated yet",""]
    L+=["## all evaluated checkpoints","",
        "| run | ours mAP50 | ours R | multi mAP50 | multi R | weights |","|---|---|---|---|---|---|"]
    for x in r:
        L.append(f"| {x['name']} | {x['ours']['mAP50']:.3f} | {x['ours']['R']:.3f} | "
                 f"{x['multi']['mAP50']:.3f} | {x['multi']['R']:.3f} | `{os.path.basename(os.path.dirname(os.path.dirname(x['weights'])))}` |")
    L+=["","## if you stop this run","",
        "Every run keeps `weights/best.pt` and `weights/last.pt`, plus epoch snapshots every 5",
        "epochs, under `runs/detect/out/runs/<name>/weights/`. Per-epoch metrics are in that",
        "run's `results.csv`. Killing the process loses nothing already written.",""]
    open("out/STATUS.md","w").write("\n".join(L))

def train(name,data,epochs,imgsz=640,weights="yolo11s.pt",lr0=None,cm=10):
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    kw=dict(data=data,epochs=epochs,imgsz=imgsz,batch=16,device="mps",workers=8,seed=0,
            deterministic=False,cache="ram",amp=True,
            patience=20,                 # a real plateau should free budget
            save_period=5,               # resumable snapshots
            close_mosaic=cm,scale=0.5,mosaic=1.0,degrees=0.0,fliplr=0.5,flipud=0.0,
            hsv_h=0.015,hsv_s=0.7,hsv_v=0.4,erasing=0.4,
            project="out/runs",name=name,exist_ok=True,val=True,plots=False,verbose=True)
    if lr0: kw["lr0"]=lr0
    t0=time.time(); YOLO(weights).train(**kw)
    log(f"{name}: trained {(time.time()-t0)/3600:.2f}h")

def wp(n):
    c=glob.glob(f"**/runs/{n}/weights/best.pt",recursive=True)
    return max(c,key=os.path.getmtime) if c else None

def ev(name,imgsz=640):
    from ultralytics import YOLO
    w=wp(name)
    if not w: log(f"{name}: NO WEIGHTS"); return None
    out={"name":name,"weights":w,"imgsz":imgsz}
    for tag,y in (("ours","out/yolo_val_full/data.yaml"),("multi","out/val_multi/data.yaml")):
        b=YOLO(w).val(data=y,imgsz=imgsz,device="mps",verbose=False,plots=False,split="val").box
        out[tag]=dict(P=round(float(b.mp),4),R=round(float(b.mr),4),
                      mAP50=round(float(b.map50),4),mAP=round(float(b.map),4))
    log(f"{name}  OURS mAP50 {out['ours']['mAP50']:.3f} R {out['ours']['R']:.3f} | "
        f"MULTI mAP50 {out['multi']['mAP50']:.3f} R {out['multi']['R']:.3f}")
    r=json.load(open("out/results.json")) if os.path.exists("out/results.json") else []
    r.append(out); json.dump(r,open("out/results.json","w"),indent=1)
    status(f"finished {name}")
    return out

open("out/full.log","a").write("\n=== FULL RUN "+time.strftime("%Y-%m-%d %H:%M")+" ===\n")
status("starting")
R={}
for nm,dat,ep in (("fullA","out/armA/data.yaml",45),("fullB","out/armB/data.yaml",25)):
    try:
        log(f"=== {nm}: {ep} epochs @640 ==="); status(f"training {nm} ({ep} epochs)")
        train(nm,dat,ep); R[nm]=ev(nm)
    except Exception: log(f"{nm} FAILED\n"+traceback.format_exc()[-800:]); status(f"{nm} failed")
ok={k:v for k,v in R.items() if v}
if ok:
    win=max(ok,key=lambda k:(ok[k]["multi"]["mAP50"],ok[k]["ours"]["mAP50"]))
    log(f"WINNER {win}")
    json.dump({"winner":win},open("out/winner.json","w"),indent=1)
    try:
        d="out/armA/data.yaml" if win=="fullA" else "out/armB/data.yaml"
        log(f"=== continuing {win} @960 for 40 epochs ==="); status(f"continuing {win} at imgsz 960")
        train(f"{win}_960",d,40,imgsz=960,weights=wp(win),lr0=0.002,cm=12)
        ev(f"{win}_960",imgsz=960)
    except Exception: log("continuation FAILED\n"+traceback.format_exc()[-800:])
    r=json.load(open("out/results.json"))
    fin=max(r,key=lambda x:(x["multi"]["mAP50"],x["ours"]["mAP50"]))
    json.dump(fin,open("out/FINAL_MODEL.json","w"),indent=1)
    log(f"FINAL BEST {fin['name']} -> {fin['weights']}")
status("complete")
log("FULL RUN COMPLETE")
