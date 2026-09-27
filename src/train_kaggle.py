"""Kaggle training: pull datasets from Roboflow, rebuild the pool, train fresh from COCO.

Fresh init (not warm-started) so the artifact is reproducible from this one script.
Add your Roboflow key as a Kaggle Secret named ROBOFLOW_KEY.
"""
import os, glob, re, json, shutil, collections, subprocess, sys
# pinned to the versions the 2026-09-03 RTX 5090 run used: resuming a checkpoint across
# ultralytics versions is not safe, and an unpinned install makes the run unreproducible.
subprocess.run([sys.executable,"-m","pip","install","-q",
                "ultralytics==8.4.138","roboflow==1.4.2"],check=True)

try:
    from kaggle_secrets import UserSecretsClient
    KEY = UserSecretsClient().get_secret("ROBOFLOW_KEY")
except Exception:
    KEY = os.environ["ROBOFLOW_KEY"]

WORK = os.environ.get("WORK", "/kaggle/working")

# the script ships inside the package, so its own directory IS the package
PKG = os.environ.get("PKG") or os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f"{PKG}/heldout_clips.json"):
    raise SystemExit(f"Package files not found next to the script.\nScript dir: {PKG}\n"
                     f"Contains: {sorted(os.listdir(PKG))[:12]}\n"
                     "Expected heldout_clips.json and yolo_val_full/ alongside train_kaggle.py")
print("package:", PKG)

# ---- 1. download the same sources, same versions ----
from roboflow import Roboflow
rf = Roboflow(api_key=KEY)
JOBS = [("frcrobots","frc-robots-1zts0",5,"frc"),
        ("frc-9wt6x","jerrywu_frc114_robot_v2-przxh",2,"jw"),
        ("6036","6036-ai-htnmi",2,"t6036"),
        ("grain-dzomt","scouting-cqm98",7,"grain"),
        ("frc4419","frc-robots-equcn",3,"f4419"),
        ("capstone-roipa","frc-automatic-scouting",30,"capstone"),
        ("-gw5oi","scouting-v7cfq",9,"scout_v7")]
for ws,pr,ver,tag in JOBS:
    d=f"{WORK}/raw/{tag}"
    if not os.path.isdir(d):
        rf.workspace(ws).project(pr).version(ver).download("yolov8",location=d)

# ---- 2. same normalisation as locally: robots only, no aug copies, hold out val clips ----
ROBOT={"robot","Robot","Robots","red_alliance","blue_alliance","unknown_alliance",
       "BLU","RED","Blue-Robot","Red-Robot","red_bumper","blue_bumper"}
rfre=re.compile(r"[._](jpg|png|jpeg)\.rf\.[0-9a-f]+\.(jpg|png|jpeg)$",re.I)
HOLD=set(json.load(open(f"{PKG}/heldout_clips.json")))
def clip(fn):
    t=rfre.sub("",fn); t=re.sub(r"\.(jpg|png|jpeg)$","",t,flags=re.I)
    return re.sub(r"[-_]?\d{2,6}$","",t).lower()
root=f"{WORK}/pool"; shutil.rmtree(root,ignore_errors=True)
os.makedirs(f"{root}/images/train"); os.makedirs(f"{root}/labels/train")
n=0; per=collections.Counter()
for ws,pr,ver,tag in JOBS:
    loc=f"{WORK}/raw/{tag}"
    y=open(f"{loc}/data.yaml").read()
    names=[x.strip("- \n") for x in y.split("names:")[1].split("nc:")[0].strip().split("\n") if x.strip()]
    keep={i for i,nm in enumerate(names) if nm in ROBOT}
    seen=set()
    for im in sorted(glob.glob(f"{loc}/*/images/*.*")):
        st=rfre.sub("",os.path.basename(im))
        if st in seen or clip(os.path.basename(im)) in HOLD: continue
        lb=im.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lb): continue
        rows=[l.split() for l in open(lb) if len(l.split())>=5 and int(l.split()[0]) in keep]
        if not rows: continue
        seen.add(st); n+=1; per[tag]+=1
        o=f"{tag}_{n:05d}"
        shutil.copy(im,f"{root}/images/train/{o}{os.path.splitext(im)[1]}")
        open(f"{root}/labels/train/{o}.txt","w").write(
            "".join("0 "+" ".join(r[1:5])+"\n" for r in rows))
print("pool:",n,"images",dict(per))

# ---- 3. val: COMBINED (our verified match + 80 unseen events) for checkpoint selection ----
for src,dst in (("yolo_val_full","val_ours"),("val_multi","val_multi"),
                ("val_combined","val_combined")):
    shutil.copytree(f"{PKG}/{src}",f"{WORK}/{dst}",dirs_exist_ok=True)
for name,d in (("val_ours",f"{WORK}/val_ours"),("val_multi",f"{WORK}/val_multi"),
               ("val_combined",f"{WORK}/val_combined")):
    open(f"{d}/data.yaml","w").write(
      f"path: {d}\ntrain: images/val\nval: {d}/images/val\nnc: 1\nnames: [robot]\n")
    # exclude CACHE=disk's sidecar .npy files, or every count reads double
    print(f"  {name}: {len([x for x in glob.glob(d+'/images/val/*.*') if not x.endswith('.npy')])} frames")
# selection must NOT depend on a single match
open(f"{root}/data.yaml","w").write(
  f"path: {root}\ntrain: images/train\nval: {WORK}/val_combined/images/val\nnc: 1\nnames: [robot]\n")

# ---- 4. verify we are actually on Kaggle GPU, then train fresh from COCO ----
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO CUDA GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2 (or P100), "
                     "then re-run. Refusing to train on CPU.")
NG=torch.cuda.device_count()
props=torch.cuda.get_device_properties(0)
GB=props.total_memory/2**30
print(f"GPU: {NG} x {props.name}  {GB:.1f} GiB each  |  cuda {torch.version.cuda}")
DEVICE=list(range(NG)) if NG>1 else 0          # DDP across both T4s when present
MODEL=os.environ.get("MODEL","yolo11l.pt"); IMGSZ=int(os.environ.get("IMGSZ","960"))
# batch sized for ~16GiB cards; scales with GPU count under DDP
base={("yolo11s.pt",640):32,("yolo11s.pt",960):24,("yolo11m.pt",640):24,
      ("yolo11m.pt",960):12,("yolo11l.pt",640):16,("yolo11l.pt",960):8}
BATCH=int(os.environ.get("BATCH", base.get((MODEL,IMGSZ),12)*max(NG,1)))
print(f"model {MODEL}  imgsz {IMGSZ}  batch {BATCH}  device {DEVICE}")
from ultralytics import YOLO, settings
settings.update({"sync": False})
EPOCHS=int(os.environ.get("EPOCHS","120"))
WORKERS=int(os.environ.get("WORKERS","4"))          # match to vCPU count on rented boxes
CACHE=os.environ.get("CACHE","")                    # "disk"/"ram" opt-in; I/O only, same pixels
if CACHE.lower() in ("","false","0","none"): CACHE=False
# resume if a previous attempt left a checkpoint that survived
LAST=f"{WORK}/runs/kaggle/weights/last.pt"
RESUME=os.path.exists(LAST)
if RESUME:
    print(f"RESUMING from {LAST} (restores weights, optimizer state and epoch counter)")
    # patience=0 (-> infinite in ultralytics) deliberately disables early stopping on
    # resume.  An early peak is normal here and does NOT mean the run is done: the real
    # best lands in the final close_mosaic=20 epochs, and patience=30 measured from a
    # lucky epoch-15 peak would kill the run long before reaching them.
    YOLO(LAST).train(resume=True, patience=0); raise SystemExit(0)
YOLO(MODEL).train(data=f"{root}/data.yaml",epochs=EPOCHS,imgsz=IMGSZ,batch=BATCH,device=DEVICE,
    workers=WORKERS,seed=0,cache=CACHE,amp=True,patience=30,save_period=10,close_mosaic=20,
    scale=0.5,mosaic=1.0,degrees=0.0,fliplr=0.5,flipud=0.0,
    hsv_h=0.015,hsv_s=0.7,hsv_v=0.4,erasing=0.4,
    project=f"{WORK}/runs",name="kaggle",exist_ok=True,val=True,plots=True)
BEST=f"{WORK}/runs/kaggle/weights/best.pt"
print("BEST:",BEST, os.path.exists(BEST))
# copy weights to /kaggle/working root so they persist as notebook output
for f in glob.glob(f"{WORK}/runs/kaggle/weights/*.pt"):
    shutil.copy(f, f"{WORK}/{os.path.basename(f)}")
shutil.copy(f"{WORK}/runs/kaggle/results.csv", f"{WORK}/results.csv")
print("weights copied to /kaggle/working — download these from the notebook Output tab")

# ---- 5. report on BOTH val sets separately (the generalisation check) ----
m=YOLO(BEST)
for tag in ("val_ours","val_multi","val_combined"):
    b=m.val(data=f"{WORK}/{tag}/data.yaml",imgsz=IMGSZ,device=0,verbose=False,plots=False,
            split="val").box
    print(f"{tag:14} P {float(b.mp):.4f}  R {float(b.mr):.4f}  "
          f"mAP50 {float(b.map50):.4f}  mAP50-95 {float(b.map):.4f}")
