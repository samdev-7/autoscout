"""Per-checkpoint validation on DISJOINT held-out sets.

val_ours (our verified match, 83 frames) and val_multi (80 unseen events, 86 frames) share
no frames.  A checkpoint that leads on BOTH is genuinely better; one that leads only on a
single subset was picked by that subset's noise.  A single 169-frame number cannot
distinguish those two cases, which is exactly why best.pt being the epoch-15 peak is not yet
evidence that epoch 15 generalises.

val_combined (169 frames / 1,119 instances) is the union of the two and is the set the run
selected best.pt against, so it is NOT independent evidence -- it is reported here as the
selection baseline, to show how much of any lead is visible only on the selecting set.
Note mAP does not decompose as a weighted mean of subset mAPs (detections are pooled and
ranked globally), so it is measured directly rather than inferred from the other two.
"""
import glob, os, warnings
warnings.filterwarnings("ignore")
os.environ["YOLO_CONFIG_DIR"] = "/workspace/.ultralytics"
from ultralytics import YOLO

W = "/workspace/run/runs/kaggle/weights"
def epnum(p):
    d = "".join(c for c in os.path.basename(p) if c.isdigit())
    return int(d) if d else -1

cks = [f"{W}/best.pt", f"{W}/last.pt"] + sorted(glob.glob(f"{W}/epoch*.pt"), key=epnum)
hdr = (f"{'checkpoint':>12} | {'ours mAP50':>10} {'ours 5095':>10} | "
       f"{'multi mAP50':>11} {'multi 5095':>11} | {'comb mAP50':>10} {'comb 5095':>10}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

for c in cks:
    vals = []
    for tag in ("val_ours", "val_multi", "val_combined"):
        try:
            b = YOLO(c).val(data=f"/workspace/run/{tag}/data.yaml", imgsz=960, device=0,
                            verbose=False, plots=False, split="val").box
            vals += [float(b.map50), float(b.map)]
        except Exception as e:
            vals += [float("nan")] * 2
            print(f"ERR {os.path.basename(c)} {tag}: {e}", flush=True)
    print(f"{os.path.basename(c):>12} | {vals[0]:10.4f} {vals[1]:10.4f} | "
          f"{vals[2]:11.4f} {vals[3]:11.4f} | {vals[4]:10.4f} {vals[5]:10.4f}", flush=True)
